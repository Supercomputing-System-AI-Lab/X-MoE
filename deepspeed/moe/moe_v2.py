from deepspeed.utils.timer import SynchronizedWallClockTimer
from deepspeed.utils import logger
from deepspeed.utils.bwc import bwc_tensor_model_parallel_world_size
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple, Union

import os
import sys
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from deepspeed.utils import groups
from .mappings import drop_tokens, gather_tokens, USE_SP

import psutil
import gc
from deepspeed.accelerator import get_accelerator
torch_memory_reserved = get_accelerator().memory_reserved
torch_max_memory_reserved = get_accelerator().max_memory_reserved

from .v2opt.gating import gather, scatter, indices_and_bins
from .v2opt.drop_gating import gather_with_token_drop, scatter_with_token_drop, indices_and_bins_and_drop

from .sharded_moe import see_memory_usage, _AllToAll, multiplicative_jitter

from .sharded_moe import _capacity, _one_hot_to_float, einsum

from .v2opt.utils import remove_zero_rows, restore_zero_rows, compare_tensors, compare_uneven_and_padded

from .v2opt.rbd import RBDispatcher, RBCombiner
from torch.profiler import profile, record_function, ProfilerActivity, schedule

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

from deepspeed import comm as dist

from .v2opt.a2a_single import _AllToAllSingle

TOPK_GATE_TIMER = 'topk_gate'
MOE_TIMER = 'moe'
FIRST_ALLTOALL_TIMER = '1st_a2a'
SECOND_ALLTOALL_TIMER = '2nd_a2a'
DISPATCH_TIMER = 'dispatch_gemm'
COMBINE_TIMER = 'combine_gemm'
EXPERTS_TIMER = 'experts'

def topkgating_unbalanced(
    logits: Tensor,
    k: int,
    capacity_factor: float,
    min_capacity: int,
    drop_tokens: bool = True,
    ep_group: Union[torch.distributed.ProcessGroup, None] = None,
    drop_policy: str = "probs",
    use_pft: bool = False,
    softmax_before_topk: bool = False,
    score_function: str = "softmax",
    expert_bias: Union[Tensor, None] = None,
    n_group: int = 1,
    topk_group: int = 1,
    norm_topk_prob: bool = False,
    routed_scaling_factor: float = 1.0,
    seq_aux: bool = False,
    n_sequences: Union[int, None] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements TopKGating on logits.

    score_function "sigmoid" is the DeepSeek-V3 router (the released DeepseekV3TopkRouter): the affinity is the sigmoid of
    the logit; the choice adds the balancing bias (expert_bias, no gradient) and keeps the topk_group best of n_group expert
    groups (a group scores by the sum of its two best affinities); the weights are the unbiased affinities of the chosen
    experts, normalised to sum 1 when norm_topk_prob, times routed_scaling_factor. The default keeps the gate as it was.

    seq_aux (the released DeepSeek gate's config.json switch) makes the returned balance loss the sequence-wise one of the
    DeepSeek-V3 report (eq. 17-20): one loss per sequence, averaged over the n_sequences interleaved in the token axis (the b
    of Megatron's [s, b, h] layout: token t belongs to sequence t mod b). Off, the loss is the batch-wise one over all tokens.
    """

    # everything is in fp32 in this function
    # get topk gates
    _, n_experts = logits.shape

    if score_function == "sigmoid":
        scores = torch.sigmoid(logits)
        with torch.no_grad():
            choice = scores + expert_bias.unsqueeze(0).to(scores.dtype) if expert_bias is not None else scores
            group_scores = choice.view(-1, n_group, n_experts // n_group).topk(2, dim=-1)[0].sum(dim=-1)
            group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
            score_mask = group_mask.unsqueeze(-1).expand(-1, n_group, n_experts // n_group).reshape(-1, n_experts)
            choice = choice.masked_fill(~score_mask.bool(), 0.0)
            top_idx = torch.topk(choice, k=k, dim=1, sorted=False)[1]
        top_gate = scores.gather(1, top_idx)
        if norm_topk_prob:
            top_gate = top_gate / (top_gate.sum(dim=-1, keepdim=True) + 1e-20)
        top_gate = top_gate * routed_scaling_factor
    elif softmax_before_topk:
        scores = F.softmax(logits, dim=1)
        top_gate, top_idx = torch.topk(scores, k=k, dim=1) # shape: (n_tokens, n_experts)
    else:
        top_gate, top_idx = torch.topk(logits, k=k, dim=1) # shape: (n_tokens, n_experts)
        scores = F.softmax(logits, dim=1)

    mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, top_idx, 1)
    if seq_aux:
        # the sequence-wise balance loss (DeepSeek-V3 report eq. 17-20) per sequence: P_i = the mean over its tokens of the
        # normalised affinity s'_{i,t} = s_{i,t} / sum_j s_{j,t} (eq. 20; the identity for softmax scores), f_i = n_experts /
        # (k T) x the number of its tokens that chose expert i (the chosen set, before any capacity drop); loss = sum_i f_i P_i,
        # then the mean over the sequences. Under a TP token shard the caller scales by 1/TP as for the batch-wise loss.
        assert n_sequences is not None and scores.shape[0] % n_sequences == 0, \
            f"seq_aux needs the number of sequences interleaved in the {scores.shape[0]} tokens (got {n_sequences})"
        p_norm = scores / scores.sum(dim=1, keepdim=True)
        seq_p = p_norm.view(-1, n_sequences, n_experts).mean(dim=0)                               # [b, E]
        seq_f = mask.float().view(-1, n_sequences, n_experts).mean(dim=0) * (n_experts / k)     # [b, E], no gradient
        l_aux = (seq_f * seq_p).sum(dim=1).mean()
    else:
        me = torch.mean(scores, dim=0)
        ce = torch.mean(mask.float(), dim=0)
        l_aux = torch.mean(me * ce) * n_experts * n_experts / k

    expert_weights = top_gate.flatten()
    top_experts = top_idx.flatten()

    with torch.no_grad():
        if drop_tokens:
            capacity = _capacity(scores, torch.tensor(capacity_factor * k), torch.tensor(min_capacity))
            # print(f"capacity is {capacity}")
            indices, bin_ids, bins, tokens_per_expert = indices_and_bins_and_drop(n_experts, top_experts, expert_weights, capacity)
        else:
            indices, bin_ids, bins, tokens_per_expert = indices_and_bins(n_experts, top_experts)
            
    return l_aux, indices, bin_ids, bins, expert_weights, tokens_per_expert

class TopKGatev2(Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (int):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(self,
                 model_dim: int,
                 num_experts: int,
                 k: int = 1,
                 capacity_factor: float = 1.0,
                 eval_capacity_factor: float = 1.0,
                 min_capacity: int = 8,
                 noisy_gate_policy: Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts: bool = True,
                 ep_group: Union[torch.distributed.ProcessGroup, None] = None,
                 top2_2nd_expert_sampling: bool = True,
                 softmax_before_topk: bool = False,
                 score_function: str = "softmax",
                 bias_update_speed: float = 0.0,
                 n_group: int = 1,
                 topk_group: int = 1,
                 norm_topk_prob: bool = False,
                 routed_scaling_factor: float = 1.0,
                 seq_aux: bool = False,
                 ) -> None:
        super().__init__()

        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        # DeepSeek-V3 router fields (defaults = the gate as it was). With a bias update speed > 0 the gate carries the
        # auxiliary-loss-free balancing bias (a persistent fp32 buffer, no gradient) and an accumulator of the tokens each
        # expert received since the last optimizer step; the engine moves the bias after every step (_update_moe_router_bias).
        self.score_function = score_function
        self.bias_update_speed = float(bias_update_speed)
        self.n_group = int(n_group)
        self.topk_group = int(topk_group)
        self.norm_topk_prob = bool(norm_topk_prob)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.seq_aux = bool(seq_aux)      # the sequence-wise balance loss (DeepSeek-V3 report eq. 17-20) instead of the batch-wise one
        if self.bias_update_speed > 0.0:
            self.register_buffer("expert_bias", torch.zeros(num_experts, dtype=torch.float32))
            self.register_buffer("load_acc", torch.zeros(num_experts, dtype=torch.float32), persistent=False)
        # False (default) -> expert_weights are raw logits (X-MoE behavior).
        # True -> top-k over softmax probs, so expert_weights are probabilities, matching
        # HF DeepSeek's MoEGate. Needed to load HF MoE checkpoints.
        self.softmax_before_topk = softmax_before_topk
        self.ep_group = ep_group
        self.k = k
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.noisy_gate_policy = noisy_gate_policy
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False
        self.gate_time = 0.0
        self.drop_tokens = drop_tokens
        self.use_rts = use_rts
        self.top2_2nd_expert_sampling = top2_2nd_expert_sampling

    def _set_ep_group(self, ep_group):
        assert self.ep_group is None, f'Attempting to override an existing ep_group'
        self.ep_group = ep_group

    def _apply(self, fn, recurse=True):
        # The engine casts the whole module to bf16 (engine._configure_distributed_model: self.module.bfloat16()), buffers
        # included. The balancing bias and the load accumulator stay fp32 under that cast: the bias moves by the speed (0.001)
        # per step and a bf16 bias stops moving once |bias| reaches 0.25 (an ulp of 2^-9 there, 2^-8 at 0.5), and the loads are
        # token counts that bf16 rounds above 256. Found 2026-09-11 in run G's layer files (the bias saved as bf16).
        r = super()._apply(fn, recurse) if 'recurse' in super()._apply.__code__.co_varnames else super()._apply(fn)
        for name in ('expert_bias', 'load_acc'):
            buf = self._buffers.get(name)
            if buf is not None and buf.dtype != torch.float32:
                self._buffers[name] = buf.float()
        return r

    def forward(self,
                input: torch.Tensor,
                use_pft: bool = False,
                n_sequences: Union[int, None] = None) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
        # n_sequences: the sequences interleaved in the token axis (the b of [s, b, h]); only the sequence-wise loss (seq_aux) reads it

        if self.wall_clock_breakdown:
            self.timers(TOPK_GATE_TIMER).start()

        input_fp32 = input.float()
        # input jittering
        if self.noisy_gate_policy == 'Jitter' and self.training:
            input_fp32 = multiplicative_jitter(input_fp32, device=input.device)
        logits = torch.nn.functional.linear(input_fp32, weight=self.wg.weight.float(), bias=None)

        gate_output = topkgating_unbalanced(logits, self.k,
                                self.capacity_factor if self.training else self.eval_capacity_factor,
                                self.min_capacity, self.drop_tokens, self.ep_group, use_pft=use_pft,
                                softmax_before_topk=self.softmax_before_topk,
                                score_function=self.score_function,
                                expert_bias=getattr(self, "expert_bias", None),
                                n_group=self.n_group, topk_group=self.topk_group,
                                norm_topk_prob=self.norm_topk_prob,
                                routed_scaling_factor=self.routed_scaling_factor,
                                seq_aux=self.seq_aux, n_sequences=n_sequences)
        if self.bias_update_speed > 0.0 and self.training:
            # the tokens each expert received in this micro-batch (the 6th return, tokens_per_expert), summed until the step
            with torch.no_grad():
                self.load_acc += gate_output[5].to(self.load_acc.dtype)

        if self.wall_clock_breakdown:
            self.timers(TOPK_GATE_TIMER).stop()
            self.gate_time = self.timers(TOPK_GATE_TIMER).elapsed(reset=False)

        return gate_output

class MOEv2Layer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    def __init__(self,
                 gate: Module,
                 experts: Module,
                 ep_group_name,
                 ep_size,
                 num_local_experts: int,
                 k: int,
                 num_shared_experts: int = 0,
                 use_pft: bool = False,
                 drop_tokens: bool = True) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.ep_group = None
        self.ep_size = ep_size
        self.ep_group_name = ep_group_name
        self.num_local_experts = num_local_experts
        self.k = k
        self.num_shared_experts = num_shared_experts
        self.time_falltoall = 0.0
        self.time_salltoall = 0.0
        self.time_moe = 0.0
        self.time_experts = 0.0
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False

        self.use_pft = use_pft
        self.drop_tokens = drop_tokens
        
        # print (f'{dist.get_rank()=}, {self.ep_group=}, {self.ep_size=}, {self.ep_group_name=}, {self.num_local_experts=}, {self.num_shared_experts}, {self.experts=}')

    def _set_ep_group(self, ep_group):
        self.ep_group = ep_group
        self.gate._set_ep_group(ep_group)

    
    def get_ep_rank(self):
        return dist.get_rank() // bwc_tensor_model_parallel_world_size(groups.mpu) % self.ep_size 

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:

        # print (f'inside moe forward')
        import os
        rank = os.getenv ('RANK')
        self.wall_clock_breakdown = os.getenv ("WALL_CLOCK_BREAKDOWN")=="true"
        # self.wall_clock_breakdown = False
        # self.wall_clock_breakdown = True 
        # print (f'[moe_v2 MOEv2Layer forward]')

        if self.wall_clock_breakdown:
            torch.distributed.barrier()
            self.timers(MOE_TIMER).start()

        with record_function("MoE - Gating & Reshaping"):
            d_model = input[0].shape[-1]
            reshaped_input = input[0].reshape(-1, d_model)
            tensor_model_world_size = bwc_tensor_model_parallel_world_size(groups.mpu)

            # sequence-sharded MoE block: drop tokens at the beginning of the sparse MoE layer.
            moe_sequence_sharded = tensor_model_world_size > 1 and groups._get_expert_model_parallel_world_size() == 1
            # Megatron TP-companion --sequence-parallel (XMOE_MEGATRON_SP=1,
            # exported alongside the flag): the layer input is ALREADY this
            # rank's sequence shard ([s/TP * b, d]), so narrowing again would
            # keep chunk r OF shard r and the exit gather would rebuild the
            # sequence out of other ranks' tokens (shapes still match --
            # silent). Skip the entry narrow and the exit gather; the gate
            # still sees a 1/TP token shard either way, so the l_aux / TP
            # scaling below and the engine's gate TP grad sum stay keyed on
            # moe_sequence_sharded exactly as landed.
            megatron_sequence_parallel = moe_sequence_sharded and os.getenv('XMOE_MEGATRON_SP') == '1'
            pad_rows = 0
            if moe_sequence_sharded and not megatron_sequence_parallel:
                orig_shape = reshaped_input.shape
                # [2026-09-12] a token count that TP does not divide (a decode step of a batch smaller than TP, the last batch of
                # an evaluation; a training batch never takes this branch): pad with zero rows to the next multiple, shard, and
                # cut the padding off after the exit gather. The padded rows route like any token and are discarded; the
                # dispatch drops no real token, so the real rows are unchanged (tested on two gloo ranks).
                rem = orig_shape[0] % tensor_model_world_size
                if rem:
                    pad_rows = tensor_model_world_size - rem
                    reshaped_input = torch.cat([reshaped_input, reshaped_input.new_zeros(pad_rows, orig_shape[1])], dim=0)
                    orig_shape = reshaped_input.shape
                reshaped_input = drop_tokens(reshaped_input, dim=0)
                assert reshaped_input.shape[0] == orig_shape[0] // tensor_model_world_size and reshaped_input.shape[1] == orig_shape[1]

            n_tokens = reshaped_input.shape[0]

            if self.use_pft:
                # n_sequences: the b of the layer's [s, b, h] input (a TP shard keeps that structure); read only by seq_aux
                self.l_aux, indices, bin_ids, bins, expert_weights, input_splits_tensor = self.gate(
                    reshaped_input, use_pft=True, n_sequences=(input[0].shape[1] if input[0].dim() == 3 else None))
                if moe_sequence_sharded:
                    # The gate ran on a 1/TP token shard, so every TP rank
                    # contributes its own shard-local l_aux to the loss and the
                    # gate gradients are summed across the TP group. Scale by
                    # 1/TP to keep the effective auxiliary loss at the TP=1
                    # magnitude.
                    self.l_aux = self.l_aux / tensor_model_world_size

                
                if self.wall_clock_breakdown:
                    torch.distributed.barrier()
                    self.timers(DISPATCH_TIMER).start()

                if self.drop_tokens:
                    flattened_input = gather_with_token_drop(reshaped_input, indices, bin_ids, bins, n_tokens, self.k)
                else:
                    flattened_input = gather(reshaped_input, indices, bin_ids, bins, self.k)

                if self.wall_clock_breakdown:
                    torch.distributed.barrier()
                    self.timers(DISPATCH_TIMER).stop()
                    self.time_dispatch = self.timers(DISPATCH_TIMER).elapsed(reset=False)

                self.exp_counts = input_splits_tensor
            else:
                self.l_aux, combine_weights, dispatch_mask, self.exp_counts = self.gate(reshaped_input, input[1])
                if moe_sequence_sharded:
                    # Same 1/TP scale as the use_pft branch above.
                    self.l_aux = self.l_aux / tensor_model_world_size
                if self.wall_clock_breakdown:
                    torch.distributed.barrier()
                    self.timers(DISPATCH_TIMER).start()
                dispatched_input = einsum("sec,sm->ecm", dispatch_mask.type_as(input[0]), reshaped_input)
                flattened_input, input_splits_tensor, padding_mask = remove_zero_rows(dispatched_input)
                if self.wall_clock_breakdown:
                    torch.distributed.barrier()
                    self.timers(DISPATCH_TIMER).stop()
                    self.time_dispatch = self.timers(DISPATCH_TIMER).elapsed(reset=False)
        
        with record_function("MoE - Dispatch A2A - 1"):
            output_splits_tensor = _AllToAll.apply(self.ep_group, input_splits_tensor)
            input_splits_tensor_ep = input_splits_tensor.view(-1, self.num_local_experts).sum(dim=1)
            output_splits_tensor_ep = output_splits_tensor.view(-1, self.num_local_experts).sum(dim=1)

            input_splits = input_splits_tensor_ep.tolist()
            output_splits = output_splits_tensor_ep.tolist()

        assert sum(input_splits) == flattened_input.shape[0], f"input_split sum {sum(input_splits)} != input shape [0] {flattened_input.shape[0]} on rank {dist.get_rank()}"

        with record_function("MoE - Dispatch A2A - 2"):
            if self.wall_clock_breakdown:
                torch.distributed.barrier()
                self.timers(FIRST_ALLTOALL_TIMER).start()
                
            # print (f'[moe_v2 MOEv2Layer forward before _AllToAllSingle.apply]')

            # dispatch all-to-all    
            dispatched_output = _AllToAllSingle.apply(self.ep_group, flattened_input, input_splits, output_splits)

            if self.wall_clock_breakdown:
                torch.distributed.barrier()
                self.timers(FIRST_ALLTOALL_TIMER).stop()
                self.time_falltoall = self.timers(FIRST_ALLTOALL_TIMER).elapsed(reset=False)


        # splits = output_splits_tensor.view(-1, self.num_local_experts).sum(dim=0).tolist() 
        # assert sum(splits) == dispatched_output.shape[0], "sum of local splits != dispatched output shape"
        
        with record_function("MoE - Expert & Regrouping"):
            if self.wall_clock_breakdown:
                # torch.distributed.barrier()
                torch.cuda.synchronize()
                self.timers(EXPERTS_TIMER).start()

            # expert_output_uneven = self.experts(dispatched_output, splits)

            ##################
            # print (f'[moe_v2.py] BEFORE EXPERT {dispatched_output.shape=}')
            expert_output_uneven = self.experts(dispatched_output, output_splits_tensor)
            # print (f'[moe_v2.py] AFTER EXPERT {expert_output_uneven.shape=}')
        
            from .experts import FusedExperts_Triton   # its kernel stores each output row at its input index: rows already rank-major
            # the per-expert loop (Experts) and FusedExperts_Primus return rows expert-major: reorder them to rank-major for the combine
            if not isinstance(self.experts, FusedExperts_Triton):
                M = output_splits_tensor.shape[0]
                N = self.num_local_experts
                K = M // N
                assert M % N == 0
                output_splits_tensor_interleaved = output_splits_tensor.reshape(K, N, -1).transpose(0, 1).reshape(M)
                expert_output_uneven_interleaved = torch.split(expert_output_uneven, output_splits_tensor_interleaved.tolist(), dim=0)
                expert_output_uneven_interleaved = torch.cat([
                    torch.cat(expert_output_uneven_interleaved[idx::K], dim=0)
                    for idx in range(K)
                ], dim=0)
            else:
                expert_output_uneven_interleaved = expert_output_uneven   # rank-major already; a reorder here would permute the rows silently
    
            if self.wall_clock_breakdown:
                # torch.distributed.barrier()
                torch.cuda.synchronize()
                self.timers(EXPERTS_TIMER).stop()
                torch.distributed.barrier()
                self.time_experts = self.timers(EXPERTS_TIMER).elapsed(reset=False)
                self.timers(SECOND_ALLTOALL_TIMER).start()
            
            # print (f'[moe_v2.py] {expert_output_uneven_interleaved.shape=}')
            
            # combine all-to-all
            with record_function ("MoE - Combine A2A"): 
                expert_output_uneven = _AllToAllSingle.apply(self.ep_group, expert_output_uneven_interleaved, output_splits, input_splits)
        
            # # If using triton groupgemm 
            # # Zixian: TODO: change the following if to use "use_triton" and "use_groupgemm" 
            # if os.getenv ("USE_TRITON_GROUPGEMM") == 'True': 
            #     # print ("USING TRITON GROUPGEMM")
            #     if self.wall_clock_breakdown:
            #         # torch.distributed.barrier()
            #         torch.cuda.synchronize()
            #         self.timers(EXPERTS_TIMER).stop()
            #         torch.distributed.barrier()
            #         self.time_experts = self.timers(EXPERTS_TIMER).elapsed(reset=False)
            #         self.timers(SECOND_ALLTOALL_TIMER).start()
                    
            #     # print (f'[moe_v2.py] {expert_output_uneven.shape=}')
            #     # combine all-to-all
            #     with record_function ("MoE - Combine A2A"): 
            #         # Feed `expert_output_uneven` directly back into the network!
            #         expert_output_uneven = _AllToAllSingle.apply(self.ep_group, expert_output_uneven, output_splits, input_splits)
        ##################


        #%%%%%%
        #expert_output_uneven = _AllToAllSingle.apply(self.ep_group, expert_output_uneven, output_splits, input_splits)
        #%%%%%%
        
        if self.wall_clock_breakdown:
            torch.distributed.barrier()
            self.timers(SECOND_ALLTOALL_TIMER).stop()
            self.time_salltoall = self.timers(SECOND_ALLTOALL_TIMER).elapsed(reset=False)
                
        with record_function ("MoE - Regrouping"): 
            
            if self.use_pft:
                expert_output = expert_output_uneven
            else:
                expert_output = restore_zero_rows(expert_output_uneven, padding_mask)

            if self.wall_clock_breakdown:
                torch.distributed.barrier()
                self.timers(COMBINE_TIMER).start()

            if self.use_pft:
                if self.drop_tokens:
                    combined_output = scatter_with_token_drop(expert_output, indices, bin_ids, expert_weights, bins, n_tokens, self.k)
                else:
                    combined_output = scatter(expert_output, indices, bin_ids, expert_weights, bins, self.k)
            else:
                combined_output = einsum("sec,ecm->sm", combine_weights.type_as(input[0]), expert_output)


            if self.wall_clock_breakdown:
                torch.distributed.barrier()
                self.timers(COMBINE_TIMER).stop()
                self.time_combine = self.timers(COMBINE_TIMER).elapsed(reset=False)

            # sequence-sharded MoE block: gather tokens at the end of the sparse MoE layer.
            # (Skipped under Megatron companion sequence parallelism: the
            # output must stay this rank's sequence shard -- see the entry
            # guard above.)
            if moe_sequence_sharded and not megatron_sequence_parallel:
                combined_output = gather_tokens(combined_output, dim=0)
                assert combined_output.shape == orig_shape
                if pad_rows:
                    combined_output = combined_output[:-pad_rows]

            a = combined_output.reshape(input[0].shape)
            
            if (os.getenv ('DEBUG_ACTIVATION_OUTPUT') == 1): 
                print (f'[moe_v2.py] AFTER COMBINE OUT {a.shape=} \n{a[0][:2]=}')

            
            if self.wall_clock_breakdown:
                torch.distributed.barrier()
                self.timers(MOE_TIMER).stop()
                self.time_moe = self.timers(MOE_TIMER).elapsed(reset=False)

        return a