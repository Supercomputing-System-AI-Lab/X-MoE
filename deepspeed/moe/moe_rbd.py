from deepspeed.utils.timer import SynchronizedWallClockTimer
from deepspeed.utils import logger
from deepspeed.utils.bwc import bwc_tensor_model_parallel_world_size
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple, Union

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
from .v2opt.utils import remove_zero_rows, restore_zero_rows, compare_tensors, compare_uneven_and_padded, print_rank
from .v2opt.rbd import RBDispatcher, RBCombiner
from .v2opt.metadata import rbd_metadata

from .v2opt.a2a_single import _AllToAllSingle


if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

from deepspeed import comm as dist

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
    softmax_before_topk: bool = False
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements TopKGating on logits."""

    # everything is in fp32 in this function
    # get topk gates
    _, n_experts = logits.shape

    if softmax_before_topk:
        scores = F.softmax(logits, dim=1)
        top_gate, top_idx = torch.topk(scores, k=k, dim=1) # shape: (n_tokens, n_experts)
    else:
        top_gate, top_idx = torch.topk(logits, k=k, dim=1) # shape: (n_tokens, n_experts)
        scores = F.softmax(logits, dim=1)

    mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, top_idx, 1)
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

class TopKGateRBD(Module):
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
                 use_rbd: bool = False,
                 ) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
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
        self.num_local_experts = self.num_experts // dist.get_world_size(ep_group)
        print_rank(self.num_local_experts, "number local experts")
        print_rank(dist.get_world_size(ep_group), "ep group size")

    def _set_local_group(self, local_group):
        self.local_group = local_group
        self.mesh_size = dist.get_world_size(local_group)
        print_rank(self.mesh_size, "mesh size")

    def forward(self,
                input: torch.Tensor,
                use_pft: bool = False) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore

        if self.wall_clock_breakdown:
            self.timers(TOPK_GATE_TIMER).start()

        input_fp32 = input.float()
        # input jittering
        if self.noisy_gate_policy == 'Jitter' and self.training:
            input_fp32 = multiplicative_jitter(input_fp32, device=input.device)
        logits = torch.nn.functional.linear(input_fp32, weight=self.wg.weight.float(), bias=None)

        gate_output = topkgating_unbalanced(logits, self.k,
                                self.capacity_factor if self.training else self.eval_capacity_factor,
                                self.min_capacity, self.drop_tokens, self.ep_group, use_pft=use_pft)

        # TODO: add rbd metadata

        l_aux, indices, bin_ids, bins, expert_weights, tokens_per_expert = gate_output
        permutation = torch.randperm(indices.shape[0], device='cuda', dtype=torch.int32)
        metadata = rbd_metadata(indices, bin_ids, self.num_experts, self.k, self.mesh_size, self.num_local_experts, self.ep_group, self.local_group, permutation)

        if self.wall_clock_breakdown:
            self.timers(TOPK_GATE_TIMER).stop()
            self.gate_time = self.timers(TOPK_GATE_TIMER).elapsed(reset=False)

        return gate_output, metadata

class MOEv2LayerRBD(Base):
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
        self.num_experts = num_local_experts * ep_size
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

        if num_local_experts > 1:
            print("uneven all-to-all for num_local_experts > 1 is not implemented")


    def _set_ep_group(self, ep_group):
        self.ep_group = ep_group
        self.gate._set_ep_group(ep_group)

    def _set_local_group(self, local_group):
        self.local_group = local_group
        self.mesh_size = dist.get_world_size(local_group)
        self.gate._set_local_group(local_group)

    def _set_rbd(self):
        self.dispatcher = RBDispatcher(self.num_experts, self.k, self.num_local_experts, self.mesh_size, self.ep_group, self.local_group)
        self.combiner = RBCombiner(self.num_experts, self.k, self.num_local_experts, self.mesh_size, self.ep_group, self.local_group)

    
    def get_ep_rank(self):
        return dist.get_rank() % self.ep_size

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:

        if self.wall_clock_breakdown:
            torch.distributed.barrier()
            self.timers(MOE_TIMER).start()

        # Implement Algorithm 2 from GShard paper.
        d_model = input[0].shape[-1]

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_input = input[0].reshape(-1, d_model)

        tensor_model_world_size = bwc_tensor_model_parallel_world_size(groups.mpu)
        if tensor_model_world_size > 1 and groups._get_expert_model_parallel_world_size() == 1:
            orig_shape = reshaped_input.shape
            reshaped_input = drop_tokens(reshaped_input, dim=0)
            assert reshaped_input.shape[0] == orig_shape[0] // 2 and reshaped_input.shape[1] == orig_shape[1]

        n_tokens = reshaped_input.shape[0]

        (self.l_aux, indices, bin_ids, bins, expert_weights, tokens_per_experts), metadata = self.gate(reshaped_input, use_pft=True)
        self.exp_counts = tokens_per_experts

        dx_exp = self.dispatcher(reshaped_input, n_tokens, metadata) 

        if self.wall_clock_breakdown:
            torch.distributed.barrier()
            self.timers(DISPATCH_TIMER).stop()
            self.time_dispatch = self.timers(DISPATCH_TIMER).elapsed(reset=False)
        
        tokens_per_expert_s1_exp = metadata[-2]
        tokens_per_experts_s2_exp = metadata[-1]

        splits = tokens_per_expert_s1_exp.view(-1, self.num_local_experts).sum(dim=0) + tokens_per_experts_s2_exp.view(-1, self.num_local_experts).sum(dim=0)
        # splits = splits.tolist()

        assert sum(splits) == dx_exp.shape[0], "sum of local splits != dispatched output shape"

        if self.wall_clock_breakdown:
            torch.distributed.barrier()
            self.timers(EXPERTS_TIMER).start()

        dout_exp = self.experts(dx_exp, splits)

        if self.wall_clock_breakdown:
            torch.distributed.barrier()
            self.timers(EXPERTS_TIMER).stop()
            self.time_experts = self.timers(EXPERTS_TIMER).elapsed(reset=False)


        if tensor_model_world_size > 1 and groups._get_expert_model_parallel_world_size() > 1:
            # the dropped duplicate tokens need to be gathered on each
            # tensor parallel rank again for the tensor-parallel
            # non-expert of the next layer.
            expert_output = gather_tokens(expert_output, dim=1)

        if self.wall_clock_breakdown:
            torch.distributed.barrier()
            self.timers(COMBINE_TIMER).start()

        out = self.combiner(dout_exp, expert_weights, n_tokens, metadata)

        if self.wall_clock_breakdown:
            torch.distributed.barrier()
            self.timers(COMBINE_TIMER).stop()
            self.time_combine = self.timers(COMBINE_TIMER).elapsed(reset=False)

        if tensor_model_world_size > 1 and groups._get_expert_model_parallel_world_size() == 1:
            # the dropped duplicate tokens need to be gathered on each
            # tensor parallel rank again for the tensor-parallel
            # non-expert of the next layer.
            out = gather_tokens(out, dim=0)
            assert out.shape == orig_shape, f"out.shape is {out.shape}, but orig shape is {orig_shape}"

        a = out.reshape(input[0].shape)
        
        if self.wall_clock_breakdown:
            torch.distributed.barrier()
            self.timers(MOE_TIMER).stop()
            self.time_moe = self.timers(MOE_TIMER).elapsed(reset=False)

        return a