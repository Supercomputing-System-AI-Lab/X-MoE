# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
The file has been adapted from two fairscale files:
 (1) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/moe_layer.py
 (2) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/top2gate.py
 Git commit hash: 34df606902a240567a0d898037ece55c2f1336cf
 We retain the following license from the original files:
"""

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

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

RANDOM_VERIFY = False

# torch.set_printoptions(threshold=float('inf'))

def see_memory_usage(message, force=False):
    if not force:
        return
    if dist.is_initialized() and not dist.get_rank() == 0:
        return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    # Print message except when distributed but not rank 0
    logger.info(message)
    logger.info(f"MA {round(get_accelerator().memory_allocated() / (1024 * 1024 * 1024),2 )} GB \
        Max_MA {round(get_accelerator().max_memory_allocated() / (1024 * 1024 * 1024),2)} GB \
        CA {round(torch_memory_reserved() / (1024 * 1024 * 1024),2)} GB \
        Max_CA {round(torch_max_memory_reserved() / (1024 * 1024 * 1024))} GB ")

    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
    logger.info(f'CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%')

    # get the peak memory to report correct data, so reset the counter for the next call
    get_accelerator().reset_peak_memory_stats()


if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

TOPK_GATE_TIMER = 'topk_gate'
MOE_TIMER = 'moe'
FIRST_ALLTOALL_TIMER = '1st_a2a'
SECOND_ALLTOALL_TIMER = '2nd_a2a'
DISPATCH_TIMER = 'dispatch_gemm'
COMBINE_TIMER = 'combine_gemm'
EXPERTS_TIMER = 'experts'

uniform_map: Dict[torch.device, Callable] = {}
gumbel_map: Dict[torch.device, Callable] = {}
exp_selection_uniform_map: Dict[torch.device, Callable] = {}

try:
    # To enable Tutel MoE optimizations:
    #   python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x
    from tutel import moe as tutel_moe
    TUTEL_INSTALLED = True
except:
    # Fail silently so we don't spam logs unnecessarily if user isn't using tutel
    TUTEL_INSTALLED = False
    pass


def multiplicative_jitter(x, device: torch.device, epsilon=1e-2):
    """
    Modified from switch transformer paper. mesh transformers
    Multiply values by a random number between 1-epsilon and 1+epsilon.
    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.
    Args:
        x: a torch.tensor
        device: torch.device
        epsilon: a floating point value
    Returns:
        a jittered x.
    """
    if epsilon == 0:
        return x
    uniform = uniform_map.get(device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(low=torch.tensor(1.0 - epsilon, device=device),
                                                      high=torch.tensor(1.0 + epsilon,
                                                                        device=device)).rsample  # type: ignore
        uniform_map[device] = uniform
    return x * uniform(x.shape)


def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


from deepspeed import comm as dist

# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.


# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))

class _AllToAllSingle(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor, input_splits: list, output_splits: list) -> Tensor:  # type: ignore

        ctx.group = group
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        input = input.contiguous()
        total_recv = sum(output_splits)
        output = torch.empty(total_recv, input.shape[-1], device=input.device, dtype=input.dtype)

        world_size = dist.get_world_size()
        rank = dist.get_rank() % dist.get_world_size(group)

        # detect_all_0_flags = torch.zeros(world_size, device='cuda', dtype=torch.int)
        # detect_all_0_flags_o = torch.zeros(world_size, device='cuda', dtype=torch.int)

        input_splits_0_flag = False
        output_splits_0_flag = False

        if sum(input_splits) == 0:
            print("!!! detect all 0 input", flush=True)
            input = torch.zeros((1, input.shape[-1]), device='cuda', dtype=input.dtype)
            temp_input_splits = input_splits.copy()
            temp_output_splits = output_splits.copy()
            temp_input_splits[rank] = 1
            temp_output_splits[rank] = 1
            extra_zeros = torch.zeros((1, output.shape[-1]), device='cuda', dtype=output.dtype)
            output = torch.cat([output, extra_zeros], dim=0)
            _position = sum(output_splits[:rank])
            input_splits_0_flag = True
        
        if sum(output_splits) == 0:
            print("!!! detect all 0 output", flush=True)
            _output = output
            temp_input_splits = input_splits.copy()
            temp_output_splits = output_splits.copy()
            temp_input_splits[rank] = 1
            temp_output_splits[rank] = 1
            extra_zeros = torch.zeros((1, output.shape[-1]), device='cuda', dtype=output.dtype)
            _position = sum(input_splits[:rank])
            input = torch.cat((input[:_position], extra_zeros, input[_position:]), dim=0)
            output = torch.zeros((1, input.shape[-1]), device='cuda', dtype=input.dtype)
            output_splits_0_flag = True

        # if sum(input_splits) == 0:
        #     print("!!! detect all 0 input", flush=True)
        #     detect_all_0_flags[rank] = 1
        #     input = torch.zeros((1, input.shape[-1]), device='cuda', dtype=input.dtype)
        #     input_splits[dist.get_world_size() - 1] = 1
        
        # if sum(output_splits) == 0:
        #     print("!!! detect all 0 output", flush=True)
        #     detect_all_0_flags_o[rank] = 1
        #     output = torch.zeros((1, input.shape[-1]), device='cuda', dtype=input.dtype)
        #     output_splits[dist.get_world_size() - 1] = 1

        # dist.all_reduce(detect_all_0_flags, op=dist.ReduceOp.SUM, group=group)
        # dist.all_reduce(detect_all_0_flags_o, op=dist.ReduceOp.SUM, group=group)

        # count_all_0 = int(detect_all_0_flags.sum())
        # count_all_0_o = int(detect_all_0_flags_o.sum())

        # if count_all_0 > 0 and rank == world_size - 1:
        #     print("!!! count_all_0 > 0", flush=True)
        #     print(f"old output_splits:", output_splits)
        #     output_splits_tensor = torch.tensor(output_splits, device='cuda', dtype=torch.int)
        #     orig_len = output.shape[0]
        #     extra_zeros = torch.zeros((count_all_0, output.shape[-1]), device='cuda', dtype=output.dtype)
        #     output = torch.cat([output, extra_zeros], dim=0)
        #     output_splits_tensor = output_splits_tensor + detect_all_0_flags
        #     output_splits = output_splits_tensor.tolist()
        #     print(f"new output_splits:", output_splits)

        # if count_all_0_o > 0 and rank == world_size - 1:
        #     print("!!! count_all_0_i > 0", flush=True)
        #     print(f"old input_splits:", input_splits)
        #     input_splits_tensor = torch.tensor(input_splits, device='cuda', dtype=torch.int)
        #     orig_len = input.shape[0]
        #     extra_zeros = torch.zeros((count_all_0_o, input.shape[-1]), device='cuda', dtype=input.dtype)
        #     input = torch.cat([input, extra_zeros], dim=0)
        #     input_splits_tensor = input_splits_tensor + detect_all_0_flags_o
        #     input_splits = input_splits_tensor.tolist()
        #     print(f"new input_splits:", input_splits)

        # torch.distributed.barrier()
        # if dist.get_rank() == 0:
        #     print("         before all_to_all_single", flush=True)
        # print(f"                  rank [{dist.get_rank()}], input_splits: {input_splits}", flush=True)
        # print(f"                  rank [{dist.get_rank()}], output_splits: {output_splits}", flush=True)
        # print(f"                  rank [{dist.get_rank()}], input_shape: {input.shape}", flush=True)
        # print(f"                  rank [{dist.get_rank()}], output_shape: {output.shape}", flush=True)
        # max_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        # max_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
        # torch.distributed.barrier()
        # print(f"        Rank {dist.get_rank()}: Max Memory Allocated: {max_allocated_mb:.2f} MB, Max Memory Reserved: {max_reserved_mb:.2f} MB", flush=True)
        # torch.distributed.barrier()
        # print(f"                  rank [{dist.get_rank()}] input:", input)


        
        if input_splits_0_flag or output_splits_0_flag:
            dist.all_to_all_single(output, input, output_split_sizes=temp_output_splits, input_split_sizes=temp_input_splits, group=group)
        else:
            dist.all_to_all_single(output, input, output_split_sizes=output_splits, input_split_sizes=input_splits, group=group)
        # if count_all_0 > 0 and rank == world_size - 1:
        #     output = output[:orig_len]

        if input_splits_0_flag:
            output = torch.cat((output[:_position], output[_position+1:]), dim=0)
            # print("recovered shape:", output.shape)
        
        if output_splits_0_flag:
            output = _output

        # torch.distributed.barrier()
        # if dist.get_rank() == 0:
        #     print("         after all_to_all_single", flush=True)
        # max_allocated_mb_2 = torch.cuda.max_memory_allocated() / (1024 ** 2)
        # max_reserved_mb_2 = torch.cuda.max_memory_reserved() / (1024 ** 2)
        # if max_allocated_mb_2 != max_allocated_mb:
        #     print(f"          Rank {dist.get_rank()}: max allocated memory: {max_allocated_mb:.2f} MB -> {max_allocated_mb_2:.2f} MB")
        # if max_reserved_mb != max_reserved_mb_2:
        #     print(f"          Rank {dist.get_rank()}: max reserved memory: {max_reserved_mb:.2f} MB -> {max_reserved_mb_2:.2f} MB")

        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAllSingle.apply(ctx.group, *grad_output, ctx.output_splits, ctx.input_splits), None, None)

# einsum rewrites are on par or more performant
# switch can be bubbled up in future
USE_EINSUM = True


# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.
def einsum(rule, a, b):
    if USE_EINSUM:
        return torch.einsum(rule, a, b)
    elif rule == 's,se->se':
        return a.reshape(a.shape[0], -1) * b
    elif rule == 'se,sc->sec':
        return a.unsqueeze(2) * b.unsqueeze(1)
    elif rule == 'se,se->s':
        return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
    elif rule == 'se,sec->sec':
        return a.unsqueeze(2) * b
    elif rule == 'sec,sm->ecm':
        s = a.shape[0]
        e = a.shape[1]
        c = a.shape[2]
        m = b.shape[1]
        return torch.matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)
    elif rule == 'sec,ecm->sm':
        return torch.matmul(a.reshape(a.shape[0], -1), b.reshape(-1, b.shape[-1]))
    elif rule == 'ks,ksm->sm':
        k = b.shape[0]
        s = b.shape[1]
        m = b.shape[2]
        # [k, s] -> [s, k] -> [s, 1, k]
        a = a.t().unsqueeze(1)
        # [k,s,m] -> [k, sm] -> [sm, k] -> [s, m, k]
        b = b.reshape(k, -1).t().reshape(s, m, k)
        # bmm([s, 1, k], [s, m, k]^t) -> [s, m, 1]
        return torch.bmm(a, b.transpose(1, 2)).squeeze(2)
    else:
        return torch.einsum(rule, a, b)


# The following functions are extracted and scripted
# because otherwise during a torch.jit.trace, the non-Tensor
# values used in the calculations get recorded as constants.
# torch.jit.script coerces them into Tensors and preserves
# their dynamic shapes. This enables ONNX export.
# We can't script the entire top1gating function because it
# includes stateful caching logic which is incompatible with ONNX.


@torch.jit.script
def _capacity(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity


@torch.jit.script
def _top_idx(source, k):
    return torch.topk(source, k=k, dim=0)[1]


@torch.jit.script
def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()


def top1gating(logits: Tensor,
               capacity_factor: float,
               min_capacity: int,
               used_token: Tensor = None,
               noisy_gate_policy: Optional[str] = None,
               drop_tokens: bool = True,
               use_rts: bool = True,
               ep_group: Union[torch.distributed.ProcessGroup, None] = None,
               use_tutel: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top1Gating on logits."""
    if noisy_gate_policy == 'RSample':
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # everything is in fp32 in this function

    gates = F.softmax(logits, dim=1)
    capacity = _capacity(gates, torch.tensor(capacity_factor), torch.tensor(min_capacity))

    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = torch.argmax(logits_w_noise if noisy_gate_policy == 'RSample' else gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # mask only used tokens
    if used_token is not None:
        mask1 = einsum("s,se->se", used_token, mask1)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to(logits.device)

    # if we don't want to drop any tokens
    if not drop_tokens:
        new_capacity = torch.max(exp_counts).to(logits.device)
        # Communicate across expert processes to pick the maximum capacity.
        if ep_group is not None:
            dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=ep_group)
        if groups._get_expert_model_parallel_world_size() == 1:
            # If the non-expert is tensor-parallel, we need to pad the capacity to 'tp'.
            # This is since we are going to activate drop_tokens() to drop duplicate tokens.
            tp = 1 if groups.mpu is None else bwc_tensor_model_parallel_world_size(mpu=groups.mpu)
            new_capacity = torch.ceil(new_capacity / tp).mul(tp).to(new_capacity.dtype)
        # Make sure the capacity value does not exceed the number of tokens.
        capacity = min(new_capacity, torch.tensor(mask1.size(0)).to(new_capacity.device))

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.sum(me * ce) * num_experts

    # Random Token Selection
    if use_rts:
        uniform = exp_selection_uniform_map.get(logits.device)
        if uniform is None:
            uniform = torch.distributions.uniform.Uniform(low=torch.tensor(0.0, device=logits.device),
                                                          high=torch.tensor(1.0, device=logits.device)).rsample
            exp_selection_uniform_map[logits.device] = uniform

        mask1_rand = mask1 * uniform(mask1.shape)
    else:
        mask1_rand = mask1

    assert logits.shape[
        0] >= min_capacity, "No. of tokens (batch-size) should be greater than min_capacity. Either set min_capacity to 0 or increase your batch size."

    top_idx = _top_idx(mask1_rand, capacity)

    new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)
    mask1 = new_mask1

    if use_tutel:
        # Tutel doesn't support index values masked with zero
        # so we need to replace masked indices with -1
        indices_mask = mask1.sum(dim=1) * num_experts - 1
        indices1_s = torch.min(indices1_s, indices_mask)

    # Compute locations in capacity buffer
    if use_tutel:
        locations1 = tutel_moe.fast_cumsum_sub_one(mask1)
    else:
        locations1 = torch.cumsum(mask1, dim=0) - 1

    if use_tutel:
        gates1_s = (gates * mask1).sum(dim=1)
        locations1_s = torch.sum(locations1 * mask1, dim=1)
        return l_aux, capacity, num_experts, [
            indices1_s,
        ], [
            locations1_s,
        ], [
            gates1_s,
        ], exp_counts

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    gates = gates * mask1_float

    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    combine_weights = einsum("se,sc->sec", gates, locations1_sc)

    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts


def top2gating(logits: Tensor,
               capacity_factor: float,
               min_capacity: int,
               drop_tokens: bool = True,
               ep_group: Union[torch.distributed.ProcessGroup, None] = None,
               top2_2nd_expert_sampling: bool = True) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    if top2_2nd_expert_sampling:
        # Create a mask for 2nd's expert per token using Gumbel-max trick
        # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
        logits += gumbel_rsample(logits.shape, device=logits.device)

    # Replace top-expert with min value
    logits_except1 = logits.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts

    # gating decisions
    exp_counts = torch.sum(mask1 + mask2, dim=0).detach().to(logits.device)

    if drop_tokens:
        # Calculate configured capacity and remove locations outside capacity from mask
        capacity = _capacity(gates, torch.tensor(capacity_factor * 2), torch.tensor(min_capacity))
        mask1 *= torch.lt(locations1, capacity)
        mask2 *= torch.lt(locations2, capacity)
    else:
        # Do not drop tokens - set capacity according to current expert assignments
        new_capacity = torch.max(exp_counts)
        if ep_group is not None:
            dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=ep_group)
        if groups._get_expert_model_parallel_world_size() == 1:
            # If the non-expert is tensor-parallel, we need to pad the capacity to 'tp'.
            # This is since we are going to activate drop_tokens() to drop duplicate tokens.
            tp = 1 if groups.mpu is None else bwc_tensor_model_parallel_world_size(mpu=groups.mpu)
            new_capacity = torch.ceil(new_capacity / tp).mul(tp).to(new_capacity.dtype)
        capacity = new_capacity

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    mask2_float = mask2.float()
    gates1_s = einsum("se,se->s", gates, mask1_float)
    gates2_s = einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    gates1 = einsum("s,se->se", gates1_s, mask1_float)
    gates2 = einsum("s,se->se", gates2_s, mask2_float)
    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    locations2_sc = _one_hot_to_float(locations2_s, capacity)
    combine1_sec = einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = einsum("se,sc->sec", gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts


def topkgating(
    logits: Tensor,
    k: int,
    capacity_factor: float,
    min_capacity: int,
    drop_tokens: bool = True,
    ep_group: Union[torch.distributed.ProcessGroup, None] = None,
    drop_policy: str = "probs",
    use_tutel: bool = False
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements TopKGating on logits."""

    # everything is in fp32 in this function
    # get topk gates
    top_gate, top_idx = torch.topk(logits, k=k, dim=1) # shape: (n_tokens, n_experts)
    # gating decisions
    gates = F.softmax(logits, dim=1)
    num_experts = int(gates.shape[1])

    if use_tutel:
        indices_s = [x.view(-1) for x in top_idx.chunk(k, dim=1)]
        masks_se = [torch.zeros([idx.size(0), num_experts], dtype=idx.dtype, device=idx.device).scatter_(1, idx.unsqueeze(-1), 1) for idx in indices_s]
        gates_s = [(gates * x).sum(dim=1) for x in masks_se]

    # get topk mask
    topk_masked_gates = torch.zeros_like(logits).scatter(1, top_idx, top_gate) # (n_tokens, n_experts)
    mask = torch.zeros_like(gates, dtype=torch.bool).scatter_(1, top_idx, 1)
    exp_counts = torch.sum(mask, dim=0).detach().to(logits.device)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask.float(), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts / k

    if use_tutel:
        locations1 = tutel_moe.fast_cumsum_sub_one(masks_se[0])
        locations_s = [torch.sum(locations1 * masks_se[0], dim=1).to(torch.int32)]
        acc_base = None
        for k in range(1, k):
            acc_base = torch.sum(masks_se[k - 1], dim=0, keepdim=True) if acc_base is None else acc_base + torch.sum(masks_se[k - 1], dim=0, keepdim=True)
            locations2 = tutel_moe.fast_cumsum_sub_one(masks_se[k])
            locations2 += acc_base
            locations_s.append(torch.sum(locations2 * masks_se[k], dim=1).to(torch.int32))
        locations2 = locations2[-1] + 1
        indices_s = [x.to(torch.int32) for x in indices_s]

        # normalize gates
        denom_s = torch.clamp(sum(gates_s), min=torch.finfo(gates_s[0].dtype).eps)
        gates_s = [x / denom_s for x in gates_s]

        capacity = _capacity(gates, torch.tensor(capacity_factor * k), torch.tensor(min_capacity))

        # if dist.get_rank() == 0:
        #     print("=== gating output ===")
        #     print("     capacity:", capacity)
        #     print("     num_experts:", num_experts)
        #     print("     indices_s:", indices_s)
        #     print("     locations_s:", locations_s)
        #     print("     gates_s:", gates_s)
        #     print("     exp_counts:", exp_counts)
        #     print("=== end ===")

        return l_aux, capacity, num_experts, indices_s, locations_s, gates_s, exp_counts

    if drop_tokens:
        # Calculate configured capacity and remove locations outside capacity from mask
        capacity = _capacity(gates, torch.tensor(capacity_factor * k), torch.tensor(min_capacity))
        # update mask and locations by capacity

        if drop_policy == 'probs':
            capacity_probs, capacity_indices = torch.topk(topk_masked_gates, k=capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1)
            mask = torch.logical_and(mask, capacity_mask)
            locations = torch.cumsum(mask, dim=0) - 1

        elif drop_policy == "position":
            locations = torch.cumsum(mask, dim=0) - 1
            mask *= torch.lt(locations, capacity)
        else:
            raise ValueError(f"Invalid drop_policy: {drop_policy}")

    else:
        # Do not drop tokens - set capacity according to current expert assignments
        new_capacity = torch.max(exp_counts)
        if ep_group is not None:
            dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=ep_group)
        if groups._get_expert_model_parallel_world_size() == 1:
            # If the non-expert is tensor-parallel, we need to pad the capacity to 'tp'.
            # This is since we are going to activate drop_tokens() to drop duplicate tokens.
            tp = 1 if groups.mpu is None else bwc_tensor_model_parallel_world_size(mpu=groups.mpu)
            new_capacity = torch.ceil(new_capacity / tp).mul(tp).to(new_capacity.dtype)
        capacity = new_capacity

    # normalize gates
    gates_masked = gates * mask

    gates_s = torch.sum(gates_masked, dim=-1, keepdim=True)
    denom_s = torch.clamp(gates_s, min=torch.finfo(gates_masked.dtype).eps)
    gates_masked = gates_masked / denom_s

    # dispatch_mask
    locations_sc = _one_hot_to_float((locations * mask), capacity)

    combine_weights = torch.einsum("se,sec->sec", gates_masked, locations_sc)

    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts



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

    if use_pft:
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

    top_gate, top_idx = torch.topk(logits, k=k, dim=1) # shape: (n_tokens, n_experts)

    # gating decisions
    gates = F.softmax(logits, dim=1)
    num_experts = int(gates.shape[1])

    # get topk mask
    topk_masked_gates = torch.zeros_like(logits).scatter(1, top_idx, top_gate) # (n_tokens, n_experts)

    mask = torch.zeros_like(gates, dtype=torch.bool).scatter_(1, top_idx, 1)

    exp_counts = torch.sum(mask, dim=0).detach().to(logits.device)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask.float(), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts / k

    if drop_tokens:
        # Calculate configured capacity and remove locations outside capacity from mask
        capacity = _capacity(gates, torch.tensor(capacity_factor * k), torch.tensor(min_capacity))
        # update mask and locations by capacity

        if drop_policy == 'probs':
            capacity_probs, capacity_indices = torch.topk(topk_masked_gates, k=capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1)
            mask = torch.logical_and(mask, capacity_mask)
            locations = torch.cumsum(mask, dim=0) - 1

        elif drop_policy == "position":
            locations = torch.cumsum(mask, dim=0) - 1
            mask *= torch.lt(locations, capacity)
        else:
            raise ValueError(f"Invalid drop_policy: {drop_policy}")

    else:
        # Do not drop tokens - set capacity according to current expert assignments
        new_capacity = torch.max(exp_counts)
        if ep_group is not None:
            dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=ep_group)
        if groups._get_expert_model_parallel_world_size() == 1:
            # If the non-expert is tensor-parallel, we need to pad the capacity to 'tp'.
            # This is since we are going to activate drop_tokens() to drop duplicate tokens.
            tp = 1 if groups.mpu is None else bwc_tensor_model_parallel_world_size(mpu=groups.mpu)
            new_capacity = torch.ceil(new_capacity / tp).mul(tp).to(new_capacity.dtype)
        capacity = new_capacity
        capacity_probs, capacity_indices = torch.topk(topk_masked_gates, k=capacity, dim=0, sorted=False)
        capacity_mask = torch.full_like(logits, float('-inf')).scatter(0, capacity_indices, 1)
        orig_mask = mask
        mask = torch.logical_and(mask, capacity_mask)
        locations = torch.cumsum(mask, dim=0) - 1

    # normalize gates
    gates_masked = gates * mask
    gates_s = torch.sum(gates_masked, dim=-1, keepdim=True)
    denom_s = torch.clamp(gates_s, min=torch.finfo(gates_masked.dtype).eps)
    gates_masked = gates_masked / denom_s

    # dispatch_mask
    locations_sc = _one_hot_to_float((locations * mask), capacity)

    combine_weights = torch.einsum("se,sec->sec", gates_masked, locations_sc)

    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts


class TopKGate(Module):
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
                 use_uneven_all2all: bool = False) -> None:
        super().__init__()

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
        self.use_uneven_all2all = use_uneven_all2all

    def _set_ep_group(self, ep_group):
        assert self.ep_group is None, f'Attempting to override an existing ep_group'
        self.ep_group = ep_group

    def forward(self,
                input: torch.Tensor,
                used_token: torch.Tensor = None,
                use_tutel: bool = False,
                use_pft: bool = False) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore

        if self.wall_clock_breakdown:
            self.timers(TOPK_GATE_TIMER).start()

        input_fp32 = input.float()
        # input jittering
        if self.noisy_gate_policy == 'Jitter' and self.training:
            input_fp32 = multiplicative_jitter(input_fp32, device=input.device)
        logits = torch.nn.functional.linear(input_fp32, weight=self.wg.weight.float(), bias=None)

        if self.k == 1:
            gate_output = top1gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                     self.min_capacity, used_token, self.noisy_gate_policy if self.training else None,
                                     self.drop_tokens, self.use_rts, self.ep_group, use_tutel)

        elif self.k == 2:
            gate_output = top2gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                     self.min_capacity, self.drop_tokens, self.ep_group, self.top2_2nd_expert_sampling)
        else:
            if self.use_uneven_all2all:
                gate_output = topkgating_unbalanced(logits, self.k,
                                        self.capacity_factor if self.training else self.eval_capacity_factor,
                                        self.min_capacity, self.drop_tokens, self.ep_group, use_pft=use_pft)
            else:
                gate_output = topkgating(logits, self.k,
                                        self.capacity_factor if self.training else self.eval_capacity_factor,
                                        self.min_capacity, self.drop_tokens, self.ep_group, use_tutel=use_tutel)

        if self.wall_clock_breakdown:
            self.timers(TOPK_GATE_TIMER).stop()
            self.gate_time = self.timers(TOPK_GATE_TIMER).elapsed(reset=False)

        return gate_output


class MOELayer(Base):
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
                 num_shared_experts: int = 0,
                 use_tutel: bool = False) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.ep_group = None
        self.ep_size = ep_size
        self.ep_group_name = ep_group_name
        self.num_local_experts = num_local_experts
        self.num_shared_experts = num_shared_experts
        self.time_falltoall = 0.0
        self.time_salltoall = 0.0
        self.time_moe = 0.0
        self.time_experts = 0.0
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False

        self.use_tutel = use_tutel and TUTEL_INSTALLED and (gate.k == 1 or gate.k > 2)

        if self.use_tutel:
            logger.info('Using Tutel optimizations.')
        elif use_tutel and not TUTEL_INSTALLED:
            logger.warning("Tutel optimization requested but not installed. "
                           "Proceeding without Tutel.")
        elif use_tutel and TUTEL_INSTALLED and gate.k != 1:
            logger.warning("To enable Tutel optimization, use top-1 instead of top-2 gate. "
                           "Proceeding without Tutel.")

    def _set_ep_group(self, ep_group):
        self.ep_group = ep_group
        self.gate._set_ep_group(ep_group)

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:

        if self.wall_clock_breakdown:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            self.timers(MOE_TIMER).start()

        # Implement Algorithm 2 from GShard paper.
        d_model = input[0].shape[-1]

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_input = input[0].reshape(-1, d_model)

        if RANDOM_VERIFY:
            torch.manual_seed(42)
            reshaped_input = torch.randn(reshaped_input.size()).to(reshaped_input.dtype).cuda()
            if dist.get_rank() == 0:
                print("reshaped_input:", reshaped_input)

        tensor_model_world_size = bwc_tensor_model_parallel_world_size(groups.mpu)
        if tensor_model_world_size > 1 and groups._get_expert_model_parallel_world_size() == 1:
            orig_shape = reshaped_input.shape
            reshaped_input = drop_tokens(reshaped_input, dim=0)
            assert reshaped_input.shape[0] == orig_shape[0] // 2 and reshaped_input.shape[1] == orig_shape[1]

        if self.wall_clock_breakdown:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            self.timers(DISPATCH_TIMER).start()

        if self.use_tutel:
            self.l_aux, C, E, indices_, locations_, gates_, self.exp_counts = self.gate(reshaped_input, input[1], True)
            S, M = reshaped_input.size(0), reshaped_input.size(1)

            if not hasattr(self, '_tutel_dispatcher'):
                self._tutel_dispatcher = tutel_moe.fast_dispatcher(E, C, M, dispatch_dtype=reshaped_input.dtype)
            self._tutel_dispatcher.update(indices_, locations_, gates_, capacity=C)
            dispatched_input = self._tutel_dispatcher.encode(reshaped_input)
        else:
            self.l_aux, combine_weights, dispatch_mask, self.exp_counts = self.gate(reshaped_input, input[1])
            dispatched_input = einsum("sec,sm->ecm", dispatch_mask.type_as(input[0]), reshaped_input)

        if self.wall_clock_breakdown:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            self.timers(DISPATCH_TIMER).stop()
            self.time_dispatch = self.timers(DISPATCH_TIMER).elapsed(reset=False)

        if RANDOM_VERIFY and dist.get_rank() == 0:
            print(f"dispatched_input {dispatched_input.size()}: {dispatched_input}")
            torch.save(dispatched_input, f"/lustre/orion/gen150/scratch/pinaster/moe-arch/system-benchmark/deepseek-style/tensor_verif/tensor_dispatched_input_sp_{USE_SP}_tp_{groups._get_expert_model_parallel_world_size()}.pth")

        if self.wall_clock_breakdown:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            self.timers(FIRST_ALLTOALL_TIMER).start()

        if tensor_model_world_size > 1 and groups._get_expert_model_parallel_world_size() > 1:
            # If the non-expert is tensor-parallel,
            # Whether expert is tensor-parallel or not , it will create
            # duplicate tokens on the tensor-parallel ranks.
            # drop duplicate tokens also doubles up as a communication
            # optimization as we are reducing the all-to-all communication volume.
            # 1: for not tensor-parallel expert,drop duplicate tokens to ensure
            # both correctness and reduce all-to-all communication.
            # 2: for tensor-parallel expert,drop duplicate tokens to reduce all-to-all
            # communication volume,before expert execution, it is necessary to perform
            # an allgather to ensure correctness,
            dispatched_input = drop_tokens(dispatched_input, dim=1)

        dispatched_input = _AllToAll.apply(self.ep_group, dispatched_input)

        if self.wall_clock_breakdown:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            self.timers(FIRST_ALLTOALL_TIMER).stop()
            self.time_falltoall = self.timers(FIRST_ALLTOALL_TIMER).elapsed(reset=False)

        if tensor_model_world_size > 1 and groups._get_expert_model_parallel_world_size() > 1:
            # if both expert and non-expert are tensor-parallel
            # the dropped duplicate tokens need to be gathered on each
            # tensor parallel rank again to ensure correctness
            dispatched_input = gather_tokens(dispatched_input, dim=1)

        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.ep_size, self.num_local_experts, -1, d_model)

        if self.wall_clock_breakdown:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            self.timers(EXPERTS_TIMER).start()
        expert_output = self.experts(dispatched_input)
        if self.wall_clock_breakdown:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            self.timers(EXPERTS_TIMER).stop()
            self.time_experts = self.timers(EXPERTS_TIMER).elapsed(reset=False)

        # Re-shape before drop_tokens: gecm -> ecm
        expert_output = expert_output.reshape(self.ep_size * self.num_local_experts, -1, d_model)
        if tensor_model_world_size > 1 and groups._get_expert_model_parallel_world_size() > 1:
            # if both expert and non-expert are tensor-parallel
            # drop duplicate tokens to ensure both correctness
            # and reduce all-to-all communication.
            expert_output = drop_tokens(expert_output, dim=1)

        if self.wall_clock_breakdown:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            self.timers(SECOND_ALLTOALL_TIMER).start()

        expert_output = _AllToAll.apply(self.ep_group, expert_output)

        if self.wall_clock_breakdown:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            self.timers(SECOND_ALLTOALL_TIMER).stop()
            self.time_salltoall = self.timers(SECOND_ALLTOALL_TIMER).elapsed(reset=False)

        if tensor_model_world_size > 1 and groups._get_expert_model_parallel_world_size() > 1:
            # the dropped duplicate tokens need to be gathered on each
            # tensor parallel rank again for the tensor-parallel
            # non-expert of the next layer.
            expert_output = gather_tokens(expert_output, dim=1)

        if self.wall_clock_breakdown:
            torch.distributed.barrier()
            self.timers(COMBINE_TIMER).start()

        if self.use_tutel:
            combined_output = self._tutel_dispatcher.decode(expert_output.view(E * C, M))
        else:
            combined_output = einsum("sec,ecm->sm", combine_weights.type_as(input[0]), expert_output)

        if self.wall_clock_breakdown:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            self.timers(COMBINE_TIMER).stop()
            self.time_combine = self.timers(COMBINE_TIMER).elapsed(reset=False)


        if tensor_model_world_size > 1 and groups._get_expert_model_parallel_world_size() == 1:
            # the dropped duplicate tokens need to be gathered on each
            # tensor parallel rank again for the tensor-parallel
            # non-expert of the next layer.
            combined_output = gather_tokens(combined_output, dim=0)
            assert combined_output.shape == orig_shape

        a = combined_output.reshape(input[0].shape)
        
        if RANDOM_VERIFY:
            if dist.get_rank() == 0:
                print("combined_output: ", a)
            torch.save(a, f"/lustre/orion/gen150/scratch/pinaster/moe-arch/system-benchmark/deepseek-style/tensor_verif/tensor_a_sp_{USE_SP}_tp_{groups._get_expert_model_parallel_world_size()}.pth")

        if self.wall_clock_breakdown:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            self.timers(MOE_TIMER).stop()
            self.time_moe = self.timers(MOE_TIMER).elapsed(reset=False)

        return a


from .v2opt.utils import remove_zero_rows, restore_zero_rows, compare_tensors, compare_uneven_and_padded


class UnblancedMOELayer(Base):
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
                 use_tutel: bool = False,
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

        self.use_tutel = use_tutel and TUTEL_INSTALLED and gate.k == 1
        self.use_pft = use_pft

        self.drop_tokens = drop_tokens

        if self.use_tutel:
            logger.info('Using Tutel optimizations.')
        elif use_tutel and not TUTEL_INSTALLED:
            logger.warning("Tutel optimization requested but not installed. "
                           "Proceeding without Tutel.")
        elif use_tutel and TUTEL_INSTALLED and gate.k != 1:
            logger.warning("To enable Tutel optimization, use top-1 instead of top-2 gate. "
                           "Proceeding without Tutel.")
        
        if num_local_experts > 1:
            print("uneven all-to-all for num_local_experts > 1 is not implemented")

    def _set_ep_group(self, ep_group):
        self.ep_group = ep_group
        self.gate._set_ep_group(ep_group)

    
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

        n_tokens = reshaped_input.shape[0]

        tensor_model_world_size = bwc_tensor_model_parallel_world_size(groups.mpu)
        if tensor_model_world_size > 1 and groups._get_expert_model_parallel_world_size() == 1:
            orig_shape = reshaped_input.shape
            reshaped_input = drop_tokens(reshaped_input, dim=0)
            assert reshaped_input.shape[0] == orig_shape[0] // 2 and reshaped_input.shape[1] == orig_shape[1]

        if self.use_tutel:
            self.l_aux, C, E, indices_, locations_, gates_, self.exp_counts = self.gate(reshaped_input, input[1], use_tutel=True)
            S, M = reshaped_input.size(0), reshaped_input.size(1)

            if not hasattr(self, '_tutel_dispatcher'):
                self._tutel_dispatcher = tutel_moe.fast_dispatcher(E, C, M, dispatch_dtype=reshaped_input.dtype)
            self._tutel_dispatcher.update(indices_, locations_, gates_, capacity=C)
            dispatched_input = self._tutel_dispatcher.encode(reshaped_input)

        elif self.use_pft:

            # print(f"rank [{dist.get_rank()}]: before megablocks gating", flush=True)
            self.l_aux, indices, bin_ids, bins, expert_weights, input_splits_tensor = self.gate(reshaped_input, input[1], use_pft=True)
           
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

            # dispatched_input = einsum("sec,sm->ecm", dispatch_mask.type_as(input[0]), reshaped_input)
            
        else:
            self.l_aux, combine_weights, dispatch_mask, self.exp_counts = self.gate(reshaped_input, input[1])
            if self.wall_clock_breakdown:
                torch.distributed.barrier()
                self.timers(DISPATCH_TIMER).start()

            dispatched_input = einsum("sec,sm->ecm", dispatch_mask.type_as(input[0]), reshaped_input)

            flattened_input, input_splits_tensor, padding_mask = remove_zero_rows(dispatched_input)

            if self.wall_clock_breakdown:
                torch.distributed.barrier()
                self.timers(DISPATCH_TIMER).stop()
                self.time_dispatch = self.timers(DISPATCH_TIMER).elapsed(reset=False)
        
        output_splits_tensor = _AllToAll.apply(self.ep_group, input_splits_tensor)
        input_splits_tensor_ep = input_splits_tensor.view(-1, self.num_local_experts).sum(dim=1)
        output_splits_tensor_ep = output_splits_tensor.view(-1, self.num_local_experts).sum(dim=1)

        input_splits = input_splits_tensor_ep.tolist()
        output_splits = output_splits_tensor_ep.tolist()

        # if dist.get_rank() == 0:
        #     print("===== input splits =====")
        # sys.stdout.flush()
        # dist.barrier()
        # print(f"input splits on devices [{dist.get_rank()}]: {input_splits_tensor_ep}")
        # sys.stdout.flush()
        # dist.barrier()

        assert sum(input_splits) == flattened_input.shape[0], f"input_split sum {sum(input_splits)} != input shape [0] {flattened_input.shape[0]} on rank {dist.get_rank()}"

        total_recv = sum(output_splits)
        # dispatched_output = torch.empty(total_recv, d_model, device=flattened_input.device, dtype=flattened_input.dtype)
        # dist.all_to_all_single(dispatched_output, flattened_input, output_split_sizes=output_splits, \
            # input_split_sizes=input_splits, group=self.ep_group)
        dispatched_output = _AllToAllSingle.apply(self.ep_group, flattened_input, input_splits, output_splits)


        if self.wall_clock_breakdown:
            torch.distributed.barrier()
            self.timers(FIRST_ALLTOALL_TIMER).start()

        # if tensor_model_world_size > 1 and groups._get_expert_model_parallel_world_size() > 1:
        #     dispatched_input = drop_tokens(dispatched_input, dim=1)

        # dispatched_input = _AllToAll.apply(self.ep_group, dispatched_input) # VERIFICATION

        # print(f"=== rank {dist.get_rank()} after 1st A2A ===")
        # print(dispatched_output.shape)
        # compare_uneven_and_padded(dispatched_output, dispatched_input)
        # sys.stdout.flush()
        # dist.barrier()

        if self.wall_clock_breakdown:
            torch.distributed.barrier()
            self.timers(FIRST_ALLTOALL_TIMER).stop()
            self.time_falltoall = self.timers(FIRST_ALLTOALL_TIMER).elapsed(reset=False)

        # if tensor_model_world_size > 1 and groups._get_expert_model_parallel_world_size() > 1:
        #     dispatched_input = gather_tokens(dispatched_input, dim=1)

        # Re-shape after all-to-all: ecm -> gecm
        # dispatched_input = dispatched_input.reshape(self.ep_size, self.num_local_experts, -1, d_model) # VERIFICATION
        
        # TODO: num_local_experts > 1: 
        # dispatched_output = dispatched_output.reshape(self.num_local_experts, -1, d_model)
        splits = output_splits_tensor.view(-1, self.num_local_experts).sum(dim=0).tolist() 

        # print(f"on ep rank {self.get_ep_rank()}, splits are {splits}, dispatched_output shape is {dispatched_output.shape}")
        assert sum(splits) == dispatched_output.shape[0], "sum of local splits != dispatched output shape"

        if self.wall_clock_breakdown:
            torch.distributed.barrier()
            self.timers(EXPERTS_TIMER).start()
        # expert_output_verif = self.experts(dispatched_input) # VERIFICATION

        expert_output_uneven = self.experts(dispatched_output, splits)

        if self.wall_clock_breakdown:
            torch.distributed.barrier()
            self.timers(EXPERTS_TIMER).stop()
            self.time_experts = self.timers(EXPERTS_TIMER).elapsed(reset=False)

        # Re-shape before drop_tokens: gecm -> ecm
        # expert_output_verif = expert_output_verif.reshape(self.ep_size * self.num_local_experts, -1, d_model) # VERIFICATION
        # expert_output_uneven = expert_output_uneven.reshape(-1, d_model)

        # if tensor_model_world_size > 1 and groups._get_expert_model_parallel_world_size() > 1:
        #     # if both expert and non-expert are tensor-parallel
        #     # drop duplicate tokens to ensure both correctness
        #     # and reduce all-to-all communication.
        #     expert_output = drop_tokens(expert_output, dim=1)

        if self.wall_clock_breakdown:
            torch.distributed.barrier()
            self.timers(SECOND_ALLTOALL_TIMER).start()

        # dist.all_to_all_single(flattened_input, expert_output_uneven, output_split_sizes=input_splits, \
        #     input_split_sizes=output_splits, group=self.ep_group)

        expert_output_uneven = _AllToAllSingle.apply(self.ep_group, expert_output_uneven, output_splits, input_splits)

        # expert_output_verif = _AllToAll.apply(self.ep_group, expert_output_verif) # VERIFICATION

        # print(f"=== rank {dist.get_rank()} after 2st A2A ===")
        # print(expert_output_uneven.shape)
        # print(expert_output_verif.shape)
        # compare_uneven_and_padded(expert_output_uneven, expert_output_verif)
        # sys.stdout.flush()
        # dist.barrier()

        if self.wall_clock_breakdown:
            torch.distributed.barrier()
            self.timers(SECOND_ALLTOALL_TIMER).stop()
            self.time_salltoall = self.timers(SECOND_ALLTOALL_TIMER).elapsed(reset=False)
        
        if self.use_pft:
            expert_output = expert_output_uneven
        else:
            expert_output = restore_zero_rows(expert_output_uneven, padding_mask)

        # compare_tensors(expert_output, expert_output_verif)
        # sys.stdout.flush()
        # dist.barrier()
        # VERIFICATION PASSED UP TO HERE

        if tensor_model_world_size > 1 and groups._get_expert_model_parallel_world_size() > 1:
            # the dropped duplicate tokens need to be gathered on each
            # tensor parallel rank again for the tensor-parallel
            # non-expert of the next layer.
            expert_output = gather_tokens(expert_output, dim=1)

        if self.wall_clock_breakdown:
            torch.distributed.barrier()
            self.timers(COMBINE_TIMER).start()

        if self.use_tutel:
            combined_output = self._tutel_dispatcher.decode(expert_output.view(E * C, M))
        elif self.use_pft:
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

        if tensor_model_world_size > 1 and groups._get_expert_model_parallel_world_size() == 1:
            # the dropped duplicate tokens need to be gathered on each
            # tensor parallel rank again for the tensor-parallel
            # non-expert of the next layer.
            combined_output = gather_tokens(combined_output, dim=0)
            assert combined_output.shape == orig_shape

        a = combined_output.reshape(input[0].shape)
        
        if self.wall_clock_breakdown:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            self.timers(MOE_TIMER).stop()
            self.time_moe = self.timers(MOE_TIMER).elapsed(reset=False)

        return a