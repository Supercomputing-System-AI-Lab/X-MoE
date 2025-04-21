import torch
from typing import Tuple, Any, Optional

from .helper import custom_bwd, custom_fwd

import sys
import os
from .drop_kernel import drop_gather
from .drop_kernel import dropped_scatter
from .drop_kernel import dropped_scatter_wgrad


# Autograd wrapper for gather kernel.
class DroppedGatherOp(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        n_tokens: int,
        top_k: int,
    ):
        ctx.save_for_backward(indices, bin_ids, bins)
        ctx.n_tokens = n_tokens
        ctx.top_k = top_k
        return drop_gather(x, indices, bin_ids, None, bins, n_tokens, top_k)

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, grad: torch.Tensor):
        grad = grad.contiguous()

        indices, bin_ids, bins = ctx.saved_tensors
        n_tokens = ctx.n_tokens
        top_k = ctx.top_k
        out = dropped_scatter(grad, indices, bin_ids, None, bins, n_tokens, top_k)
        return out, None, None, None, None, None


gather_with_token_drop = DroppedGatherOp.apply

# Autograd wrapper for scatter kernel.
class DroppedScatterOp(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        weights: torch.Tensor,
        bins: torch.Tensor,
        n_tokens: int,
        top_k: int
    ) -> torch.Tensor:
        maybe_x = [x] if ctx.needs_input_grad[3] else []
        ctx.save_for_backward(indices, bin_ids, weights, bins, *maybe_x)
        ctx.x_shape = x.shape
        ctx.top_k = top_k
        ctx.n_tokens = n_tokens
        return dropped_scatter(x, indices, bin_ids, weights, bins, n_tokens, top_k)

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, grad: torch.Tensor):
        grad = grad.contiguous()
        saved_tensors = ctx.saved_tensors

        indices, bin_ids, weights, bins = saved_tensors[:4]

        top_k = ctx.top_k
        n_tokens = ctx.n_tokens

        dgrad = None
        if ctx.needs_input_grad[0]:
            dgrad = drop_gather(
                grad,
                indices,
                bin_ids,
                weights,
                bins,
                n_tokens,
                top_k
            )

        wgrad = None
        if ctx.needs_input_grad[3]:  # need wgrad
            x = saved_tensors[-1]
            wgrad = dropped_scatter_wgrad(
                x,
                grad,
                indices,
                bin_ids,
                bins,
                n_tokens,
                top_k
            )
        
        return dgrad, None, None, wgrad, None, None, None


def scatter_with_token_drop(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    weights: torch.Tensor,
    bins: torch.Tensor,
    n_tokens: int,
    top_k: int,
) -> Optional[torch.Tensor]:
    return DroppedScatterOp.apply(x, indices, bin_ids, weights, bins, n_tokens, top_k)



def indices_and_bins_and_drop(num_experts: int,
                              top_experts: torch.Tensor,
                              expert_weights: Optional[torch.Tensor] = None,
                              capacity: Optional[int] = None,
                              drop_policy: Optional[str] = 'probs',
                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    top_experts = top_experts.int()
    output = torch.sort(top_experts)
    assert output is not None
    bin_ids, indices = output

    if capacity is not None:
        if drop_policy == 'probs':
            sorted_order = torch.argsort(expert_weights, descending=True)
            sorted_expert_choices = top_experts[sorted_order]

            # sorted_expert_choices = sorted_expert_choices.to(torch.int64)
            # one_hot_experts = torch.nn.functional.one_hot(sorted_expert_choices, num_classes=num_experts).float()
            # cumsum_expert_choices = torch.cumsum(one_hot_experts, dim=0)

            # rows = torch.arange(sorted_expert_choices.size(0), device='cuda')
            # rank_in_expert = cumsum_expert_choices[rows, sorted_expert_choices]

            N = sorted_expert_choices.size(0)
            one_hot_counts = torch.zeros((num_experts, N), dtype=torch.half, device='cuda')
            token_index = torch.arange(N, device='cuda')
            one_hot_counts[sorted_expert_choices, token_index] = 1
            prefix_sum = torch.cumsum(one_hot_counts, dim=1)
            rank_in_expert = prefix_sum[sorted_expert_choices, token_index]

            weight_mask = rank_in_expert <= capacity
            filtered_index = sorted_order[weight_mask]

            mask = torch.isin(indices, filtered_index)
            indices = indices[mask]

            bin_ids = top_experts[indices]
        else:
            raise NotImplementedError(f"drop policy {drop_policy} is not implemented")

    tokens_per_expert = torch.histc(bin_ids, num_experts, 0, num_experts - 1)

    bins = torch.cumsum(tokens_per_expert, 0)
    assert bins is not None
    bins = bins.view(1) if not len(bins.size()) else bins

    return indices, bin_ids, bins.int(), tokens_per_expert
