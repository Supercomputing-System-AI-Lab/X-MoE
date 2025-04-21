import torch
from typing import Tuple, Any, Optional

from .helper import custom_bwd, custom_fwd

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
from .kernels import gather as _gather
from .kernels import scatter as _scatter
from .kernels import scatter_wgrad as _scatter_wgrad


# Autograd wrapper for gather kernel.
class GatherOp(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        top_k: int,
    ):
        ctx.save_for_backward(indices, bin_ids, bins)
        ctx.top_k = top_k
        return _gather(x, indices, bin_ids, None, bins, top_k)

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, grad: torch.Tensor):
        grad = grad.contiguous()

        indices, bin_ids, bins = ctx.saved_tensors
        out = _scatter(grad, indices, bin_ids, None, bins, ctx.top_k)
        return out, None, None, None, None, None


gather = GatherOp.apply

# Autograd wrapper for scatter kernel.
class ScatterOp(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        weights: torch.Tensor,
        bins: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        maybe_x = [x] if ctx.needs_input_grad[3] else []
        ctx.save_for_backward(indices, bin_ids, weights, bins, *maybe_x)
        ctx.top_k = top_k
        ctx.x_shape = x.shape
        return _scatter(x, indices, bin_ids, weights, bins, top_k)

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, grad: torch.Tensor):
        grad = grad.contiguous()
        saved_tensors = ctx.saved_tensors

        indices, bin_ids, weights, bins = saved_tensors[:4]
        dgrad = None
        if ctx.needs_input_grad[0]:
            dgrad = _gather(
                grad,
                indices,
                bin_ids,
                weights,
                bins,
                ctx.top_k,
            )

        wgrad = None
        if ctx.needs_input_grad[3]:  # need wgrad
            x = saved_tensors[-1]
            wgrad = _scatter_wgrad(
                x,
                grad,
                indices,
                bin_ids,
                bins,
                ctx.top_k,
            )
        return dgrad, None, None, wgrad, None, None, None


def scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    weights: torch.Tensor,
    bins: torch.Tensor,
    top_k: int,
) -> Optional[torch.Tensor]:
    return ScatterOp.apply(x, indices, bin_ids, weights, bins, top_k)




class MBgate():
    def __init__(self, n_experts, model_dim, k):
        self.num_experts = n_experts
        self.model_dim = model_dim
        self.k = k


    def indices_and_bins(self, top_expert: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        #
        # TODO(tgale): Is it worth doing this conversion to 32-bit
        # prior? Could we place the `torch.max` operation to return
        # 32-bit expert indices?
        top_expert = top_expert.int()
        output = torch.sort(top_expert)
        assert output is not None
        bin_ids, indices = output

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        #
        # TODO(tgale): Does the sorted data produce a more favorable
        # data distribution for histogram? Or is the op parallelism
        # worth more?
        tokens_per_expert = torch.histc(top_expert, self.num_experts, 0, self.num_experts - 1)

        # Calculate the bin bounds for the sorted tokens.
        bins = torch.cumsum(tokens_per_expert, 0)
        assert bins is not None
        bins = bins.view(1) if not len(bins.size()) else bins

        return indices, bin_ids, bins.int(), tokens_per_expert

def indices_and_bins(num_experts: int, top_expert: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Sort the expert ids to produce the scatter/gather
    # indices for the permutation.
    #
    # TODO(tgale): Is it worth doing this conversion to 32-bit
    # prior? Could we place the `torch.max` operation to return
    # 32-bit expert indices?
    top_expert = top_expert.int()
    output = torch.sort(top_expert)
    assert output is not None
    bin_ids, indices = output

    # Histogram the expert ids to identify the number of
    # tokens routed to each expert.
    #
    # TODO(tgale): Does the sorted data produce a more favorable
    # data distribution for histogram? Or is the op parallelism
    # worth more?
    tokens_per_expert = torch.histc(top_expert, num_experts, 0, num_experts - 1)

    # Calculate the bin bounds for the sorted tokens.
    bins = torch.cumsum(tokens_per_expert, 0)
    assert bins is not None
    bins = bins.view(1) if not len(bins.size()) else bins

    return indices, bin_ids, bins.int(), tokens_per_expert