import torch
from typing import Tuple, Any, Optional

from .helper import custom_bwd, custom_fwd

import sys
import os
from .rbd_kernels import s2_order, s2_order_recover, _s1_to_s2_gather
from .drop_kernel import dropped_scatter
from .drop_kernel import dropped_scatter_wgrad

from .utils import print_rank


class S1ToS2GatherOp(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx: Any,
        x: torch.Tensor,
        s1exp_to_s2_indices: torch.Tensor,
        bin_ids_s2: torch.Tensor,
        bins_s2: torch.Tensor,
    ):
        ctx.save_for_backward(s1exp_to_s2_indices, bin_ids_s2, bins_s2)
        ctx.xshape = x.shape
        ctx.hidden_dim = x.shape[-1]
        return _s1_to_s2_gather(x, s1exp_to_s2_indices, bin_ids_s2, None, bins_s2)

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, grad: torch.Tensor):
        grad = grad.contiguous()
        s1exp_to_s2_indices, bin_ids_s2, bins_s2 = ctx.saved_tensors

        out = torch.zeros(ctx.xshape, device=grad.device, dtype=grad.dtype)
        expanded_indices = s1exp_to_s2_indices.unsqueeze(1).expand(-1, ctx.hidden_dim)
        out.scatter_add_(0, expanded_indices, grad)
        # out = _s2_to_s1_scatter(grad, s1exp_to_s2_indices, bin_ids_s2, None, bins_s2)
        return out, None, None, None

s1_to_s2_gather = S1ToS2GatherOp.apply

class S2ToS1ScatterOp(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx: Any,
        dout_s1_exp: torch.Tensor,
        dout_s2: torch.Tensor,
        s1exp_to_s2_indices: torch.Tensor,
        bin_ids_s2: torch.Tensor,
        bins_s2: torch.Tensor,
    ):
        ctx.save_for_backward(s1exp_to_s2_indices, bin_ids_s2, bins_s2)
        ctx.xshape = dout_s1_exp.shape
        ctx.hidden_dim = dout_s1_exp.shape[-1]

        expanded_indices = s1exp_to_s2_indices.unsqueeze(1).expand(-1, ctx.hidden_dim)
        return dout_s1_exp.scatter_add_(0, expanded_indices, dout_s2)

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, grad: torch.Tensor):
        grad = grad.contiguous()
        s1exp_to_s2_indices, bin_ids_s2, bins_s2 = ctx.saved_tensors

        grad_dout_s1_exp = grad
        grad_dout_s2 = _s1_to_s2_gather(grad, s1exp_to_s2_indices, bin_ids_s2, None, bins_s2)

        return grad_dout_s1_exp, grad_dout_s2, None, None, None

s2_to_s1_scatter = S2ToS1ScatterOp.apply


# Autograd wrapper for gather kernel.
class LocalOrderOp(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx: Any,
        x_s1: torch.Tensor,
        tokens_per_experts_s1_exp: torch.Tensor,
        x_s2: Optional[torch.Tensor] = None,
        tokens_per_experts_s2_exp: Optional[torch.Tensor] = None,
    ):
        ctx.save_for_backward(tokens_per_experts_s1_exp, tokens_per_experts_s2_exp)
        return s2_order(x_s1, tokens_per_experts_s1_exp, x_s2, tokens_per_experts_s2_exp)

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, grad: torch.Tensor):
        grad = grad.contiguous()
        indices, bin_ids, bins = ctx.saved_tensors
        out = s2_order_recover(grad, indices, bin_ids, bins)
        return out, None, None, None


class LocalOrderRecoverOp(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
    ):
        ctx.save_for_backward(indices, bin_ids, bins)
        return s2_order_recover(x, indices, bin_ids, bins)

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, grad: torch.Tensor):
        grad = grad.contiguous()
        indices, bin_ids, bins = ctx.saved_tensors
        s2_order(grad, indices, bin_ids, bins)
        return grad, None, None, None



