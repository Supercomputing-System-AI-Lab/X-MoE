# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from .kernels import assert_is_tensor, assert_is_matrix, assert_is_vector, assert_equal, _padded_copy, _padded_copy_wgrad


def drop_gather(x, indices, bin_ids, weights, bins, n_tokens, top_k):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_equal(indices.shape[0], bin_ids.shape[0])

    if weights is not None:
        assert_equal(weights.shape[0], n_tokens * top_k)

    # NOTE: There is no padding so the output rows equals the
    # input rows multiplied by top_k.
    output_rows = indices.shape[0]
    out = torch.empty((output_rows, x.shape[1]), dtype=x.dtype, device=x.device)
    _padded_copy[(indices.shape[0],)](
        x,
        out,
        indices,
        bin_ids,
        weights,
        bins,
        bins,
        NUM_COLUMNS=x.shape[1],
        TOP_K=top_k,
        A_TO_B=True,
        SCALE=weights is not None,
    )
    return out



def dropped_scatter(x, indices, bin_ids, weights, bins, n_tokens, top_k):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_equal(indices.shape[0], bin_ids.shape[0])

    if weights is not None:
        assert_equal(n_tokens * top_k, weights.shape[0])

    out = torch.zeros((n_tokens, top_k, x.shape[1]), dtype=x.dtype, device=x.device)

    _padded_copy[(indices.shape[0],)](
        out,
        x,
        indices,
        bin_ids,
        weights,
        bins,
        bins,
        NUM_COLUMNS=x.shape[1],
        TOP_K=top_k,
        A_TO_B=False,
        SCALE=weights is not None,
    )

    # Reduce along the top-k dimension, if needed.
    return out.sum(dim=1) if top_k > 1 else out.view(n_tokens, x.shape[1])



def dropped_scatter_wgrad(x, grad, indices, bin_ids, bins, n_tokens, top_k):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_matrix(grad)
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_equal(indices.shape[0], bin_ids.shape[0])

    # tokens = indices.shape[0] // top_k
    # out = torch.zeros((indices.shape[0]), dtype=x.dtype, device=x.device)
    out = torch.zeros(n_tokens * top_k, dtype=x.dtype, device=x.device)
    _padded_copy_wgrad[(indices.shape[0],)](
        x,
        grad,
        out,
        indices,
        bin_ids,
        bins,
        bins,
        NUM_COLUMNS=x.shape[1],
        TOP_K=top_k
    )
    return out

