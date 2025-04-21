import torch
import triton
import triton.language as tl

from .kernels import assert_is_tensor, assert_is_matrix, assert_is_vector, assert_equal, _padded_copy, _padded_copy_wgrad
from .utils import print_rank
# a: (tokens, hidden_size), real.
# indices: (tokens * top_k), integer.
# bin_ids: (tokens * top_k), integer.
# weights: (tokens * top_k), real.
# bins: (num_experts), integer.
# padded_bins: (num_experts), integer.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_X': 64}, num_warps=2),
        triton.Config({'BLOCK_X': 128}, num_warps=2),
        triton.Config({'BLOCK_X': 256}, num_warps=2),
        triton.Config({'BLOCK_X': 128}, num_warps=4),
        triton.Config({'BLOCK_X': 256}, num_warps=4),
    ],
    key=['NUM_COLUMNS'],
)
@triton.jit
def _map_copy(
    a,
    b,
    indices,
    b_indices,
    weights,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_X: tl.constexpr,
    A_TO_B: tl.constexpr,
    SCALE: tl.constexpr,
):
    # Our index into array 'a'.
    index_a = tl.load(indices + tl.program_id(0))

    # One threadblock per row in 'a'. Array 'b' has greater or equal
    # number of rows since they could be padded.
    index_b = tl.load(b_indices + tl.program_id(0))

    # Offset the input and output pointers.
    #
    # If we're going from A to B, divide the input index to copy
    # the same input repeatedly. If we're going from B to A we
    # need to reduce the result. Using atomics is slow, so we
    # do the reduce step in a second kernel.
    # offset = index_a // TOP_K if A_TO_B else index_a
    offset = index_a
    a += tl.multiple_of(offset * NUM_COLUMNS, NUM_COLUMNS)
    b += tl.multiple_of(index_b * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

    # Load the scale, if requested.
    scale = tl.load(weights + index_a) if SCALE else 1

    # Swap the pointers depending on the direction.
    iptr = a if A_TO_B else b
    optr = b if A_TO_B else a

    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        x = tl.load(iptr + offsets, mask=mask)
        x = x.to(tl.float32) * scale.to(tl.float32)

        tl.store(optr + offsets, x.to(optr.dtype.element_ty), mask=mask)

        offsets += BLOCK_X


def _map(x, indices, weights):
    assert_is_matrix(x)
    assert_is_vector(indices)

    # if weights is not None:
    #     assert_equal(weights.shape[0], n_tokens * top_k)

    b_indices = torch.arange(indices.shape[0], dtype=indices.dtype, device=indices.device)

    output_rows = indices.shape[0]
    out = torch.empty((output_rows, x.shape[1]), dtype=x.dtype, device=x.device)

    _map_copy[(indices.shape[0],)](
        x,
        out,
        indices,
        b_indices,
        weights,
        NUM_COLUMNS=x.shape[1],
        A_TO_B=True,
        SCALE=weights is not None,
    )
    return out


def s2_order(x, indices, bin_ids, bins):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_equal(indices.shape[0], bin_ids.shape[0])
    assert_equal(x.shape[0], indices.shape[0])

    # NOTE: There is no padding so the output rows equals the
    # input rows multiplied by top_k.
    output_rows = indices.shape[0]
    out = torch.empty((output_rows, x.shape[1]), dtype=x.dtype, device=x.device)
    _padded_copy[(indices.shape[0],)](
        x,
        out,
        indices,
        bin_ids,
        None,
        bins,
        bins,
        NUM_COLUMNS=x.shape[1],
        TOP_K=1,
        A_TO_B=True,
        SCALE=None,
    )
    return out

def s2_order_recover(x, indices, bin_ids, bins):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_equal(x.shape[0], indices.shape[0])
    assert_equal(indices.shape[0], bin_ids.shape[0])

    output_rows = indices.shape[0]
    out = torch.empty((output_rows, x.shape[1]), dtype=x.dtype, device=x.device)

    _padded_copy[(indices.shape[0],)](
        out,
        x,
        indices,
        bin_ids,
        None,
        bins,
        bins,
        NUM_COLUMNS=x.shape[1],
        TOP_K=1,
        A_TO_B=False,
        SCALE=None,
    )

    # Reduce along the top-k dimension, if needed.
    return out


def _s1_to_s2_gather(x, s1exp_to_s2_indices, bin_ids_s2, weights, bins_s2):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(s1exp_to_s2_indices)
    assert_is_vector(bin_ids_s2)
    assert_is_vector(bins_s2)
    assert_equal(s1exp_to_s2_indices.shape[0], bin_ids_s2.shape[0])

    assert weights is None

    # if weights is not None:
    #     assert_equal(weights.shape[0], n_tokens * top_k)

    # NOTE: There is no padding so the output rows equals the
    # input rows multiplied by top_k.
    output_rows = s1exp_to_s2_indices.shape[0]
    out = torch.empty((output_rows, x.shape[1]), dtype=x.dtype, device=x.device)

    _padded_copy[(s1exp_to_s2_indices.shape[0],)](
        x,
        out,
        s1exp_to_s2_indices,
        bin_ids_s2,
        weights,
        bins_s2,
        bins_s2,
        NUM_COLUMNS=x.shape[1],
        TOP_K=1,
        A_TO_B=True,
        SCALE=weights is not None,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_X': 64}, num_warps=2),
        triton.Config({'BLOCK_X': 128}, num_warps=2),
        triton.Config({'BLOCK_X': 256}, num_warps=2),
        triton.Config({'BLOCK_X': 128}, num_warps=4),
        triton.Config({'BLOCK_X': 256}, num_warps=4),
    ],
    key=['NUM_COLUMNS'],
)
@triton.jit
def _s1s2_copy(
    a,
    b,
    indices,
    bin_ids,
    weights,
    bins,
    padded_bins,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_X: tl.constexpr,
    A_TO_B: tl.constexpr,
    SCALE: tl.constexpr,
):
    # Our index into array 'a'.
    index_a = tl.load(indices + tl.program_id(0))

    # One threadblock per row in 'a'. Array 'b' has greater or equal
    # number of rows since they could be padded.
    bin_idx = tl.load(bin_ids + tl.program_id(0))

    # Now we know what bin we're assigned to, but we need to know how
    # many threadblocks were assigned to earlier bins so we can offset
    # in our bin properly.
    offset_in_bin = tl.program_id(0)
    if bin_idx > 0:
        offset_in_bin -= tl.load(bins + bin_idx - 1)

    # Load the starting index of our bin in array 'b'.
    index_b = offset_in_bin
    if bin_idx > 0:
        index_b += tl.load(padded_bins + bin_idx - 1)

    # Offset the input and output pointers.
    #
    # If we're going from A to B, divide the input index to copy
    # the same input repeatedly. If we're going from B to A we
    # need to reduce the result. Using atomics is slow, so we
    # do the reduce step in a second kernel.
    offset = index_a
    a += tl.multiple_of(offset * NUM_COLUMNS, NUM_COLUMNS)
    b += tl.multiple_of(index_b * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

    # Load the scale, if requested.
    scale = tl.load(weights + index_a) if SCALE else 1

    # Swap the pointers depending on the direction.
    iptr = a if A_TO_B else b
    optr = b if A_TO_B else a

    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        x = tl.load(iptr + offsets, mask=mask)
        x = x.to(tl.float32) * scale.to(tl.float32)

        tl.store(optr + offsets, x.to(optr.dtype.element_ty), mask=mask)

        offsets += BLOCK_X



# Add the value of s2 to s1's indices
# def _s2_to_s1_scatter(x, s1exp_to_s2_indices, bin_ids_s2, weights, bins_s2):
#     # Validate the input shapes.
#     assert_is_matrix(x)
#     assert_is_vector(s1exp_to_s2_indices)
#     assert_is_vector(bin_ids_s2)
#     assert_is_vector(bins_s2)
#     assert_equal(s1exp_to_s2_indices.shape[0], bin_ids_s2.shape[0])

#     # if weights is not None:
#     #     assert_equal(n_tokens * top_k, weights.shape[0])

#     _s1s2_copy[(s1exp_to_s2_indices.shape[0],)](
#         out,
#         x,
#         s1exp_to_s2_indices,
#         bin_ids_s2,
#         weights,
#         bins_s2,
#         bins_s2,
#         NUM_COLUMNS=x.shape[1],
#         A_TO_B=False,
#         SCALE=weights is not None,
#     )

#     # Reduce along the top-k dimension, if needed.
#     return out

