# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from .utils import print_rank

def assert_is_tensor(x, ndim):
    if x.ndim != ndim:
        raise ValueError(f'Expected {ndim}-tensor but got {x.ndim}-tensor')


def assert_is_matrix(x):
    assert_is_tensor(x, 2)


def assert_is_vector(x):
    if x.ndim != 1:
        raise ValueError(f'Expected 1-tensor but got {x.ndim}-tensor')


def assert_equal(a, b):
    if a != b:
        raise ValueError(f'Expected dimensions to be equal but got {a} and {b}.',)


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
def _padded_copy(
    a,
    b,
    indices,
    bin_ids,
    weights,
    bins,
    padded_bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
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
    offset = index_a // TOP_K if A_TO_B else index_a
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


def padded_gather(x, indices, bin_ids, weights, bins, padded_bins, top_k):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_is_vector(padded_bins)
    assert_equal(indices.shape[0], x.shape[0] * top_k)
    assert_equal(bin_ids.shape[0], x.shape[0] * top_k)
    assert_equal(bins.size(), padded_bins.size())

    if weights is not None:
        assert_equal(weights.shape[0], x.shape[0] * top_k)

    # NOTE: Because of the padding, the output size is dynamic.
    # We load the final padded bin bound to get the output rows.
    output_rows = padded_bins[-1].cpu().item()
    out = torch.zeros((output_rows, x.shape[1]), dtype=x.dtype, device=x.device)
    _padded_copy[(indices.shape[0],)](
        x,
        out,
        indices,
        bin_ids,
        weights,
        bins,
        padded_bins,
        NUM_COLUMNS=x.shape[1],
        A_TO_B=True,
        TOP_K=top_k,
        SCALE=weights is not None,
    )
    return out


def gather(x, indices, bin_ids, weights, bins, top_k):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_equal(indices.shape[0], x.shape[0] * top_k)
    assert_equal(bin_ids.shape[0], x.shape[0] * top_k)

    if weights is not None:
        assert_equal(weights.shape[0], x.shape[0] * top_k)

    # NOTE: There is no padding so the output rows equals the
    # input rows multiplied by top_k.
    output_rows = x.shape[0] * top_k
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
        A_TO_B=True,
        TOP_K=top_k,
        SCALE=weights is not None,
    )
    return out


def padded_scatter(x, indices, bin_ids, weights, bins, padded_bins, top_k):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_is_vector(padded_bins)
    assert_equal(indices.shape[0], bin_ids.shape[0])
    assert_equal(bins.size(), padded_bins.size())

    if weights is not None:
        assert_equal(indices.shape[0], weights.shape[0])

    tokens = indices.shape[0] // top_k
    out = torch.empty((tokens, top_k, x.shape[1]), dtype=x.dtype, device=x.device)
    _padded_copy[(indices.shape[0],)](
        out,
        x,
        indices,
        bin_ids,
        weights,
        bins,
        padded_bins,
        NUM_COLUMNS=x.shape[1],
        A_TO_B=False,
        TOP_K=top_k,
        SCALE=weights is not None,
    )

    # Reduce along the top-k dimension, if needed.
    return out.sum(dim=1) if top_k > 1 else out.view(tokens, x.shape[1])


def scatter(x, indices, bin_ids, weights, bins, top_k):
    return padded_scatter(x, indices, bin_ids, weights, bins, bins, top_k)


# x: (tokens, top_k, hidden_size), real
# grad: (tokens, hidden_size), real.
# wgrad: (tokens, top_k), real.
# indices: (tokens * top_k), integer.
# bin_ids: (tokens * top_k), integer.
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
def _padded_copy_wgrad(
    x,
    grad,
    wgrad,
    indices,
    bin_ids,
    bins,
    padded_bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    # Our index into 'tokens * top_k'.
    index_out = tl.load(indices + tl.program_id(0))

    # One threadblock per row in 'a'. Array 'b' has greater or equal
    # number of rows since they could be padded.
    bin_idx = tl.load(bin_ids + tl.program_id(0))

    # Now we know what bin we're assigned to, but we need to know how
    # many threadblocks were assigned to earlier bins so we can offset
    # in our bin properly.
    offset_in_bin = tl.program_id(0)
    if bin_idx > 0:
        offset_in_bin -= tl.load(bins + bin_idx - 1)

    # Load the starting index of our bin in array 'x'.
    index_x = offset_in_bin
    if bin_idx > 0:
        index_x += tl.load(padded_bins + bin_idx - 1)

    # Offset the input and output pointers.
    wgrad += index_out
    grad += tl.multiple_of((index_out // TOP_K) * NUM_COLUMNS, NUM_COLUMNS)
    x += tl.multiple_of(index_x * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

    acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        data = tl.load(x + offsets, mask=mask).to(tl.float32)
        scale = tl.load(grad + offsets, mask=mask).to(tl.float32)
        acc += data * scale
        offsets += BLOCK_X

    # Reduce to get the final result and store.
    out = tl.sum(acc).to(wgrad.dtype.element_ty)
    tl.store(wgrad, out)


def padded_scatter_wgrad(x, grad, indices, bin_ids, bins, padded_bins, top_k):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_matrix(grad)
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_is_vector(padded_bins)
    assert_equal(indices.shape[0], bin_ids.shape[0])
    assert_equal(bins.size(), padded_bins.size())

    tokens = indices.shape[0] // top_k
    out = torch.empty((tokens * top_k), dtype=x.dtype, device=x.device)
    _padded_copy_wgrad[(indices.shape[0],)](
        x,
        grad,
        out,
        indices,
        bin_ids,
        bins,
        padded_bins,
        NUM_COLUMNS=x.shape[1],
        TOP_K=top_k,
    )
    return out


def scatter_wgrad(x, grad, indices, bin_ids, bins, top_k):
    return padded_scatter_wgrad(x, grad, indices, bin_ids, bins, bins, top_k)


# a: (tokens, hidden_size), real.
# b: (num_experts, expert_capacity, num_columns), real.
# indices: (tokens * top_k), integer.
# weights: (tokens * top_k), real.
# bins: (num_experts), integer.
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
def _binned_copy(
    a,
    b,
    num_experts,
    expert_capacity,
    indices,
    weights,
    bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
    A_TO_B: tl.constexpr,
    SCALE: tl.constexpr,
):
    # Load our indices into the output.
    expert_idx = tl.program_id(0)
    entry_idx = tl.program_id(1)

    # Calculate our offset into the output.
    index_b = expert_idx * expert_capacity + entry_idx

    # Load the index bounds for our bin and calculate
    # the number of tokens assigned to our expert.
    start = 0
    if expert_idx > 0:
        start = tl.load(bins + expert_idx - 1)
    end = tl.load(bins + expert_idx)
    num_tokens = end - start

    # Calculate our offset into the input. If we don't
    # have an input exit early.
    if entry_idx >= num_tokens:
        return
    index_a = tl.load(indices + start + entry_idx)

    # Offset the input and output pointers.
    #
    # If we're going from A to B, divide the input index to copy
    # the same input repeatedly. If we're going from B to A we
    # need to reduce the result. Using atomics is slow, so we
    # do the reduce step in a second kernel.
    offset = index_a // TOP_K if A_TO_B else index_a
    a += tl.multiple_of(offset * NUM_COLUMNS, NUM_COLUMNS)
    b += tl.multiple_of(index_b * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

    # Load the scale, if requested.
    scale = tl.load(weights + index_a) if SCALE else 1

    # Swap the pointers depending on the direction.
    #
    # NOTE: We need to zero the output in both directions.
    iptr = a if A_TO_B else b
    optr = b if A_TO_B else a

    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        x = tl.load(iptr + offsets, mask=mask)
        x = x.to(tl.float32) * scale.to(tl.float32)

        tl.store(optr + offsets, x.to(optr.dtype.element_ty), mask=mask)

        offsets += BLOCK_X


def binned_gather(x, indices, weights, bins, expert_capacity, top_k):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bins)
    assert_equal(indices.shape[0], x.shape[0] * top_k)

    if weights is not None:
        assert_equal(weights.shape[0], x.shape[0] * top_k)

    num_experts = bins.shape[0]
    out = torch.zeros((num_experts, expert_capacity, x.shape[1]), dtype=x.dtype, device=x.device)

    _binned_copy[(num_experts, expert_capacity)](
        x,
        out,
        num_experts,
        expert_capacity,
        indices,
        weights,
        bins,
        NUM_COLUMNS=x.shape[1],
        A_TO_B=True,
        TOP_K=top_k,
        SCALE=weights is not None,
    )
    return out


def binned_scatter(x, indices, weights, bins, top_k):
    # Validate the input shapes.
    assert_is_tensor(x, 3)
    assert_is_vector(indices)
    assert_is_vector(bins)
    assert_equal(bins.shape[0], x.shape[0])

    if weights is not None:
        assert_equal(indices.shape[0], weights.shape[0])

    num_experts, expert_capacity, hidden_size = x.shape
    tokens = indices.shape[0] // top_k
    out = torch.zeros((tokens, top_k, hidden_size), dtype=x.dtype, device=x.device)
    _binned_copy[(num_experts, expert_capacity)](
        out,
        x,
        num_experts,
        expert_capacity,
        indices,
        weights,
        bins,
        NUM_COLUMNS=hidden_size,
        A_TO_B=False,
        TOP_K=top_k,
        SCALE=weights is not None,
    )

    # Reduce along the top-k dimension, if needed.
    return out.sum(dim=1) if top_k > 1 else out.view(tokens, hidden_size)


# a: (tokens, hidden_size), real.
# b: (num_experts, expert_capacity, num_columns), real.
# indices: (tokens * top_k), integer.
# weights: (tokens * top_k), real.
# bins: (num_experts), integer.
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
def _binned_copy_wgrad(
    x,
    grad,
    wgrad,
    num_experts,
    expert_capacity,
    indices,
    bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    # Load our indices into the output.
    expert_idx = tl.program_id(0)
    entry_idx = tl.program_id(1)

    # Calculate our offset into the output.
    index_x = expert_idx * expert_capacity + entry_idx

    # Load the index bounds for our bin and calculate
    # the number of tokens assigned to our expert.
    start = 0
    if expert_idx > 0:
        start = tl.load(bins + expert_idx - 1)
    end = tl.load(bins + expert_idx)
    num_tokens = end - start

    # Calculate our offset into the input. If we don't
    # have an input exit early.
    if entry_idx >= num_tokens:
        return
    index_out = tl.load(indices + start + entry_idx)

    # Offset the input and output pointers.
    wgrad += index_out
    grad += tl.multiple_of((index_out // TOP_K) * NUM_COLUMNS, NUM_COLUMNS)
    x += tl.multiple_of(index_x * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

    acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        data = tl.load(x + offsets, mask=mask).to(tl.float32)
        scale = tl.load(grad + offsets, mask=mask).to(tl.float32)
        acc += data * scale
        offsets += BLOCK_X

    # Reduce to get the final result and store.
    out = tl.sum(acc).to(wgrad.dtype.element_ty)
    tl.store(wgrad, out)


def binned_scatter_wgrad(x, grad, indices, bins, top_k):
    # Validate the input shapes.
    assert_is_tensor(x, 3)
    assert_is_matrix(grad)
    assert_is_vector(indices)
    assert_is_vector(bins)
    assert_equal(bins.shape[0], x.shape[0])

    num_experts, expert_capacity, hidden_size = x.shape
    tokens = indices.shape[0] // top_k
    out = torch.zeros((tokens * top_k), dtype=x.dtype, device=x.device)
    _binned_copy_wgrad[(num_experts, expert_capacity)](
        x,
        grad,
        out,
        num_experts,
        expert_capacity,
        indices,
        bins,
        NUM_COLUMNS=hidden_size,
        TOP_K=top_k,
    )
    return out


## UnPaddedMatMulOps

def is_cuda():
    if torch.cuda.is_available():
        # return triton.runtime.driver.active.get_current_target().backend == "cuda"
        return True
    return False


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def num_sms():
    if is_cuda():
        return torch.cuda.get_device_properties("cuda").multi_processor_count
    return 148

## Code taken from: https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html#sphx-glr-getting-started-tutorials-08-grouped-gemm-py. ##
@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 64,
            'NUM_SM': num_sms(),
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 64,
            'NUM_SM': num_sms(),
        }),
    ],
    key=['group_size'],
)
@triton.jit
def grouped_matmul_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # precision of activations and weights.
    activation: tl.constexpr,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            if activation == 'float16':
                a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
                b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
                c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            else:
                a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float32))
                b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float32))
                c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float32))
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                # tl.multiple_of(a_ptrs, [16, 16])
                # tl.multiple_of(b_ptrs, [16, 16])
                # assume full tile for now
                a = tl.load(a_ptrs, mask= \
                            ## Question, what on earth goes in here? ##
                            (offs_am[:, None] < gm)  \
                            & (offs_k[None, :] + kk*BLOCK_SIZE_K < k))
                b = tl.load(b_ptrs, mask = \
                            (offs_k[:, None] + kk*BLOCK_SIZE_K < k) \
                            & (offs_bn[None, :] < gn))
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(tl.float16)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

            # assumes full tile for now
            tl.store(c_ptrs, c, mask=\
                     (offs_cm[:, None] < gm) \
                     & (offs_cn[None, :] < gn))

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


## Group_a are tokens paced as [tokens routed to each expert, hidden_dimension].
## Group_B are experts packed as [num_experts, expert_hidden_dim]. -> hid_dim could differ between experts.
def group_gemm_fn(group_A, group_B, DEVICE):
    assert len(group_A) == len(group_B)
    group_size = len(group_A)

    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    for i in range(group_size):
        A = group_A[i]
        B = group_B[i]
        assert A.shape[1] == B.shape[0]
        M, K = A.shape
        K, N = B.shape
        C = torch.empty((M, N), device=DEVICE, dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]

    # note these are device tensors
    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)
    # we use a fixed number of CTA, and it's auto-tunable
    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_kernel[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
        activation="float16" if group_A[0].dtype == torch.float16  else "float32"
    )

    return group_C


## TODO(ahangupta): add support for Zeroed out experts. ##
def grouped_gemm(x: torch.Tensor, w: torch.Tensor,
                 sizes: list[tuple[int, int, int]]) -> torch.Tensor:
    assert len(x.shape) == 2, 'x should be 2-d tensor.'
    assert x.device == w[0].device, 'both tensors need to be on the same device.'
    ## First, we have to re-shape the input. ##
    slice_idxs = [i[0] for i in sizes]
    x = torch.split(x, slice_idxs)
    gemm_out = group_gemm_fn(x, w, x[0].device)
    ## This is for debugging only, remove once finished. ##
    ## We compare against pytorch ground-truth. For debugging only. ##
    # torch_out = [torch.matmul(xi, wi) for xi, wi in zip(x, w)]
    # for g_out, t_out in zip(gemm_out, torch_out):
    #     print(f'largest delta: {torch.abs(g_out - t_out).max().item()}')
    return torch.cat(gemm_out, dim=0)


if __name__ == '__main__':
    ## Here, we test the correctness of the grouped_gemm kernel. ##

    def test_case(ms: list[int], ns: list[int], ks: list[int], ty: torch.dtype):
        grp_B = []
        sizes = []
        cum_size_a = 0
        for one, two, three in zip(ms, ns, ks):
            cum_size_a += one
            b = torch.randn((three, two), dtype=ty, device="cuda" if torch.cuda.is_available() else "cpu")
            sizes.append((one, two, three))
            grp_B.append(b)

        return grouped_gemm(torch.randn(cum_size_a, ks[0], device="cuda" if torch.cuda.is_available() else "cpu", dtype=ty), grp_B, sizes)


    ## Since the most common case is expert_count = 4, we specifically test for that. ##
    ## We do a mXk by a kXn mat-mul. ##
    ## However, the model_dimension is kept constant through all ops. ##

    ## Tiny test case in float32 for debugging. ##
    ty : torch.dtype = torch.float32
    m = [16, 16, 16, 16]
    n = [16, 16, 16, 16]
    k = [16, 16, 16, 16]
    test_case(m, n, k, ty)

    # ## Smaller test-case.
    ty : torch.dtype = torch.float16
    m = [97, 75, 49, 60]
    n = [82, 82, 82, 82]  ## ffn_hidden_size.
    k = [54, 54, 54, 54]  ## model hidden size.
    test_case(m, n, k, ty)

    # ## Larger test-case. More realisitc.
    ty : torch.dtype = torch.float16
    m = [1024, 256, 512, 190]
    n = [512, 512, 512, 512]
    k = [150, 150, 150, 150]
    test_case(m, n, k, ty)

