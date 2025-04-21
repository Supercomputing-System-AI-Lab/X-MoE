# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

# from .utils import print_rank

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
def reconstruct(
    out,
    s1_x,
    s1_expert_ids,
    s1_bins,
    s1_len, 
    s1_tokens_per_expert,
    s2_x,
    s2_expert_ids,
    s2_bins,
    total_bins,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_X: tl.constexpr,
    A_TO_B: tl.constexpr,
    RBD: tl.constexpr,
):
    
    pid = tl.program_id(0)
    if_s1 = True if pid < s1_len else False

    index_a = pid if if_s1 else pid - s1_len

    if RBD:
        _expert_ids = s1_expert_ids if if_s1 else s2_expert_ids
        _bins = s1_bins if if_s1 else s2_bins
    else:
        _expert_ids = s1_expert_ids
        _bins = s1_bins

    expert_id = tl.load(_expert_ids + tl.program_id(0))

    offset_in_curr_stage = tl.program_id(0)
    if expert_id > 0:
        offset_in_curr_stage -= tl.load(_bins + expert_id - 1)

    index_b = offset_in_curr_stage
    if expert_id > 0:
        index_b += tl.load(total_bins + expert_id - 1)
    if RBD:
        index_b += tl.load(s1_tokens_per_expert + expert_id)


    if RBD:
        a = s1_x if if_s1 else s2_x
    else:
        a = s1_x
    b = out

    a += tl.multiple_of(index_a * NUM_COLUMNS, NUM_COLUMNS)
    b += tl.multiple_of(index_b * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

    
    iptr = a if A_TO_B else b
    optr = b if A_TO_B else a

    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)

    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        x = tl.load(iptr + offsets, mask=mask)
        # x = x.to(tl.float32) * scale.to(tl.float32)
        tl.store(optr + offsets, x.to(optr.dtype.element_ty), mask=mask)
        offsets += BLOCK_X


def expert_input_reconstruction(x, expert_ids, bins, rbd=False, s2_x=None, s2_expert_ids=None, s2_bins=None, s1_tokens_per_expert=None):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(expert_ids)
    assert_is_vector(bins)

    output_rows = x.shape[0]
    out = torch.zeros((output_rows, x.shape[1]), dtype=x.dtype, device=x.device)
    reconstruct[(output_rows,)](
        out,
        x,
        expert_ids,
        bins,
        output_rows,
        s1_tokens_per_expert, # tokens per expert
        s2_x,
        s2_expert_ids,
        s2_bins,
        bins,
        NUM_COLUMNS=x.shape[1],
        A_TO_B=True,
        RBD=rbd,
    )
    return out

def expert_input_recover(x, expert_ids, bins, rbd=False, s2_x=None, s2_expert_ids=None, s2_bins=None, s1_tokens_per_expert=None):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(expert_ids)
    assert_is_vector(bins)

    output_rows = x.shape[0]
    out = torch.zeros((output_rows, x.shape[1]), dtype=x.dtype, device=x.device)
    reconstruct[(output_rows,)](
        out,
        x,
        expert_ids,
        bins,
        output_rows,
        s1_tokens_per_expert, # tokens per expert
        s2_x,
        s2_expert_ids,
        s2_bins,
        bins,
        NUM_COLUMNS=x.shape[1],
        A_TO_B=False,
        RBD=rbd,
    )
    return out




def generate_mock_inputs(num_tokens=8, hidden_size=16, num_experts=4):
    torch.manual_seed(0)

    x = torch.randn((num_tokens, hidden_size), device='cuda', dtype=torch.float16)
    expert_ids = torch.randint(low=0, high=num_experts, size=(num_tokens,), device='cuda', dtype=torch.int32)

    sorted_expert_ids, sorted_indices = torch.sort(expert_ids)
    x_sorted = x[sorted_indices]

    bins = torch.zeros(num_experts, dtype=torch.int32, device='cuda')
    for i in range(num_experts):
        bins[i] = torch.sum(sorted_expert_ids == i)
    cumsum_bins = torch.cumsum(bins, dim=0)

    return x_sorted, x, sorted_expert_ids, bins, cumsum_bins.int()


def main():
    num_tokens = 4096
    hidden_size = 7168
    num_experts = 4

    x_sorted, x, expert_ids, tokens_per_experts, bins = generate_mock_inputs(num_tokens, hidden_size, num_experts)

    print("Running expert_input_reconstruction...")
    print(f"Shape: {x_sorted.shape}, dtype: {x_sorted.dtype}, device: {x_sorted.device}")

    # Warm-up
    for _ in range(5):
        _ = expert_input_reconstruction(x_sorted, expert_ids, bins)

    # Timing with torch.cuda.Event
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()

    out = expert_input_reconstruction(x, expert_ids, bins)
    recovered_in = expert_input_recover(out, expert_ids, bins)

    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)

    print(f"\nTime elapsed: {elapsed_time_ms:.3f} ms")

    # Check correctness
    if torch.allclose(out, x_sorted, atol=1e-2, rtol=1e-2) and torch.allclose(recovered_in, x_sorted, atol=1e-2, rtol=1e-2):
        print("✅ Test passed! Output matches the input.")
    else:
        print("❌ Test failed! Output does not match the input.")
        print("out Difference max:", (out - x_sorted).abs().max().item())
        print("recovered in Difference max:", (out - x_sorted).abs().max().item())


if __name__ == "__main__":
    main()