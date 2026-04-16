import torch
import triton
import triton.language as tl

def get_num_sms():
    return torch.cuda.get_device_properties("cuda").multi_processor_count

autotune_configs = [
    # -------------------------------------------------------------------------
    # [FIX-3] LDS bank conflict mitigation via num_warps=4 with K=64:
    # On MI250X, 32 LDS banks × 4B = 128B/cycle. A 128-wide BF16 tile row is
    # exactly 128B — perfect alignment for 4-way bank conflicts when num_warps=8
    # sends all 8 wavefronts to LDS simultaneously. num_warps=4 halves the
    # concurrent pressure and shifts access timing, reducing conflict rate.
    # tl.arange requires power-of-2 sizes — non-power-of-2 K values are invalid.
    # -------------------------------------------------------------------------

    # K=64, num_warps=4 — primary LDS conflict mitigation + deep pipeline
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=4),
    # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=4),
    # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=4),

    # K=64, num_warps=8 — higher throughput if conflicts aren't the bottleneck
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=8),
    # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
    # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),

    # K=32 — narrower K reduces LDS pressure per access cycle
    # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=4),
    # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
    # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=8),
    # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
    # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
    # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),

    # K=128 — very wide K, fewest loop iterations, max MFMA reuse per fetch
    # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
    # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),

    # Fallback shallow-pipeline configs from v12
    # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
    # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
    # triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    # triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
]

# ==============================================================================
# 1. INDEX BUILDER
# ==============================================================================
# def build_index_tensors(splits, num_experts, device):
#     num_ranks = len(splits) // num_experts
#     offsets   = [0]
#     for s in splits[:-1]:
#         offsets.append(offsets[-1] + s)
#     indices, expert_starts, m_sizes = [], [],[]
#     for e in range(num_experts):
#         expert_starts.append(len(indices))
#         exp_m = 0
#         for r in range(num_ranks):
#             idx = r * num_experts + e
#             m   = splits[idx]
#             if m > 0:
#                 indices.extend(range(offsets[idx], offsets[idx] + m))
#                 exp_m += m
#         m_sizes.append(exp_m)
#     d_indices       = torch.tensor(indices,       dtype=torch.int32, device=device)
#     d_expert_starts = torch.tensor(expert_starts, dtype=torch.int32, device=device)
#     d_m_sizes       = torch.tensor(m_sizes,       dtype=torch.int32, device=device)
#     return d_indices, d_expert_starts, d_m_sizes
def build_index_tensors(output_splits_tensor, num_experts):
    """
    100% GPU-native index builder. Zero CPU syncs.
    output_splits_tensor: 1D GPU tensor of shape [num_ranks * num_experts]
    """
    device = output_splits_tensor.device
    num_ranks = output_splits_tensor.shape[0] // num_experts
    
    # 1. Calculate m_sizes: [num_experts]
    d_m_sizes = output_splits_tensor.reshape(num_ranks, num_experts).sum(dim=0).to(torch.int32)
    
    # 2. Calculate expert_starts: [num_experts]
    d_expert_starts = torch.cumsum(
        torch.cat([torch.zeros(1, dtype=torch.int32, device=device), d_m_sizes[:-1]]), 
        dim=0
    ).to(torch.int32)
    
    # 3. Build the Gather Map (d_indices)
    # Create an array mapping each chunk to its Expert ID
    expert_ids = torch.arange(num_experts, device=device).repeat(num_ranks)
    
    # Repeat the Expert ID by the number of tokens in that chunk
    token_expert_ids = torch.repeat_interleave(expert_ids, output_splits_tensor)
    
    # A stable sort groups all tokens by Expert ID, naturally yielding 
    # the original Rank-Primary indices!
    d_indices = torch.argsort(token_expert_ids, stable=True).to(torch.int32)
    
    return d_indices, d_expert_starts, d_m_sizes



# ==============================================================================
# 3. KERNEL 4v2 (GATHER-SCATTER GEMM)
# ==============================================================================
# @triton.autotune(configs=autotune_configs, key=['K', 'D'])
@triton.jit
def kernel4v2_forward(
    x_ptr, w_ptr, c_ptr, indices_ptr, expert_starts_ptr, m_sizes_ptr,
    K: tl.constexpr, D: tl.constexpr, NUM_EXPERTS, NUM_SMS: tl.constexpr,
    stride_am, stride_ad, stride_we, stride_wk, stride_wd, stride_cm, stride_ck,
    # BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 128,   # CHANGED: was plain param, now has default
    BLOCK_SIZE_N: tl.constexpr = 128,   # CHANGED
    BLOCK_SIZE_K: tl.constexpr = 64,    # CHANGED
    GROUP_SIZE_M: tl.constexpr = 8,     # CHANGED
):
    tidx = tl.program_id(0)
    processed_tiles = 0
    # for expert_idx in range(NUM_EXPERTS):
    expert_idx = 0
    while expert_idx < NUM_EXPERTS:
        m_size    = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
        exp_start = tl.load(expert_starts_ptr + expert_idx).to(tl.int32)
        if m_size > 0:
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(K, BLOCK_SIZE_N)
            num_tiles   = num_m_tiles * num_n_tiles
            while tidx >= processed_tiles and tidx < processed_tiles + num_tiles:
                tile_idx   = tidx - processed_tiles
                tile_m_idx = tile_idx // num_n_tiles
                tile_n_idx = tile_idx  % num_n_tiles
                logical_m  = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                mask_m     = logical_m < m_size
                physical_m = tl.load(indices_ptr + exp_start + logical_m, mask=mask_m, other=0)
                offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                mask_n = offs_n < K
                offs_k = tl.arange(0, BLOCK_SIZE_K)
                x_ptrs = x_ptr + physical_m[:, None] * stride_am + offs_k[None, :] * stride_ad
                w_ptrs = w_ptr + expert_idx * stride_we + offs_n[:, None] * stride_wk + offs_k[None, :] * stride_wd
                acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for _ in range(tl.cdiv(D, BLOCK_SIZE_K)):
                    mask_k = offs_k < D
                    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                    w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
                    acc    += tl.dot(x, w.T)
                    offs_k += BLOCK_SIZE_K
                    x_ptrs += BLOCK_SIZE_K * stride_ad
                    w_ptrs += BLOCK_SIZE_K * stride_wd
                c_ptrs = c_ptr + physical_m[:, None] * stride_cm + offs_n[None, :] * stride_ck
                tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])
                tidx += NUM_SMS
            processed_tiles += num_tiles
        expert_idx += 1

# @triton.autotune(configs=autotune_configs, key=['IN_DIM', 'OUT_DIM'])
@triton.jit
def kernel4v2_backward_dx(
    dy_ptr, w_ptr, dx_ptr, indices_ptr, expert_starts_ptr, m_sizes_ptr,
    IN_DIM: tl.constexpr, OUT_DIM: tl.constexpr, NUM_EXPERTS, NUM_SMS: tl.constexpr,
    stride_dym, stride_dyout, stride_we, stride_wout, stride_win, stride_dxm, stride_dxin,
    # BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 128,   # CHANGED
    BLOCK_SIZE_K: tl.constexpr = 128,   # CHANGED
    BLOCK_SIZE_N: tl.constexpr = 64,    # CHANGED
    GROUP_SIZE_M: tl.constexpr = 8,     # CHANGED
):
    tidx = tl.program_id(0)
    processed_tiles = 0
    # for expert_idx in range(NUM_EXPERTS):
    expert_idx = 0
    while expert_idx < NUM_EXPERTS:
        m_size    = tl.load(m_sizes_ptr      + expert_idx).to(tl.int32)
        exp_start = tl.load(expert_starts_ptr + expert_idx).to(tl.int32)
        if m_size > 0:
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_k_tiles = tl.cdiv(IN_DIM, BLOCK_SIZE_K)
            num_tiles   = num_m_tiles * num_k_tiles
            while tidx >= processed_tiles and tidx < processed_tiles + num_tiles:
                local_tile = tidx - processed_tiles
                tile_m_idx = local_tile // num_k_tiles
                tile_k_idx = local_tile  % num_k_tiles
                logical_m  = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                mask_m     = logical_m < m_size
                physical_m = tl.load(indices_ptr + exp_start + logical_m, mask=mask_m, other=0)
                offs_k = tile_k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                offs_n = tl.arange(0, BLOCK_SIZE_N)
                mask_k = offs_k < IN_DIM
                dy_ptrs = dy_ptr + physical_m[:, None] * stride_dym + offs_n[None, :] * stride_dyout
                w_ptrs  = w_ptr  + expert_idx * stride_we + offs_n[:, None] * stride_wout + offs_k[None, :] * stride_win
                acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
                for _ in range(tl.cdiv(OUT_DIM, BLOCK_SIZE_N)):
                    mask_n  = offs_n < OUT_DIM
                    dy = tl.load(dy_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
                    w  = tl.load(w_ptrs,  mask=mask_n[:, None] & mask_k[None, :], other=0.0)
                    acc     += tl.dot(dy, w)
                    offs_n  += BLOCK_SIZE_N
                    dy_ptrs += BLOCK_SIZE_N * stride_dyout
                    w_ptrs  += BLOCK_SIZE_N * stride_wout
                dx_ptrs = dx_ptr + physical_m[:, None] * stride_dxm + offs_k[None, :] * stride_dxin
                tl.store(dx_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_k[None, :])
                tidx += NUM_SMS
            processed_tiles += num_tiles
        expert_idx += 1

# @triton.autotune(configs=autotune_configs, key=['OUT_DIM', 'IN_DIM'])
@triton.jit
def kernel4v2_backward_dw(
    x_ptr, dy_ptr, dw_ptr, indices_ptr, expert_starts_ptr, m_sizes_ptr,
    OUT_DIM: tl.constexpr, IN_DIM: tl.constexpr, NUM_EXPERTS, NUM_SMS: tl.constexpr,
    stride_am, stride_ain, stride_dym, stride_dyout, stride_we, stride_wout, stride_win,
    # BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr = 128,   # CHANGED
    BLOCK_SIZE_K: tl.constexpr = 128,   # CHANGED
    BLOCK_SIZE_M: tl.constexpr = 64,    # CHANGED
    GROUP_SIZE_M: tl.constexpr = 8,     # CHANGED
):
    tidx = tl.program_id(0)
    num_n_tiles      = tl.cdiv(OUT_DIM, BLOCK_SIZE_N)
    num_k_tiles      = tl.cdiv(IN_DIM,  BLOCK_SIZE_K)
    tiles_per_expert = num_n_tiles * num_k_tiles
    total_tiles      = NUM_EXPERTS * tiles_per_expert
    for global_tile in range(tidx, total_tiles, NUM_SMS):
        expert_idx = global_tile // tiles_per_expert
        rem        = global_tile  % tiles_per_expert
        tile_n_idx = rem // num_k_tiles
        tile_k_idx = rem  % num_k_tiles
        m_size    = tl.load(m_sizes_ptr      + expert_idx).to(tl.int32)
        exp_start = tl.load(expert_starts_ptr + expert_idx).to(tl.int32)
        offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tile_k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_n = offs_n < OUT_DIM
        mask_k = offs_k < IN_DIM
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
        if m_size > 0:
            for m_step in range(tl.cdiv(m_size, BLOCK_SIZE_M)):
                logical_m  = m_step * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                mask_m     = logical_m < m_size
                physical_m = tl.load(indices_ptr + exp_start + logical_m, mask=mask_m, other=0)
                dy_ptrs = dy_ptr + offs_n[:, None] * stride_dyout + physical_m[None, :] * stride_dym
                x_ptrs  = x_ptr  + physical_m[:, None] * stride_am  + offs_k[None, :] * stride_ain
                dy_T = tl.load(dy_ptrs, mask=mask_n[:, None] & mask_m[None, :], other=0.0)
                x    = tl.load(x_ptrs,  mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                acc += tl.dot(dy_T, x)
        dw_ptrs = dw_ptr + expert_idx * stride_we + offs_n[:, None] * stride_wout + offs_k[None, :] * stride_win
        tl.store(dw_ptrs, acc.to(tl.bfloat16), mask=mask_n[:, None] & mask_k[None, :])

# ==============================================================================
# 4. AUTOGRAD WRAPPER
# ==============================================================================
class Kernel4v2_GEMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, d_indices, d_expert_starts, d_m_sizes):
        E, OUT_DIM, IN_DIM = weights.shape
        Total  = inputs.shape[0]
        output = torch.empty((Total, OUT_DIM), device='cuda', dtype=inputs.dtype)
        NUM_SMS = get_num_sms()
        if d_indices.numel() > 0:
            kernel4v2_forward[(NUM_SMS,)](
                inputs, weights, output, d_indices, d_expert_starts, d_m_sizes,
                OUT_DIM, IN_DIM, E, NUM_SMS,
                inputs.stride(0), inputs.stride(1), weights.stride(0), weights.stride(1), weights.stride(2),
                output.stride(0), output.stride(1),
                num_warps=4, num_stages=1,   # <-- ADD THIS
            )
        ctx.save_for_backward(inputs, weights, d_indices, d_expert_starts, d_m_sizes)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights, d_indices, d_expert_starts, d_m_sizes = ctx.saved_tensors
        E, OUT_DIM, IN_DIM = weights.shape
        NUM_SMS = get_num_sms()
        grad_inputs  = torch.zeros_like(inputs)
        # grad_weights = torch.empty_like(weights)
        grad_weights = torch.zeros_like(weights)
        if d_indices.numel() > 0:
            kernel4v2_backward_dx[(NUM_SMS,)](
                grad_output, weights, grad_inputs, d_indices, d_expert_starts, d_m_sizes,
                IN_DIM, OUT_DIM, E, NUM_SMS,
                grad_output.stride(0), grad_output.stride(1), weights.stride(0), weights.stride(1), weights.stride(2),
                grad_inputs.stride(0), grad_inputs.stride(1),
                num_warps=4, num_stages=1,   # CHANGED: added these two args
            )
            kernel4v2_backward_dw[(NUM_SMS,)](
                inputs, grad_output, grad_weights, d_indices, d_expert_starts, d_m_sizes,
                OUT_DIM, IN_DIM, E, NUM_SMS,
                inputs.stride(0), inputs.stride(1), grad_output.stride(0), grad_output.stride(1),
                grad_weights.stride(0), grad_weights.stride(1), grad_weights.stride(2),
                num_warps=4, num_stages=1,   # <-- ADD THIS
            )
        else:
            grad_weights.zero_()
        return grad_inputs, grad_weights, None, None, None







