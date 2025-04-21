import torch
import time
from deepspeed import comm as dist
from typing import Optional

# from .utils import create_group, print_rank, distributed_setup
from .utils import create_group, print_rank, distributed_setup
from deepspeed.moe.v2opt.rbd_kernels import _map, s2_order
from deepspeed.moe.v2opt.rbd_gating import LocalOrderOp, LocalOrderRecoverOp
# from deepspeed.moe.moe_v2 import _AllToAllSingle
from .a2a_single import _AllToAllSingle
from deepspeed.moe.sharded_moe import _AllToAll


# print(indices)
# print(bin_ids)

def map_dup_tokens(indices, bin_ids, n_experts, k, mesh_size, num_local_experts, permutation=None, verification=False):

    permutation = (permutation if permutation is not None else torch.arange(indices.shape[0], device=indices.device, dtype=torch.int32))

    indices_permuted = indices[permutation]
    bin_ids_permuted = bin_ids[permutation]

    token_indices = indices_permuted // k
    mesh_indices = bin_ids_permuted // (mesh_size * num_local_experts)

    x = mesh_indices * (indices.shape[0] * mesh_size * num_local_experts) + token_indices

    unique_sorted, inv_sorted = torch.unique(x, return_inverse=True, sorted=False)

    # unique_sorted[inv_sorted] == x

    first_occ_in_permuted = torch.full_like(unique_sorted, x.shape[0]).to(torch.int32)
    # scatter_reduce(…, reduce="amin") requires PyTorch >= 2.0
    position_in_permuted = torch.arange(indices.shape[0], device=indices.device, dtype=torch.int32)
    first_occ_in_permuted = first_occ_in_permuted.scatter_reduce(0, inv_sorted, position_in_permuted, reduce="amin")

    s2_in_permuted_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
    s2_in_permuted_mask[first_occ_in_permuted] = False
    inv_sorted_s2 = inv_sorted[s2_in_permuted_mask] # unique_sorted -> duplicated x
    positions_s2_in_permuted = torch.where(s2_in_permuted_mask)[0]


    sorted_first_occur_in_permuted, order_by_first_occur = torch.sort(first_occ_in_permuted)

    inv_order_by_first_occur = torch.empty_like(order_by_first_occur)
    inv_order_by_first_occur[order_by_first_occur] = torch.arange(order_by_first_occur.size(0), device=x.device)

    # map_indices = inv_order_by_first_occur[inv_sorted]
    s1_to_s2_indices = inv_order_by_first_occur[inv_sorted_s2]

    # final_idx = first_occ_in_permuted[stable_order]
    s1_indices = indices_permuted[sorted_first_occur_in_permuted]
    s1_bin_ids = bin_ids_permuted[sorted_first_occur_in_permuted]

    s2_bin_ids = bin_ids_permuted[positions_s2_in_permuted] 


    # comb_input = [(index.item(), bin_id.item()) for index, bin_id in zip(indices, bin_ids)]
    # print("before permute")
    # print(comb_input)
    # comb_input = [(index.item(), bin_id.item()) for index, bin_id in zip(indices_permuted, bin_ids_permuted)]
    # print("after permute")
    # print(comb_input)

    # comb_ = s1_indices[s1_to_s2_indices], s1_bin_ids[s1_to_s2_indices], \
    #          indices_permuted[positions_s2_in_permuted], bin_ids_permuted[positions_s2_in_permuted]
    # print([f"({i_s1.item()}, {b_s1.item()}), ({i_s2.item()}, {b_s2.item()}))" for i_s1, b_s1, i_s2, b_s2 in zip(s1_indices[s1_to_s2_indices], s1_bin_ids[s1_to_s2_indices], \
    #          indices_permuted[positions_s2_in_permuted], bin_ids_permuted[positions_s2_in_permuted])])
    # corresponding_s1_bin_ids = s1_bin_ids[s1_to_s2_indices]
    # corresponding_s2_bin_ids = bin_ids_permuted[positions_s2_in_permuted]
    # comb = [(index.item(), s1_bin_id.item(), s2_bin_id.item()) for index, s1_bin_id, s2_bin_id in zip(s1_to_s2_indices, corresponding_s1_bin_ids, corresponding_s2_bin_ids)]


    # start from now
    s2_corresponding_s1_bin_ids = s1_bin_ids[s1_to_s2_indices]
    s2_encoding = s2_corresponding_s1_bin_ids * n_experts + s2_bin_ids
    _, s2_corresponding_s1_bin_ids_order = torch.sort(s2_encoding)
    s1_bin_ids, s1_bin_ids_order = torch.sort(s1_bin_ids)
    s1_indices = s1_indices[s1_bin_ids_order]

    inverse_s1_order = torch.empty_like(s1_bin_ids_order)
    inverse_s1_order[s1_bin_ids_order] = torch.arange(len(inverse_s1_order), device=inverse_s1_order.device, dtype=inverse_s1_order.dtype)
    s2_bin_ids = s2_bin_ids[s2_corresponding_s1_bin_ids_order]
    s1_to_s2_indices = inverse_s1_order[s1_to_s2_indices[s2_corresponding_s1_bin_ids_order]]



    # s1_bin_ids, s1_bin_ids_order = torch.sort(s1_bin_ids)
    # s1_indices = s1_indices[s1_bin_ids_order]
    # s2_bin_ids, s2_indices = torch.sort(s2_bin_ids)
    # inverse_s1_order = torch.empty_like(s1_bin_ids_order)

    # inverse_s1_order[s1_bin_ids_order] = torch.arange(len(inverse_s1_order), device=inverse_s1_order.device, dtype=inverse_s1_order.dtype)
    # s1_to_s2_indices = inverse_s1_order[s1_to_s2_indices[s2_indices]]

    # verification:
    if verification:
        assert s2_bin_ids.shape == s1_to_s2_indices.shape
        comb = [(index.item(), bin_id.item()) for index, bin_id in zip(indices, bin_ids)]
        comb_s1 = [(index.item(), bin_id.item()) for index, bin_id in zip(s1_indices, s1_bin_ids)]
        assert all(elem in comb for elem in comb_s1)

        sorted_total_x, _ = torch.sort(torch.cat((s1_bin_ids, s2_bin_ids)))
        sorted_x, _ = torch.sort(bin_ids)
        assert torch.equal(sorted_total_x, sorted_x)
        assert torch.equal(s1_indices[s1_to_s2_indices] // k, 
                            indices_permuted[positions_s2_in_permuted][s2_corresponding_s1_bin_ids_order] // k)
        assert not torch.equal(s1_indices[s1_to_s2_indices], 
                            indices_permuted[positions_s2_in_permuted][s2_corresponding_s1_bin_ids_order])
        
        print("token_filter verification passed.")

    tokens_per_experts_s2_virtual = torch.histc(s2_corresponding_s1_bin_ids, n_experts, 0, n_experts - 1) 

    s2_indices = indices_permuted[positions_s2_in_permuted][s2_corresponding_s1_bin_ids_order]

    return s1_indices, s2_indices, s1_bin_ids, s2_bin_ids, s1_to_s2_indices, tokens_per_experts_s2_virtual

def get_a2a_splits(input_splits_tensor: torch.Tensor, group: dist.ProcessGroup, num_local_experts: Optional[int] = None):
    output_splits_tensor = _AllToAll.apply(group, input_splits_tensor)
    if num_local_experts is not None:
        input_splits = input_splits_tensor.view(-1, num_local_experts).sum(dim=1).tolist()
        output_splits = output_splits_tensor.view(-1, num_local_experts).sum(dim=1).tolist()
    else:
        input_splits = input_splits_tensor.tolist()
        output_splits = output_splits_tensor.tolist()
    return input_splits, output_splits, output_splits_tensor

def get_bins(bin_ids, num_experts):
    tokens_per_expert = torch.histc(bin_ids, num_experts, 0, num_experts - 1)

    bins = torch.cumsum(tokens_per_expert, 0)
    assert bins is not None
    bins = bins.view(1) if not len(bins.size()) else bins

    return bins.to(torch.int32), tokens_per_expert

def map_s1_to_s2_indices_to_relative(bins, indices):
    bin_idx = torch.bucketize(indices, bins, right=True) - 1 
    valid = (bin_idx >= 0) & (bin_idx < len(bins) - 1)
    adjusted = indices.clone()
    adjusted[valid] -= bins[bin_idx[valid]]
    return adjusted

# s1_tokens_per_experts, s2_tokens_per_experts: output splits.
def map_s1_to_s2_indices_relative_to_abs(adjusted, s1_tokens_per_experts, s2_tokens_per_experts) -> torch.Tensor:
    s1_bins = torch.cumsum(s1_tokens_per_experts, 0)
    x = torch.arange(s2_tokens_per_experts.shape[0], device=s2_tokens_per_experts.device, dtype=s2_tokens_per_experts.dtype)
    s2_indices = torch.torch.repeat_interleave(x, s2_tokens_per_experts)
    s1_bins = torch.cat([s1_bins, torch.zeros([1], device=s1_bins.device, dtype=s1_bins.dtype)])
    return adjusted + s1_bins[s2_indices - 1]


def rbd_metadata(indices, bin_ids, num_experts, top_k, mesh_size, num_local_experts, ep_group, local_group, permutation=None, verification=False):

    permutation = (permutation if permutation is not None else torch.arange(indices.shape[0], device=indices.device, dtype=torch.int32))
    indices_s1, indices_s2, bin_ids_s1, bin_ids_s2_virtual, s1_to_s2_virtual_indices, tokens_per_experts_s2_virtual = \
        map_dup_tokens(indices, bin_ids, num_experts, top_k, mesh_size, num_local_experts, permutation, verification)

    # s1 global a2a metadata
    bins_s1, s1_tokens_per_expert = get_bins(bin_ids_s1, num_experts)
    # print(s1_tokens_per_expert)
    # print(s1_tokens_per_expert.dtype)
    ins_s1, outs_s1, tokens_per_expert_s1_exp = get_a2a_splits(s1_tokens_per_expert, ep_group, num_local_experts)

    s1_to_s2_virtual_indices = map_s1_to_s2_indices_to_relative(bins_s1, s1_to_s2_virtual_indices)

    # virtual dispatching
    ins_s2_virtual, outs_s2_virtual, tokens_per_experts_s2 = get_a2a_splits(tokens_per_experts_s2_virtual, ep_group, num_local_experts)

    s1exp_to_s2_indices = _AllToAllSingle.apply(ep_group, s1_to_s2_virtual_indices, ins_s2_virtual, outs_s2_virtual)
    bin_ids_s2 = _AllToAllSingle.apply(ep_group, bin_ids_s2_virtual, ins_s2_virtual, outs_s2_virtual) 

    s1exp_to_s2_indices = map_s1_to_s2_indices_relative_to_abs(s1exp_to_s2_indices, tokens_per_expert_s1_exp, tokens_per_experts_s2)

    # sort bin_ids_s2 to ensure when s1_to_s2 gather, the s2 bin_ids is in correct order
    bin_ids_s2, bin_ids_s2_indices = torch.sort(bin_ids_s2)
    s1exp_to_s2_indices = s1exp_to_s2_indices[bin_ids_s2_indices]

    # s2 local a2a metadata
    local_expert_idx = ((dist.get_rank() // mesh_size) * num_local_experts * mesh_size) % num_experts

    tokens_per_experts_s2_ = torch.histc(bin_ids_s2, mesh_size*num_local_experts, local_expert_idx, local_expert_idx + mesh_size*num_local_experts - 1) 

    bins_s2 = torch.cumsum(tokens_per_experts_s2_, 0).to(torch.int32)
    bins_s2 = bins_s2.view(1) if not len(bins_s2.size()) else bins_s2

    ins_s2, outs_s2, tokens_per_experts_s2_exp = get_a2a_splits(tokens_per_experts_s2_, local_group, num_local_experts)
    # outs_s2 = _AllToAll.apply(local_group, ins_s2)

    bin_ids_s2 = bin_ids_s2 - local_expert_idx

    # s2 dispatching: s1exp_to_s2_indices, s2_bin_ids, 
    return (indices_s1, indices_s2, bin_ids_s1, bins_s1, ins_s1, outs_s1, ins_s2_virtual, outs_s2_virtual, \
             s1exp_to_s2_indices, bin_ids_s2, bins_s2, ins_s2, outs_s2, tokens_per_expert_s1_exp, tokens_per_experts_s2_exp)

def rbd_metadata_opt(indices, bin_ids, num_experts, top_k, mesh_size, num_local_experts, ep_group, local_group, permutation=None, verification=False):

    permutation = (permutation if permutation is not None else torch.arange(indices.shape[0], device=indices.device, dtype=torch.int32))
    indices_s1, indices_s2, bin_ids_s1, bin_ids_s2_virtual, s1_to_s2_virtual_indices, tokens_per_experts_s2_virtual = \
        map_dup_tokens(indices, bin_ids, num_experts, top_k, mesh_size, num_local_experts, permutation, verification)

    # s1 global a2a metadata
    bins_s1, s1_tokens_per_expert = get_bins(bin_ids_s1, num_experts)
    ins_s1, outs_s1, tokens_per_expert_s1_exp = get_a2a_splits(s1_tokens_per_expert, ep_group, num_local_experts)

    s1_to_s2_virtual_indices = map_s1_to_s2_indices_to_relative(bins_s1, s1_to_s2_virtual_indices)

    # virtual dispatching
    ins_s2_virtual, outs_s2_virtual, tokens_per_experts_s2 = get_a2a_splits(tokens_per_experts_s2_virtual, ep_group, num_local_experts)

    s1exp_to_s2_indices = _AllToAllSingle.apply(ep_group, s1_to_s2_virtual_indices, ins_s2_virtual, outs_s2_virtual)
    bin_ids_s2 = _AllToAllSingle.apply(ep_group, bin_ids_s2_virtual, ins_s2_virtual, outs_s2_virtual) 

    s1exp_to_s2_indices = map_s1_to_s2_indices_relative_to_abs(s1exp_to_s2_indices, tokens_per_expert_s1_exp, tokens_per_experts_s2)

    # sort bin_ids_s2 to ensure when s1_to_s2 gather, the s2 bin_ids is in correct order
    bin_ids_s2, bin_ids_s2_indices = torch.sort(bin_ids_s2)
    s1exp_to_s2_indices = s1exp_to_s2_indices[bin_ids_s2_indices]

    # s2 local a2a metadata
    local_expert_idx = (dist.get_rank() // mesh_size) * num_local_experts * mesh_size

    tokens_per_experts_s2_ = torch.histc(bin_ids_s2, mesh_size*num_local_experts, local_expert_idx, local_expert_idx + mesh_size*num_local_experts - 1) 

    bins_s2 = torch.cumsum(tokens_per_experts_s2_, 0).to(torch.int32)
    bins_s2 = bins_s2.view(1) if not len(bins_s2.size()) else bins_s2

    ins_s2, outs_s2, tokens_per_experts_s2_exp = get_a2a_splits(tokens_per_experts_s2_, local_group, num_local_experts)
    # outs_s2 = _AllToAll.apply(local_group, ins_s2)

    # s2 dispatching: s1exp_to_s2_indices, s2_bin_ids, 
    return (indices_s1, indices_s2, bin_ids_s1, bins_s1, ins_s1, outs_s1, ins_s2_virtual, outs_s2_virtual, \
             s1exp_to_s2_indices, bin_ids_s2, bins_s2, ins_s2, outs_s2, tokens_per_expert_s1_exp, tokens_per_experts_s2_exp)


def compare(a, b, path=""):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if not torch.equal(a, b):
            print(f"Mismatch at {path}: Tensors differ\n{a}\n!=\n{b}")
            return False
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            print(f"Mismatch at {path}: List lengths differ ({len(a)} != {len(b)})")
            return False
        for i, (x, y) in enumerate(zip(a, b)):
            if not compare(x, y, f"{path}[{i}]"):
                return False
    else:
        if a != b:
            print(f"Mismatch at {path}: {a} != {b}")
            return False
    return True

def compare_metadata(meta1, meta2):
    if len(meta1) != len(meta2):
        print(f"Metadata length mismatch: {len(meta1)} != {len(meta2)}")
        return
    for i, (a, b) in enumerate(zip(meta1, meta2)):
        if not compare(a, b, path=f"[{i}]"):
            return
    print("All metadata entries match.")


def main():
    seq_len = 4096
    top_k = 8
    iterations = 100
    mesh_size = 8
    num_local_experts = 4
    num_experts = 64
    device = 'cuda'

    torch.manual_seed(1)
    indices = torch.randint(0, seq_len * top_k, [seq_len * top_k], device='cuda', dtype=torch.int32)
    bin_ids = torch.arange(0, num_experts, device='cuda', dtype=torch.int32).unsqueeze(1).repeat(1, seq_len * top_k // num_experts).reshape([top_k * seq_len])
    # bin_ids = torch.arange(0, top_k, device='cuda', dtype=torch.int32)
    permutation = torch.randperm(seq_len * top_k, device='cuda', dtype=torch.int32)

    print("setup...")
    distributed_setup()
    ep_group = create_group()
    local_group = create_group(mesh_size, do_test=True)
    print("finish setup...")

    metadata = rbd_metadata(indices, bin_ids, num_experts, top_k, mesh_size, num_local_experts, ep_group, local_group, permutation, verification=False)
    print("???")
    metadata_opt = rbd_metadata_opt(indices, bin_ids, num_experts, top_k, mesh_size, num_local_experts, ep_group, local_group, permutation, verification=False)

    compare_metadata(metadata, metadata_opt)

    metadata = rbd_metadata(indices, bin_ids, num_experts, top_k, mesh_size, num_local_experts, ep_group, local_group, verification=False)
    metadata = rbd_metadata_opt(indices, bin_ids, num_experts, top_k, mesh_size, num_local_experts, ep_group, local_group, verification=False)

    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.time()
        metadata = rbd_metadata(indices, bin_ids, num_experts, top_k, mesh_size, num_local_experts, ep_group, local_group, verification=False)
        torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)

    avg_time = sum(times) / iterations
    print(f"Average original metadata time over {iterations} runs: {avg_time:.3f} ms")

    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.time()
        metadata = rbd_metadata_opt(indices, bin_ids, num_experts, top_k, mesh_size, num_local_experts, ep_group, local_group, verification=False)
        torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)

    avg_time = sum(times) / iterations
    print(f"Average optimized metadata time over {iterations} runs: {avg_time:.3f} ms")


    # comb_s1 = [(index.item(), bin_ids_s1.item()) for index, bin_ids_s1 in zip(torch.arange(bin_ids_s1.shape[0]), bin_ids_s1)]
    # print(comb_s1)

    # corresponding_s1_bin_ids = bin_ids_s1[s1exp_to_s2_indices]
    # comb = [(index.item(), s1_bin_id.item(), s2_bin_id.item()) for index, s1_bin_id, s2_bin_id in zip(s1exp_to_s2_indices, corresponding_s1_bin_ids, bin_ids_s2)]
    # print(comb)

    # print(tokens_per_experts_s2_virtual)
    # tokens_per_experts_s2_virtual_ = tokens_per_experts_s2_virtual


    # bins_s1, s1_tokens_per_expert = get_bins(s1_bin_ids, num_experts)
    # print(bins_s1)
    # print(map_indices_s2)

    # s1_to_s2_indices_rel = map_s1_to_s2_indices_to_relative(bins_s1, map_indices_s2)
    # print(s1_to_s2_indices_rel)


    # s1_to_s2_indices_abs = map_s1_to_s2_indices_relative_to_abs(s1_to_s2_indices_rel, s1_tokens_per_expert, tokens_per_experts_s2_virtual_)
    # print(s1_to_s2_indices_abs)


    # Timed iterations
    # times = []
    # for _ in range(iterations):
    #     x = torch.randint(0, seq_len * top_k, [seq_len * top_k], device='cuda', dtype=torch.int32)
    #     torch.cuda.synchronize()
    #     start = time.time()
    #     s1_indices, s1_bin_ids, s2_bin_ids, map_indices_s2, tokens_per_experts_s2_virtual = map_dup_tokens(indices, bin_ids, num_experts, top_k, mesh_size, num_local_experts, permutation)
        
    #     torch.cuda.synchronize()
    #     end = time.time()
    #     times.append((end - start) * 1000)

    # avg_time = sum(times) / iterations
    # print(f"Average unique time over {iterations} runs: {avg_time:.3f} ms")
    # 0.812 ms


if __name__ == "__main__":
    main()
