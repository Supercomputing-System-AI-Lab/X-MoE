import torch
from typing import Tuple, Any, Optional

from deepspeed.moe.v2opt.drop_gating import indices_and_bins_and_drop, gather_with_token_drop, scatter_with_token_drop

from deepspeed import comm as dist

from .utils import print_rank
from deepspeed.moe.v2opt.rbd_gating import LocalOrderOp, LocalOrderRecoverOp, s1_to_s2_gather, s2_to_s1_scatter
# from deepspeed.moe.moe_v2 import _AllToAllSingle
from deepspeed.moe.sharded_moe import _AllToAll

from .metadata import rbd_metadata
from .a2a_single import _AllToAllSingle

import time

class RBDispatcher(torch.nn.Module):
    def __init__(self, num_experts: int, top_k: int, num_local_experts: int, mesh_size: int, ep_group: dist.ProcessGroup, local_group: dist.ProcessGroup):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.mesh_size = mesh_size

        self.ep_group = ep_group
        self.local_group = local_group

    def _start_timer(self, enabled):
        if enabled:
            torch.cuda.synchronize()
            dist.barrier()
            return time.time()
        return None

    def _stop_timer(self, start_time):
        if start_time is None:
            return None
        torch.cuda.synchronize()
        dist.barrier()
        return time.time() - start_time

    def forward(self, x: torch.Tensor, n_tokens: int, metadata, time_breakdown=False):

        indices_s1, indices_s2, bin_ids_s1, bins_s1, ins_s1, outs_s1, ins_s2_virtual, outs_s2_virtual, \
                s1exp_to_s2_indices, bin_ids_s2, bins_s2, ins_s2, outs_s2, tokens_per_expert_s1_exp, tokens_per_experts_s2_exp = metadata

        st_gather = self._start_timer(time_breakdown)
        dx_s1 = gather_with_token_drop(x, indices_s1, bin_ids_s1, bins_s1, n_tokens, self.top_k)
        t_gather = self._stop_timer(st_gather)

        st_a2a_g = self._start_timer(time_breakdown)
        dx_s1_exp = _AllToAllSingle.apply(self.ep_group, dx_s1, ins_s1, outs_s1)
        t_a2a_g = self._stop_timer(st_a2a_g)

        st_s1tos2 = self._start_timer(time_breakdown)
        dx_s2 = s1_to_s2_gather(dx_s1_exp, s1exp_to_s2_indices, bin_ids_s2, bins_s2) # TODO
        t_s1tos2 = self._stop_timer(st_s1tos2)

        # indices_s1_exp = _AllToAllSingle.apply(self.ep_group, indices_s1, ins_s1, outs_s1)
        # pair_s1 = [(index.item(), bin_id.item()) for index, bin_id in zip(indices_s1, bin_ids_s1)] 
        # print_rank(pair_s1, "pair_s1")
        # pair_s2 = [(index.item(), bin_id.item()) for index, bin_id in zip(indices_s1_exp[s1exp_to_s2_indices], bin_ids_s2)] 
        # print_rank(pair_s2, "pair_s2")
        # print_rank(tokens_per_expert_s1_exp, "tokens_per_experts_s1_exp")

        # print_rank(f"x_s2.shape:{dx_s2.shape}, sum of ins: {sum(ins_s2)}, ins_s2:{ins_s2}")

        assert dx_s2.shape[0] == sum(ins_s2), f"dx_s2.shape:{dx_s2.shape}, sum of ins: {sum(ins_s2)}, ins_s2:{ins_s2}"

        st_a2a_l = self._start_timer(time_breakdown)
        dx_s2_exp = _AllToAllSingle.apply(self.local_group, dx_s2, ins_s2, outs_s2)
        t_a2a_l = self._stop_timer(st_a2a_l)

        st_local_order = self._start_timer(time_breakdown)
        dx_exp_ = torch.cat([dx_s2_exp, dx_s1_exp])
        t_local_order = self._stop_timer(st_local_order)

        if time_breakdown:
            return dx_exp_, {
                "gather": t_gather,
                "a2a_global": t_a2a_g,
                "s1_to_s2": t_s1tos2,
                "a2a_local": t_a2a_l,
                "local_order": t_local_order
            }

        return dx_exp_



class RBCombiner(torch.nn.Module):
    def __init__(self, num_experts: int, top_k: int, num_local_experts: int, mesh_size: int, ep_group: dist.ProcessGroup, local_group: dist.ProcessGroup):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.mesh_size = mesh_size

        self.ep_group = ep_group
        self.local_group = local_group

    def _start_timer(self, enabled):
        if enabled:
            torch.cuda.synchronize()
            dist.barrier()
            return time.time()
        return None

    def _stop_timer(self, start_time):
        if start_time is None:
            return None
        torch.cuda.synchronize()
        dist.barrier()
        return time.time() - start_time

    def forward(self, dout_exp: torch.Tensor, weights: torch.Tensor, n_tokens: int, metadata, time_breakdown=False):

        indices_s1, indices_s2, bin_ids_s1, bins_s1, ins_s1, outs_s1, ins_s2_virtual, outs_s2_virtual, \
                s1exp_to_s2_indices, bin_ids_s2, bins_s2, ins_s2, outs_s2, tokens_per_expert_s1_exp, tokens_per_experts_s2_exp = metadata

        # dout_s2_exp = LocalOrderRecoverOp.apply(dout_exp, indices_s2, bin_ids_s2, bins_s2, n_tokens, self.top_k) 

        timing = {}

        st_weight_calc = self._start_timer(time_breakdown)
        weights = weights.to(dout_exp.dtype)
        weights_s1 = weights[indices_s1]
        weights_s2_virtual = weights[indices_s2]
        weights_s1_exp = _AllToAllSingle.apply(self.ep_group, weights_s1, ins_s1, outs_s1)
        weights_s2 = _AllToAllSingle.apply(self.ep_group, weights_s2_virtual, ins_s2_virtual, outs_s2_virtual)
        timing['weight_calc'] = self._stop_timer(st_weight_calc)

        st_reorder = self._start_timer(time_breakdown)
        dout_s1_exp, dout_s2_exp = torch.split(dout_exp, [sum(outs_s1), sum(outs_s2)])
        timing['reorder'] = self._stop_timer(st_reorder)

        st_a2a_local = self._start_timer(time_breakdown)
        dout_s2 = _AllToAllSingle.apply(self.local_group, dout_s2_exp, outs_s2, ins_s2)
        timing['a2a_local'] = self._stop_timer(st_a2a_local)

        # print_rank(dout_s1_exp.is_contiguous(), "is dout_s1 contiguous?")
        # print_rank(dout_s2.is_contiguous(), "is dout_s2 contiguous?")

        st_scale = self._start_timer(time_breakdown)
        weighted_dout_s1_exp = dout_s1_exp * weights_s1_exp[:, None]
        weighted_dout_s2 = dout_s2 * weights_s2[:, None]
        # weighted_dout_s1_exp = dout_s1_exp 
        # weighted_dout_s2 = dout_s2 
        timing['scale'] = self._stop_timer(st_scale)

        # assert weighted_dout_s1_exp.shape == dout_s1_exp.shape
        # assert weighted_dout_s2.shape == dout_s2.shape

        # print_rank(weighted_dout_s1_exp.is_contiguous(), "is weighted s1 contiguous?")
        # print_rank(weighted_dout_s2.is_contiguous(), "is weighted s2 contiguous?")

        st_scatter_s2_to_s1 = self._start_timer(time_breakdown)
        dout_s1_exp = s2_to_s1_scatter(weighted_dout_s1_exp, weighted_dout_s2, s1exp_to_s2_indices, bin_ids_s2, bins_s2)
        timing['scatter_s2_to_s1'] = self._stop_timer(st_scatter_s2_to_s1)

        # print_rank(dout_s1_exp.shape)
        # assert dout_s1_exp.shape[0] == sum(outs_s1)

        st_a2a_global = self._start_timer(time_breakdown)
        dout_s1 = _AllToAllSingle.apply(self.ep_group, dout_s1_exp, outs_s1, ins_s1)
        timing['a2a_global'] = self._stop_timer(st_a2a_global)

        st_final_scatter = self._start_timer(time_breakdown)
        out = scatter_with_token_drop(dout_s1, indices_s1, bin_ids_s1, None, bins_s1, n_tokens, self.top_k)
        timing['final_scatter'] = self._stop_timer(st_final_scatter)

        if time_breakdown:
            return out, timing

        return out

# model_dim = 4
# x = torch.arange(4, device='cuda', dtype=torch.float16).unsqueeze(1).repeat(1, model_dim)

# indices = torch.tensor([6, 2, 7, 4, 3, 0, 5, 1], dtype=torch.int, device='cuda')
# bin_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int,device='cuda')
# bins = torch.tensor([2, 4, 6, 8], dtype=torch.int, device='cuda')
# mesh_size = 2

# dx = drop_gather(x, indices, bin_ids, None, bins, 4, 2)

# # ep_group = create_group(do_test=True)

# print(dx)

# print(indices)
# print(bin_ids)
# print(bins)

# indices_s1, bin_ids_s1, bins_s1, map_indices, tokens_per_experts = tokens_filter(indices, bin_ids, 4, 4, 2, mesh_size, 1)

# print(indices_s1)
# print(bin_ids_s1)
# print(bins_s1)
# print(map_indices)

# dx_s1 = drop_gather(x, indices_s1, bin_ids_s1, None, bins, 4, 2)

# dx_s2 = _map(dx_s1, map_indices, None, 4, 2)

# print(dx_s1)
# print(dx_s2)