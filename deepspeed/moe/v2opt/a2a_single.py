
import sys
import torch
from torch import Tensor
from deepspeed import comm as dist
from typing import Any, Tuple
from deepspeed.utils import groups
from deepspeed.utils.bwc import bwc_tensor_model_parallel_world_size

class _AllToAllSingle(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor, input_splits: list, output_splits: list, debug_timing: bool = False) -> Tensor:  # type: ignore

        ctx.group = group
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits

        if debug_timing:
            import time
            torch.cuda.synchronize()
            dist.barrier()
            st = time.time()

        input = input.contiguous()
        total_recv = sum(output_splits)
        if len(input.shape) == 1:
            output = torch.empty([total_recv], device=input.device, dtype=input.dtype)
        else:
            output = torch.empty(total_recv, input.shape[-1], device=input.device, dtype=input.dtype)

        if debug_timing:
            torch.cuda.synchronize()
            dist.barrier()
            layout_time = time.time() - st

        world_size = dist.get_world_size()
        rank = (dist.get_rank() // groups._get_expert_model_parallel_world_size()) % dist.get_world_size(group)
        # rank = dist.get_rank() % dist.get_world_size(group)

        input_splits_0_flag = False
        output_splits_0_flag = False


        # dist.barrier()
        # print(f"{rank}: before process", flush=True)
        # print(f"{rank}: {input.shape}", flush=True)
        # print(f"{rank}: {output.shape}", flush=True)
        # dist.barrier()

        if sum(input_splits) == 0:
            # print("!!! detect all 0 input", flush=True)
            if len(input.shape) == 1: 
                input = torch.zeros([1], device='cuda', dtype=input.dtype)
            else:
                input = torch.zeros((1, input.shape[-1]), device='cuda', dtype=input.dtype)
            temp_input_splits = input_splits.copy()
            temp_output_splits = output_splits.copy()
            temp_input_splits[rank] = 1
            temp_output_splits[rank] = 1
            
            if len(input.shape) == 1: 
                extra_zeros = torch.zeros([1], device='cuda', dtype=output.dtype)
            else:
                extra_zeros = torch.zeros((1, output.shape[-1]), device='cuda', dtype=output.dtype)
            output = torch.cat([output, extra_zeros], dim=0)

            _position = sum(output_splits[:rank])
            input_splits_0_flag = True
        
        if sum(output_splits) == 0 and not input_splits_0_flag:
            # print("!!! detect all 0 output", flush=True)
            _output = output
            temp_input_splits = input_splits.copy()
            temp_output_splits = output_splits.copy()
            temp_input_splits[rank] = 1
            temp_output_splits[rank] = 1

            if len(input.shape) == 1: 
                extra_zeros = torch.zeros([1], device='cuda', dtype=output.dtype)
            else:
                extra_zeros = torch.zeros((1, output.shape[-1]), device='cuda', dtype=output.dtype)
            _position = sum(input_splits[:rank])
            input = torch.cat((input[:_position], extra_zeros, input[_position:]), dim=0)
            if len(input.shape) == 1: 
                output = torch.zeros([1], device='cuda', dtype=input.dtype)
            else:
                output = torch.zeros((1, input.shape[-1]), device='cuda', dtype=input.dtype)
            output_splits_0_flag = True

        # dist.barrier()
        # print(f"{rank}: after process", flush=True)
        # print(f"{rank}: input {input.shape}", flush=True)
        # print(f"{rank}: output {output.shape}", flush=True)
        # dist.barrier()

        if debug_timing:
            torch.cuda.synchronize()
            dist.barrier()
            st = time.time()


        if input_splits_0_flag or output_splits_0_flag:
            # print(f"rank {dist.get_rank()} as local rank {rank}: {input.shape}", flush=True)
            # print(f"rank {dist.get_rank()} as local rank {rank}: {temp_input_splits}", flush=True)
            dist.all_to_all_single(output, input, output_split_sizes=temp_output_splits, input_split_sizes=temp_input_splits, group=group)
        else:
            dist.all_to_all_single(output, input, output_split_sizes=output_splits, input_split_sizes=input_splits, group=group)

        if debug_timing:
            torch.cuda.synchronize()
            dist.barrier()
            comm_time = time.time() - st
            if rank == 0:
                print(f"----layout time: {layout_time}", flush=True)
                print(f"----comm time: {comm_time}", flush=True)

        if input_splits_0_flag:
            output = torch.cat((output[:_position], output[_position+1:]), dim=0)

        
        if output_splits_0_flag:
            output = _output

        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAllSingle.apply(ctx.group, *grad_output, ctx.output_splits, ctx.input_splits), None, None, None)
