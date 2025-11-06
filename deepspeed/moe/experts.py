# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy
from typing import Callable, List, Optional

import torch
from torch import nn
import torch.nn.functional as F
from deepspeed.accelerator import get_accelerator
from .grouped_gemm.fused_bias_gelu import bias_gelu_impl
from .grouped_gemm.interface import grouped_gemm

class Experts(nn.Module):

    def __init__(self, expert: nn.Module, num_local_experts: int = 1, expert_group_name: Optional[str] = None, is_uneven_tokens = False) -> None:
        super(Experts, self).__init__()

        self.deepspeed_experts = nn.ModuleList([copy.deepcopy(expert) for _ in range(num_local_experts)])
        self.num_local_experts = num_local_experts

        # TODO: revisit allreduce for moe.gate...
        for expert in self.deepspeed_experts:
            # TODO: Create param groups to handle expert + data case (e.g. param.group = moe_group)
            for param in expert.parameters():
                param.allreduce = False
                param.group_name = expert_group_name

        self.is_uneven_tokens = is_uneven_tokens

    # def forward(self, inputs: torch.Tensor, splits: Optional[List[int]] = None) -> torch.Tensor:
    #     expert_outputs: List[torch.Tensor] = []

    #     if not self.is_uneven_tokens:
    #         chunks = inputs.chunk(self.num_local_experts, dim=1)
    #         for chunk, expert in zip(chunks, self.deepspeed_experts):
    #             out = expert(chunk)
    #             if isinstance(out, tuple):
    #                 out = out[0]  # Ignore the bias term for now
    #             expert_outputs.append(out)
    #         return torch.cat(expert_outputs, dim=1)
    #     else:
    #         assert splits is not None, "'splits' must be provided when is_uneven_tokens is True"
    #         assert len(splits) == self.num_local_experts, "splits length must match number of experts"
    #         split_chunks = torch.split(inputs, splits, dim=0)
    #         for chunk, expert in zip(split_chunks, self.deepspeed_experts):
    #             out = expert(chunk)
    #             if isinstance(out, tuple):
    #                 out = out[0]  # Ignore the bias term for now
    #             expert_outputs.append(out)
    #         return torch.cat(expert_outputs, dim=0)

    def forward(self, inputs: torch.Tensor, output_splits_tensor: torch.Tensor = None) -> torch.Tensor:
        expert_outputs: List[torch.Tensor] = []

        if not self.is_uneven_tokens:
            chunks = inputs.chunk(self.num_local_experts, dim=1)
            for chunk, expert in zip(chunks, self.deepspeed_experts):
                out = expert(chunk)
                if isinstance(out, tuple):
                    out = out[0]  # Ignore the bias term for now
                expert_outputs.append(out)
            return torch.cat(expert_outputs, dim=1)
        else:
            split_chunks = torch.split(inputs, output_splits_tensor.tolist(), dim=0)
            
            i = 0
            for expert in self.deepspeed_experts:
                chunk = split_chunks[i::self.num_local_experts]
                chunk = torch.cat(chunk, dim=0)
                out = expert(chunk)
                if isinstance(out, tuple):
                    out = out[0]  # Ignore the bias term for now
                expert_outputs.append(out)
                i += 1
            return torch.cat(expert_outputs, dim=0)

class MoeFusedLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "num_experts"]
    in_features: int
    out_features: int
    num_experts: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        init_method: Callable,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.empty((num_experts, out_features, in_features), **factory_kwargs))
        init_method(self.weight)

    def forward(self, input: torch.Tensor, m_sizes: torch.Tensor) -> torch.Tensor:
        # print("weight shape:", self.weight.shape)
        return grouped_gemm(input, self.weight, m_sizes)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, num_experts={self.num_experts}"


class FusedExperts(nn.Module):
    def __init__(self, config, num_local_experts: int = 1, expert_group_name: Optional[str] = None, is_uneven_tokens = False) -> None:
        super(FusedExperts, self).__init__()

        self.num_local_experts = num_local_experts
        self.hidden_size = config.hidden_size
        self.add_bias = config.add_bias_linear
        self.mlp_intermediate_size = config.ffn_hidden_size 
        self.init_method = config.init_method
        self.up_proj = MoeFusedLinear(self.hidden_size, self.mlp_intermediate_size, num_local_experts, self.init_method,device=get_accelerator().current_device_name(), dtype=config.params_dtype)
        self.down_proj = MoeFusedLinear(self.mlp_intermediate_size, self.hidden_size, num_local_experts, self.init_method,device=get_accelerator().current_device_name(), dtype=config.params_dtype)
        
        self.add_bias = False

        if self.add_bias:
            if config.use_cpu_initialization:
                self.bias = nn.Parameter(torch.empty(
                    self.mlp_intermediate_size, dtype=config.params_dtype))
            else:
                self.bias = nn.Parameter(torch.empty(
                    self.mlp_intermediate_size,
                    device=get_accelerator().current_device_name(),
                    dtype=config.params_dtype))
        for param in self.up_proj.parameters():
            param.allreduce = False
            param.group_name = expert_group_name
        for param in self.down_proj.parameters():
            param.allreduce = False
            param.group_name = expert_group_name
        
        # def swiglu(x):
        #     x = torch.chunk(x, 2, dim=-1)
        #     return F.silu(x[0]) * x[1]
        # self.activation_func = swiglu
        self.activation_func = F.gelu


    def forward(self, inputs: torch.Tensor, output_splits_tensor: torch.Tensor = None) -> torch.Tensor:
        grouped_input: torch.Tensor
        m_size: List[int]

        # split chunks according to output_splits_tensor
        split_chunks = torch.split(inputs, output_splits_tensor.tolist(), dim=0)
        expert_input_chunks: List[torch.Tensor] = []
        m_size: List[int] = []
        # For each expert, gather its input chunks
        i = 0
        for _ in range(self.num_local_experts):
            chunk_tuple = split_chunks[i::self.num_local_experts]
            expert_input = torch.cat(chunk_tuple, dim=0)
            expert_input_chunks.append(expert_input)
            m_size.append(expert_input.shape[0]) 
            i += 1
        # get the grouped input by concatenating all expert inputs
        grouped_input = torch.cat(expert_input_chunks, dim=0)
        m_size = torch.tensor(m_size, device=grouped_input.device, dtype=torch.int32)
        # forward through fused MLP
        hidden = self.up_proj(grouped_input, m_size)
        if self.add_bias:
            hidden = bias_gelu_impl(hidden, self.bias)
        else:
            hidden = self.activation_func(hidden)
        output = self.down_proj(hidden, m_size)
        
        return output
        

