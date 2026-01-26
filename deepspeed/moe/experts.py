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
import primus_turbo.pytorch as turbo

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

class FusedExperts_Primus(nn.Module):
    def __init__(self, expert: nn.Module, config, num_local_experts: int = 1, expert_group_name: Optional[str] = None, is_uneven_tokens = False) -> None:
        super(FusedExperts_Primus, self).__init__()

        self.num_local_experts = num_local_experts
        self.hidden_size = config.hidden_size
        self.add_bias = config.add_bias_linear
        self.mlp_intermediate_size = config.ffn_hidden_size 
        self.init_method = config.init_method
        
        # construct fused up_proj and down_proj
        temp_expert_list = nn.ModuleList([copy.deepcopy(expert) for _ in range(num_local_experts)])
        up_proj_weights = []
        down_proj_weights = []

        for expert in temp_expert_list:
            up_proj_weights.append(list(expert.parameters())[0])
            down_proj_weights.append(list(expert.parameters())[1])
        
        stacked_up_proj_weights = torch.stack(up_proj_weights, dim=0).transpose(1, 2).contiguous()
        stacked_down_proj_weights = torch.stack(down_proj_weights, dim=0).transpose(1, 2).contiguous()
        del up_proj_weights
        del down_proj_weights
        del temp_expert_list
        self.up_proj_weight = nn.Parameter(stacked_up_proj_weights)
        self.down_proj_weight = nn.Parameter(stacked_down_proj_weights)

        self.up_proj_weight.allreduce = False
        self.up_proj_weight.group_name = expert_group_name
        self.down_proj_weight.allreduce = False
        self.down_proj_weight.group_name = expert_group_name

        self.add_bias = False # Explicitly disabled as per original code

        if self.add_bias:
            if config.use_cpu_initialization:
                self.bias = nn.Parameter(torch.empty(
                    self.mlp_intermediate_size, dtype=config.params_dtype))
            else:
                self.bias = nn.Parameter(torch.empty(
                    self.mlp_intermediate_size,
                    device=get_accelerator().current_device_name(),
                    dtype=config.params_dtype))

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
        grouped_input = torch.cat(expert_input_chunks, dim=0)
        m_size = torch.tensor(m_size, device=grouped_input.device, dtype=torch.int32)
        m_size = m_size.long()
        hidden = turbo.ops.grouped_gemm(grouped_input, self.up_proj_weight, m_size, trans_b=False)
        if self.add_bias:
            hidden = bias_gelu_impl(hidden, self.bias)
        else:
            hidden = self.activation_func(hidden)
        output = turbo.ops.grouped_gemm(hidden, self.down_proj_weight, m_size, trans_b=False)
        return output

class FusedExperts_Triton(nn.Module):
    def __init__(self, expert: nn.Module, config, num_local_experts: int = 1, expert_group_name: Optional[str] = None, is_uneven_tokens = False) -> None:
        super(FusedExperts_Triton, self).__init__()

        self.num_local_experts = num_local_experts
        self.hidden_size = config.hidden_size
        self.add_bias = config.add_bias_linear
        self.mlp_intermediate_size = config.ffn_hidden_size 
        self.init_method = config.init_method
        
        # construct fused up_proj and down_proj
        temp_expert_list = nn.ModuleList([copy.deepcopy(expert) for _ in range(num_local_experts)])
        up_proj_weights = []
        down_proj_weights = []

        for expert in temp_expert_list:
            up_proj_weights.append(list(expert.parameters())[0])
            down_proj_weights.append(list(expert.parameters())[1])
        
        stacked_up_proj_weights = torch.stack(up_proj_weights, dim=0).contiguous()
        stacked_down_proj_weights = torch.stack(down_proj_weights, dim=0).contiguous()

        self.up_proj_weight = nn.Parameter(stacked_up_proj_weights)
        self.down_proj_weight = nn.Parameter(stacked_down_proj_weights)

        self.up_proj_weight.allreduce = False
        self.up_proj_weight.group_name = expert_group_name
        self.down_proj_weight.allreduce = False
        self.down_proj_weight.group_name = expert_group_name

        self.add_bias = False # Explicitly disabled as per original code

        if self.add_bias:
            if config.use_cpu_initialization:
                self.bias = nn.Parameter(torch.empty(
                    self.mlp_intermediate_size, dtype=config.params_dtype))
            else:
                self.bias = nn.Parameter(torch.empty(
                    self.mlp_intermediate_size,
                    device=get_accelerator().current_device_name(),
                    dtype=config.params_dtype))

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
        grouped_input = torch.cat(expert_input_chunks, dim=0)
        m_size = torch.tensor(m_size, device=grouped_input.device, dtype=torch.int32)
        hidden = grouped_gemm(grouped_input, self.up_proj_weight, m_size)
        if self.add_bias:
            hidden = bias_gelu_impl(hidden, self.bias)
        else:
            hidden = self.activation_func(hidden)
        output = grouped_gemm(hidden, self.down_proj_weight, m_size)
        return output


    # def forward(self, inputs: torch.Tensor, output_splits_tensor: torch.Tensor = None) -> torch.Tensor:
    #     """
    #     Optimized forward pass using GPU-based index reordering.
    #     Eliminates CPU-GPU sync (no .tolist()) and reduces memory copies.
    #     """
    #     num_chunks = output_splits_tensor.shape[0]
    #     chunk_ids = torch.arange(num_chunks, device=inputs.device)
    #     chunk_ids_per_token = torch.repeat_interleave(chunk_ids, output_splits_tensor)
    #     token_expert_ids = chunk_ids_per_token % self.num_local_experts
    #     sort_indices = torch.argsort(token_expert_ids, stable=True)
    #     grouped_input = inputs[sort_indices]
    #     m_size = torch.bincount(token_expert_ids, minlength=self.num_local_experts).to(torch.int32)

    #     # Up Projection
    #     hidden1 = turbo.ops.grouped_gemm(grouped_input, self.up_proj_weight, m_size, trans_b=True)
    #     hidden2 = grouped_gemm(grouped_input, self.up_proj_weight, m_size)

    #     # Activation
    #     if self.add_bias:
    #         hidden1 = bias_gelu_impl(hidden1, self.bias)
    #         hidden2 = bias_gelu_impl(hidden2, self.bias)
    #     else:
    #         hidden1 = self.activation_func(hidden1)
    #         hidden2 = self.activation_func(hidden2)
            
    #     # Down Projection
    #     output1 = turbo.ops.grouped_gemm(hidden1, self.down_proj_weight, m_size, trans_b=True)
    #     output2 = grouped_gemm(hidden2, self.down_proj_weight, m_size)

    #     # assert torch.allclose(output1, output2, atol=1e-5), "Outputs from turbo and grouped_gemm do not match!"
    #     # print("Outputs from turbo and grouped_gemm match!")
    #     return output1