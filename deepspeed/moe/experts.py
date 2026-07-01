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
        
from .grouped_gemm.fused_bias_gelu import bias_gelu_impl
try:
    from .grouped_gemm.interface import grouped_gemm
except ImportError:
    # Provide a clear error message if the import fails during instantiation
    raise ImportError(
        "[deepspeed/moe/experts.py] FusedExperts_Triton requires the grouped_gemm kernel. "
        "Please install it or use a different expert class."
    )
import os 
print (f'os.getenv ("USE_AITER_GROUPGEMM"): {os.getenv ("USE_AITER_GROUPGEMM")}')
if os.getenv ("USE_AITER_GROUPGEMM") == "True": 
    try:
        import primus_turbo.pytorch as turbo
    except ImportError:
        # Provide a clear error message if the import fails during instantiation
        raise ImportError(
            "[deepspeed/moe/experts.py] FusedExperts_Primus requires import primus_turbo.pytorch as turbo "
            )

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


# ============================================================================
# IMPLEMENTING SHARED EXPERT
#   Added: SharedExpert class.   Changed: none.   Removed: none.
#
# A DeepSeek-style "shared expert" is a SINGLE dense (SwiGLU) MLP that processes
# EVERY token locally (NO routing, NO chunking, NO all-to-all) and is summed with
# the routed experts' output:   y = routed_experts(x) + shared_expert(x).
#
# Unlike routed experts (Experts / FusedExperts_*), its parameters are REPLICATED
# on every EP/DP rank exactly like the attention layer. Therefore we MUST NOT tag
# them with `param.allreduce = False` / `param.group_name`. Leaving them untagged
# makes is_moe_param() return False (see deepspeed/moe/utils.py), so their gradients
# flow through the standard FULL data-parallel all-reduce
# (deepspeed/runtime/engine.py::_reduce_non_expert_gradients) and ZeRO partitions
# their optimizer state over the full DP group -- identical to attention weights.
# ============================================================================
class SharedExpert(nn.Module):

    def __init__(self, shared_mlp: nn.Module) -> None:
        super(SharedExpert, self).__init__()
        # `shared_mlp` is a pre-built dense MLP (Megatron ParallelMLP) whose FFN width
        # is moe_intermediate_size * num_shared_experts. It is constructed on the caller
        # side (megatron/model/transformer.py) where ParallelMLP + config are available.
        # NOTE: we intentionally do NOT touch param.allreduce / param.group_name here.
        self.shared_mlp = shared_mlp

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.shared_mlp(hidden_states)
        # Megatron ParallelMLP returns (output, output_bias) because of skip_bias_add.
        if isinstance(out, tuple):
            out, out_bias = out
            if out_bias is not None:
                out = out + out_bias
        return out

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

from .elmoe_triton_groupgemm import Kernel4v2_GEMM, build_index_tensors

class FusedExperts_Triton(nn.Module):
    def __init__(self, expert: nn.Module, config, num_local_experts: int = 1, expert_group_name: Optional[str] = None, is_uneven_tokens = False) -> None:
        super(FusedExperts_Triton, self).__init__()

        self.num_local_experts = num_local_experts
        self.hidden_size = config.hidden_size
        self.mlp_intermediate_size = config.ffn_hidden_size 
        self.init_method = config.init_method
        self.add_bias = config.add_bias_linear
        
        # 1. Safely grab the Up and Down projections
        linear_weights =[p for p in expert.parameters() if p.ndim == 2]
        base_up_weight = linear_weights[0]  
        base_down_weight = linear_weights[-1] 
        
        # 2. EXACT DEEPSPEED MATCH: Copy the exact same weights to all experts!
        # (.expand().contiguous() does the exact same thing as copy.deepcopy)
        stacked_up = base_up_weight.unsqueeze(0).expand(num_local_experts, -1, -1).contiguous()
        stacked_down = base_down_weight.unsqueeze(0).expand(num_local_experts, -1, -1).contiguous()

        self.up_proj_weight = nn.Parameter(stacked_up)
        self.down_proj_weight = nn.Parameter(stacked_down)

        # 3. Set DeepSpeed / Megatron parallel attributes
        self.up_proj_weight.allreduce = False
        self.up_proj_weight.group_name = expert_group_name
        self.down_proj_weight.allreduce = False
        self.down_proj_weight.group_name = expert_group_name
        
        print(f'[experts.py] Triton Fused Experts Initialized. Up Proj Shape: {self.up_proj_weight.shape}')

        # Bias handling (if your config enables it)
        if self.add_bias:
            init_device = torch.device("cpu") if getattr(config, "use_cpu_initialization", False) else torch.cuda.current_device()
            self.bias = nn.Parameter(torch.empty(
                self.mlp_intermediate_size, 
                device=init_device, 
                dtype=config.params_dtype
            ))
            nn.init.zeros_(self.bias)
            
        self.activation_func = F.gelu
            
    def forward(self, inputs: torch.Tensor, output_splits_tensor: torch.Tensor = None) -> torch.Tensor:
        # Convert splits tensor to a standard Python list for the index builder
        splits = output_splits_tensor.tolist()

        # 1. BUILD INDICES ONCE
        # We build the phonebook once and use it for both projections!
        # d_idx, d_es, d_ms = build_index_tensors(splits, self.num_local_experts, inputs.device)
        d_idx, d_es, d_ms = build_index_tensors(output_splits_tensor, self.num_local_experts)

        # 2. UP PROJECTION 
        # (Rank-Primary Input -> Rank-Primary Output via Gather-Scatter)
        hidden = Kernel4v2_GEMM.apply(
            inputs, 
            self.up_proj_weight, 
            d_idx, d_es, d_ms
        )
        
        # 3. ACTIVATION 
        if self.add_bias:
            hidden = bias_gelu_impl(hidden, self.bias) 
        else:
            hidden = self.activation_func(hidden)
            
        
        # 4. DOWN PROJECTION 
        # (Rank-Primary Input -> Rank-Primary Output via Gather-Scatter)
        output = Kernel4v2_GEMM.apply(
            hidden, 
            self.down_proj_weight, 
            d_idx, d_es, d_ms
        )
        
        return output
        