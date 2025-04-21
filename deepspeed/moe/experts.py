# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy
from typing import List, Optional

import torch
from torch import nn


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
