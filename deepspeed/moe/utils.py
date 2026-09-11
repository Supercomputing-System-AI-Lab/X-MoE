# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union, cast

import torch
from torch import nn

from .layer import MoE


def has_moe_layers(m: nn.Module) -> Tuple[bool, int]:
    has_moe = False
    num_experts = 0

    for module in m.modules():
        if isinstance(module, MoE):
            has_moe = True
            num_experts = module.num_experts
            break
    return has_moe, num_experts


def is_moe_param(param: torch.Tensor) -> bool:
    if hasattr(param, "allreduce") and not param.allreduce:
        return True
    return False


def split_params_into_shared_and_expert_params(
        params: List[torch.nn.Parameter]) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    shared_params: List[nn.Parameter] = []
    expert_params: List[nn.Parameter] = []

    for p in params:
        if is_moe_param(p):
            expert_params.append(p)
        else:
            shared_params.append(p)
    return shared_params, expert_params


def split_params_grads_into_shared_and_expert_params(
        group: List[torch.nn.Parameter]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Split grad of parameters into grads of non-expert params
    and grads of expert params. This is useful while computing
    grad-norms for clipping and overflow detection

        group (List[torch.nn.Parameter]):
    Args:
            The group of parameters to split

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor]]:
        list of gradients for non MoE params, list of gradients of MoE params
    """
    expert_grads: List[torch.Tensor] = []
    shared_grads: List[torch.Tensor] = []

    for p in group:
        if p.grad is not None:
            if is_moe_param(p):
                expert_grads.append(p.grad.to(p.dtype))
            else:
                shared_grads.append(p.grad.to(p.dtype))
    return shared_grads, expert_grads


def split_params_into_different_moe_groups_for_optimizer(
        param_groups: Union[Dict[str, Any], Tuple[Dict[str, Any], ...], List[Dict[str, Any]]],
        max_group_size: Union[int, float] = 178956971) -> List[Dict[str, Any]]:
    """Split parameters into different MoE groups for optimizer

    Args:
        param_groups (Union[Dict[str, Any], Tuple[Dict[str, Any], ...], List[Dict[str, Any]]])
            The list of parameter groups to split

    Returns:
        List[Dict[str, Any]]:
        list of MoE/non-MoE groups for optimizer
    """
    if isinstance(param_groups, tuple):
        param_groups = list(param_groups)  # Tuple cannot be modified
    elif isinstance(param_groups, dict):
        param_groups = [param_groups]
    elif not isinstance(param_groups, list):
        raise ValueError(f"Unknown param group type of {type(param_groups)}")

    # gather all data parallel group names
    data_parallel_group_names: Set[str] = set()
    for param_group in param_groups:
        for param in cast(List[nn.Parameter], param_group["params"]):
            if is_moe_param(param):
                data_parallel_group_names.add(param.group_name)

    # Create the param MoE groups, leave param assign to next step
    group_moe: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for param_group in param_groups:
        for key in data_parallel_group_names:
            group_moe[param_group['name']][key] = {
                **param_group,
                'name': key,
                'moe': True,
                'params': [],
            }

    # Assign param
    for param_group in param_groups:
        new_params: List[nn.Parameter] = []

        for param in cast(List[nn.Parameter], param_group['params']):
            if is_moe_param(param):
                group_moe[param_group['name']][param.group_name]['params'].append(param)
            else:
                new_params.append(param)
        param_group['params'] = new_params

    # Flatten the moe groups
    if max_group_size is not None:
        for moe_group in group_moe.values():
            for param_group in moe_group.values():
                cur_group: List[nn.Parameter] = []
                all_groups: List[List[nn.Parameter]] = []
                size_of_cur_group = 0

                for param in cast(List[nn.Parameter], param_group['params']):
                    if size_of_cur_group + param.numel() <= max_group_size:
                        cur_group.append(param)
                        size_of_cur_group += param.numel()
                    else:
                        all_groups.append(cur_group)
                        cur_group = [param]
                        size_of_cur_group = param.numel()

                if cur_group:
                    all_groups.append(cur_group)

                for group in all_groups:
                    param_groups.append({**param_group, 'params': group})
    else:
        for moe_group in group_moe.values():
            for param_group in moe_group.values():
                param_groups.append(param_group)

    return param_groups


def globalize_expert_param_names(
        module: nn.Module,
        param_names: Dict[torch.nn.Parameter, str]) -> Tuple[Dict[torch.nn.Parameter, str], List[str]]:
    """Rewrite rank-local expert parameter names so that every rank agrees on them.

    Universal checkpointing addresses a parameter's optimizer state by the parameter's name
    -- one directory per name under <checkpoint>/zero/. That is only an address if every
    rank produces the same name for the same tensor. Expert parallelism breaks it: Experts
    builds each rank's share as its own nn.ModuleList, so named_parameters() numbers them
    0..num_local_experts-1 on *every* expert-parallel rank. All of them then resolve to the
    same directory: with 8 expert-parallel ranks the checkpoint ends up holding one rank's
    8 experts instead of all 64, and nothing raises.

    DeepSpeed already solves exactly this for expert *weights*.
    DeepSpeedEngine._save_moe_checkpoint files them as layer_<L>_expert_<global id>_... with

        global id = expert_parallel_rank * num_local_experts + local id

    This applies the same numbering to the name the optimizer state is addressed by. Doing it
    here -- once, where the names are minted -- is what keeps ZeRO, ds_to_universal.py and the
    universal loader free of any notion of an expert: they go on addressing parameters by
    name, and the names are now unique.

    Returns `(renamed, unsupported)`. `renamed` is a new {parameter: name} mapping;
    `param_names` is not modified. `unsupported` is the sorted list of expert parameter names
    that could not be made globally unique, left in `renamed` under their rank-local name.

    An expert implementation that stacks its local experts into one tensor has no per-expert
    name to renumber, so it lands in `unsupported` rather than raising: expert parallelism is
    a training feature and universal checkpointing is not, and a model using such an
    implementation must keep training normally whether or not it will ever be converted. The
    caller is responsible for refusing the conversion -- a checkpoint written from colliding
    names loads without error and gives most experts another expert's weights, so it must not
    be allowed to convert silently.

    A parameter that `is_moe_param` accepts but that belongs to neither case is still an
    error: it means the module layout changed underneath this function.
    """
    from deepspeed.utils import groups

    renamed = dict(param_names)
    renumbered = set()
    unsupported = set()

    for module_name, submodule in module.named_modules():
        num_local_experts = getattr(submodule, 'num_local_experts', None)
        if num_local_experts is None:
            continue

        local_experts = getattr(submodule, 'deepspeed_experts', None)
        if local_experts is None:
            # num_local_experts alone does not make a module an expert container: MoE keeps
            # the count next to the experts it builds, so it matches here while owning no
            # expert tensor of its own and having nothing to rename. What distinguishes a
            # fused implementation is that it holds expert parameters DIRECTLY -- one stacked
            # [num_local_experts, ...] Parameter instead of a submodule per expert.
            if not any(is_moe_param(p) for _, p in submodule.named_parameters(recurse=False)):
                continue
            # A fused expert's state is a slice of a shared tensor rather than something with
            # a name of its own: there is nothing to renumber, and the tensor's contents
            # depend on the expert-parallel degree, so it could not be resharded either.
            # Report it and carry on -- see the note on `unsupported` in the docstring.
            unsupported.update(name for _, name in
                               ((p, param_names[p]) for _, p in submodule.named_parameters()
                                if p in param_names))
            continue

        # Every expert parameter is tagged with its expert-parallel group in
        # Experts.__init__; that group is what the local numbering is relative to.
        group_name = next((getattr(p, 'group_name') for p in submodule.parameters() if hasattr(p, 'group_name')),
                          None)
        if group_name is None:
            raise RuntimeError(f"expert module '{module_name}' has no parameter carrying a "
                               f"group_name, so its expert-parallel rank cannot be determined")
        expp_rank = groups._get_expert_parallel_rank(group_name)

        for local_id, expert in enumerate(local_experts):
            global_id = expp_rank * num_local_experts + local_id
            for suffix, param in expert.named_parameters():
                local_name = f'{module_name}.deepspeed_experts.{local_id}.{suffix}'
                # If this ever fails the module layout has changed underneath us, and a
                # silently wrong rename is worse than a stopped job.
                assert param_names.get(param) == local_name, (
                    f'expected this parameter to be named {local_name!r}, found '
                    f'{param_names.get(param)!r}')
                renamed[param] = f'{module_name}.deepspeed_experts.{global_id}.{suffix}'
                renumbered.add(param)

    # Backstop against a layout nobody anticipated: is_moe_param is DeepSpeed's own test for
    # "this parameter belongs to one expert-parallel rank" (it is the allreduce=False tag set
    # in Experts.__init__), so anything it matches that was not renumbered above would be
    # written under a colliding name.
    missed = sorted(name for param, name in param_names.items()
                    if is_moe_param(param) and param not in renumbered and name not in unsupported)
    if missed:
        raise RuntimeError(
            f'{len(missed)} expert-parallel parameters were not given a global name, so their '
            f'checkpoint state would collide across expert-parallel ranks: {missed[:3]}'
            f'{" ..." if len(missed) > 3 else ""}')

    return renamed, sorted(unsupported)


def is_moe_param_group(param_group):
    return param_group.get('moe', False)


def configure_moe_param_groups(model_parameters: List):
    assert isinstance(model_parameters, list), "model_parameters must be a list"

    for p in model_parameters:
        # match torch.optim.Optimizer expectations,
        # see: https://github.com/pytorch/pytorch/blob/2ffab6e663b9c6951048b8c8ba82d2cc5ca5c2fc/torch/optim/optimizer.py#L270-L272
        if not isinstance(p, (torch.Tensor, dict)):
            raise TypeError("param argument that would be given to the optimizer should be "
                            f"an iterable of Tensors or dicts, but got {type(p)}")

    # peak at the first element to determine how to proceed
    first = model_parameters[0]

    # Case 1: model_parameters is a list of torch.nn.Parameter
    #   -> need to create moe compatible param groups
    if isinstance(first, torch.nn.Parameter):
        param_group = {'params': model_parameters, 'name': 'dense-params'}
        return split_params_into_different_moe_groups_for_optimizer(param_group)

    # Case 2: model_parameters is a list of param groups List[dict]
    #   -> moe compatible param groups might already exist, if not create them
    elif isinstance(first, dict):
        #there are no moe groups created
        if not any(['moe' in param_group for param_group in model_parameters]):
            return split_params_into_different_moe_groups_for_optimizer(model_parameters)
        else:
            # moe groups exist, nothing to do
            return model_parameters
