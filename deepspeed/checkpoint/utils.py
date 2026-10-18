# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import inspect
import os
import torch
from .constants import (MODEL_FILE_PREFIX, MODEL_FILE_SUFFIX, OPTIM_FILE_SUFFIX, ZERO_FILE_PREFIX)


# Probed once, at import. torch.load's signature does not change at runtime, and this is on
# the path of every checkpoint file DeepSpeed reads.
_TORCH_LOAD_ACCEPTS_WEIGHTS_ONLY = 'weights_only' in inspect.signature(torch.load).parameters


def load_checkpoint_file(path, map_location=torch.device('cpu')):
    """torch.load() for DeepSpeed checkpoint files.

    torch >= 2.6 defaults ``weights_only=True``. That unpickler restores tensors and plain
    containers only, and a DeepSpeed checkpoint legitimately contains other objects --
    ``argparse.Namespace`` in mp_rank files, ``LossScaler`` / ``ZeroStageEnum`` /
    ``DeepSpeedConfig`` in ZeRO shards. These files are written by DeepSpeed itself, so full
    unpickling is what is intended, and it is what the runtime's own checkpoint engine
    already does (runtime/checkpoint_engine/torch_checkpoint_engine.py).

    ``weights_only`` is forwarded only when the running torch accepts it, so the same call
    works unchanged on versions that predate the argument.
    """
    kwargs = {'map_location': map_location}
    if _TORCH_LOAD_ACCEPTS_WEIGHTS_ONLY:
        kwargs['weights_only'] = False
    return torch.load(path, **kwargs)


def get_model_ckpt_name_for_rank(base_folder, mp_rank_str):
    ckpt_name = os.path.join(
        base_folder,
        MODEL_FILE_PREFIX + mp_rank_str + MODEL_FILE_SUFFIX,
    )
    return ckpt_name


def get_zero_ckpt_name_for_rank(base_folder, dp_rank, mp_rank):
    zero_prefix = f'{ZERO_FILE_PREFIX}{dp_rank}'
    mp_rank_string = f'_{MODEL_FILE_PREFIX}{mp_rank:02d}'
    zero_ckpt_name = os.path.join(
        base_folder,
        zero_prefix + mp_rank_string + OPTIM_FILE_SUFFIX,
    )
    return zero_ckpt_name


def get_layer_ckpt_name_for_rank(base_folder, layer_id, tp_rank):
    ckpt_file = f'{layer_id}-model_{tp_rank:02d}{MODEL_FILE_SUFFIX}'
    ckpt_path = os.path.join(base_folder, ckpt_file)
    return ckpt_path


# We pass cloned tensors to torch.save() to avoid checkpoint bloat that occurs when torch.save()
# saves the underlying storage rather than the slice of the storage corresponding to individual tensors.
# This is a problem in DeepSpeed because we often allocate tensors using slices of large flattened buffers.
# Tensor cloning helps to avoid this problem because the storage of cloned tensors are closer to the true size.
# It is expected that the garbage collector will reclaim the cloned tensor storage to avoid memory bloat.
# See https://pytorch.org/docs/stable/notes/serialization.html#preserve-storage-sharing
def clone_tensors_for_torch_save(item, device=torch.device('cpu')):
    """
    Returns a copy of ``item`` with all enclosed tensors replaced by clones on a specified device.
    Works on individual tensors, and tensors contained/nested in lists, tuples, and dicts.

    Parameters:
        - ``item``: tensor to clone or (possibly nested) container of tensors to clone.
        - ``device``: target device (defaults to 'cpu')

    Returns:
        - copy of ``item`` with cloned tensors on target device
    """
    if torch.is_tensor(item):
        return item.detach().clone().to(device)
    elif isinstance(item, list):
        return [clone_tensors_for_torch_save(v, device) for v in item]
    elif isinstance(item, tuple):
        return tuple([clone_tensors_for_torch_save(v, device) for v in item])
    elif isinstance(item, dict):
        return type(item)({k: clone_tensors_for_torch_save(v, device) for k, v in item.items()})
    else:
        return item
