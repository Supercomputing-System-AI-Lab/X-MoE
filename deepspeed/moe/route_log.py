"""route_log.py (2026-10-06): the routing record.

What it records: for every MoE layer on this rank, for every micro-batch of every training step, the number of tokens the gate sent to
each expert (the gate's own `tokens_per_expert`, a histogram of the top-k expert ids of the rank's local tokens, taken BEFORE any
capacity drop, so it is the demand). One int32 per expert; 1 KB per layer per micro-batch at 256 experts.

How, and why it costs nothing on the step path: the gate appends the count tensor (already on the GPU) to a list; after each training
step Megatron's train loop calls flush(step), which stacks the lists, copies them to the host ONCE, and appends one block to a binary
file per rank that is opened once at the first flush and kept open (one open per job, no prints, the rank logs untouched).

Switch: XMOE_ROUTE_LOG=1. Directory: XMOE_ROUTE_LOG_DIR, else <dirname of $profile_dir>/routing (= <job dir>/routing under the
Frontier template), else ./routing.

File format, rank<r>.bin: a sequence of blocks, each = int32 header [MAGIC, step, n_gates, n_micro, E] followed by int32 data of
shape [n_gates, n_micro, E] in row-major order (gate = the MoE layers of this rank's pipeline stage in construction order).
Sidecar rank<r>.json: rank, world, n_gates, experts, k, capacity_factor, drop_tokens, pp_rank, ep_size. Reader + figures:
Megatron-DeepSpeed-X-MoE/examples_elmoe/utils/routing_view.py.

Under TP (with Megatron sequence parallelism) and Ulysses every rank sees a different sequence shard, so the local counts differ
between ranks; the load of an expert in a step is the SUM over the expert-parallel group (the reader does that sum).
"""
import json
import os
import struct

import torch

MAGIC = 0x524F5554  # 'ROUT'
_ON = os.getenv('XMOE_ROUTE_LOG', '0') == '1'
_GATES = []           # the gates of this rank in construction order
_FH = None
_SIDECAR_DONE = False
_N_BLOCKS = 0


def enabled():
    return _ON


def register(gate):
    """Called by TopKGatev2.__init__ when the record is on."""
    gate._route_idx = len(_GATES)
    gate._route_buf = []
    _GATES.append(gate)


def record(gate, counts):
    """Called by the gate's forward with this micro-batch's tokens_per_expert (float from torch.histc, on the device). No sync."""
    gate._route_buf.append(counts.detach().to(torch.int32))


def _rank_world():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _dir():
    d = os.getenv('XMOE_ROUTE_LOG_DIR')
    if not d:
        p = os.getenv('profile_dir')
        d = os.path.join(os.path.dirname(os.path.abspath(p)), 'routing') if p else os.path.join(os.getcwd(), 'routing')
    os.makedirs(d, exist_ok=True)
    return d


def _pp_rank():
    try:
        from megatron.core import mpu
        return int(mpu.get_pipeline_model_parallel_rank())
    except Exception:
        return 0


def _ep_size(gate):
    try:
        import torch.distributed as dist
        g = getattr(gate, 'ep_group', None)
        return int(dist.get_world_size(group=g)) if g is not None else 1
    except Exception:
        return 0


def flush(step):
    """Once per training step, after the optimizer step: one host copy, one append. Safe to call when nothing was recorded."""
    global _FH, _SIDECAR_DONE, _N_BLOCKS
    if not _ON or not _GATES:
        return
    bufs = [g._route_buf for g in _GATES]
    n_micro = min(len(b) for b in bufs)
    if n_micro == 0:
        return
    data = torch.stack([torch.stack(b[:n_micro]) for b in bufs])      # [n_gates, n_micro, E], on the device
    for g in _GATES:
        g._route_buf = []
    host = data.to('cpu', dtype=torch.int32).contiguous()               # the one device-to-host copy of the step
    E = int(host.shape[-1])
    r, w = _rank_world()
    d = _dir()
    if _FH is None:
        _FH = open(os.path.join(d, f'rank{r}.bin'), 'ab', buffering=0)
    if not _SIDECAR_DONE:
        g0 = _GATES[0]
        meta = dict(rank=r, world=w, n_gates=len(_GATES), experts=E, k=int(getattr(g0, 'k', 0)),
                    capacity_factor=float(getattr(g0, 'capacity_factor', 0.0)), drop_tokens=bool(getattr(g0, 'drop_tokens', False)),
                    pp_rank=_pp_rank(), ep_size=_ep_size(g0),
                    format='blocks: int32 header [MAGIC, step, n_gates, n_micro, E] then int32 data [n_gates, n_micro, E] row-major')
        with open(os.path.join(d, f'rank{r}.json'), 'w') as f:
            json.dump(meta, f)
        _SIDECAR_DONE = True
    _FH.write(struct.pack('<5i', MAGIC, int(step), len(_GATES), int(n_micro), E))
    _FH.write(host.numpy().tobytes())
    _N_BLOCKS += 1


def close():
    global _FH
    if _FH is not None:
        try:
            _FH.flush()
            os.fsync(_FH.fileno())
        except Exception:
            pass
        _FH.close()
        _FH = None


def read(path):
    """Read one rank<r>.bin: a list of (step, int32 array [n_gates, n_micro, E]) in file order (numpy)."""
    import numpy as np
    out = []
    with open(path, 'rb') as f:
        raw = f.read()
    off = 0
    hdr = struct.calcsize('<5i')
    while off + hdr <= len(raw):
        magic, step, n_gates, n_micro, E = struct.unpack_from('<5i', raw, off)
        off += hdr
        if magic != MAGIC:
            raise ValueError(f'{path}: bad block magic at offset {off - hdr}')
        n = n_gates * n_micro * E
        arr = np.frombuffer(raw, dtype='<i4', count=n, offset=off).reshape(n_gates, n_micro, E)
        off += n * 4
        out.append((step, arr))
    return out
