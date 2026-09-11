# coding=utf-8
"""The gradient consistency probe (2026-09-11; the V3 enablement's layout-consistency question): at the first N optimizer
steps of a run (XMOE_GRAD_PROBE=N), just before the bf16 optimizer consumes the fp32 gradients, one table on global rank 0:

    per parameter family        the true global norm     each distinct parameter counted once: a replicated tensor on one
                                                         tensor-parallel rank of one data-parallel rank, a sliced tensor on
                                                         every tensor-parallel rank of one data-parallel rank, an expert on
                                                         the first rank of its expert-data-parallel group
                                the engine-counted norm  what the bf16 optimizer's bookkeeping counts for the family (its
                                                         selection rule and reductions reproduced), so a family whose two
                                                         numbers differ is a family the reported grad norm gets wrong
                                the spread over TP       max |g - g_tp0| / max |g_tp0| over the family's replicated
                                                         tensors: 0 when the replicas carry the same gradient
                                the spread over the      the same over the expert-data-parallel group (the replicas of one
                                expert-dp group          expert): 0 after the expert gradient reduction

    the router bias buffers     their spread over the data-parallel and the tensor-parallel groups (0 = every replica moves
                                the same), and how many expert update signs would differ between the loads summed over
                                the data-parallel group alone (the mover's rule) and over data x tensor parallel (the whole
                                batch under the sequence-sharded MoE block, where each TP rank routes a disjoint token shard)

The same weights, the same batch and two layouts must agree on every "true" number to the bf16 rounding of a sum
(~1e-3 relative); a family that does not is where the layouts compute different gradients. After the step the engine's own
global norm is printed per rank for comparison with the engine-counted total. No tensor is modified; the probe is off
without the environment variable and costs a few collectives per probed step."""
import os

import torch

from deepspeed import comm as dist
from deepspeed.moe.utils import is_moe_param, is_moe_param_group
from deepspeed.runtime.utils import is_model_parallel_parameter
from deepspeed.utils import groups, logger
from deepspeed.utils.bwc import (bwc_pipeline_parallel_group, bwc_pipeline_parallel_world_size,
                                 bwc_tensor_model_parallel_group, bwc_tensor_model_parallel_rank,
                                 bwc_tensor_model_parallel_world_size)

_NAME_RULES = (('embedding', ('word_embeddings',)), ('lm head', ('lm_head', 'output_layer')),
               ('router gate', ('.gate.',)), ('indexer', ('.indexer.',)), ('attention', ('attention',)),
               ('mtp', ('mtp', 'eh_proj', 'enorm', 'hnorm')), ('mlp', ('mlp',)), ('norm', ('norm',)))


def probe_steps():
    try:
        return int(os.getenv('XMOE_GRAD_PROBE', '0') or 0)
    except ValueError:
        return 0


def _family(name, lp):
    if is_moe_param(lp):
        return 'experts'
    base = 'other'
    for fam, needles in _NAME_RULES:
        if any(n in name for n in needles):
            base = fam
            break
    if base in ('router gate', 'norm', 'indexer'):
        return base
    return f'{base} ({"sliced" if is_model_parallel_parameter(lp) else "replicated"})'


def _dist_ok():
    return dist.is_initialized() and dist.get_world_size() > 1


def _sum_over(t, group=None):
    if _dist_ok() and (group is None or dist.get_world_size(group=group) > 1):
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
    return t


def _max_over(t, group=None):
    if _dist_ok() and (group is None or dist.get_world_size(group=group) > 1):
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return t


def _min_over(t, group=None):
    if _dist_ok() and (group is None or dist.get_world_size(group=group) > 1):
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
    return t


def _spread(g, group):
    """max |g - g_first| and max |g_first| over `group` (g_first = the tensor on the group's first rank)."""
    ref = g.detach().clone()
    if _dist_ok() and group is not None and dist.get_world_size(group=group) > 1:
        dist.broadcast(ref, src=dist.get_global_rank(group, 0), group=group)
    return (g.detach() - ref).abs().max().double(), ref.abs().max().double()


def maybe_probe_before_step(engine):
    n = probe_steps()
    if n <= 0:
        return
    done = getattr(engine, '_xmoe_probe_done', 0)
    if done >= n:
        return
    engine._xmoe_probe_done = done + 1
    try:
        _probe(engine, done + 1, n)
    except Exception as e:  # the probe never takes the run down
        logger.warning(f'[grad probe] skipped: {type(e).__name__}: {e}')


def maybe_probe_after_step(engine):
    n = probe_steps()
    if n <= 0 or getattr(engine, '_xmoe_probe_done', 0) > n or not getattr(engine, '_xmoe_probe_pending', False):
        return
    engine._xmoe_probe_pending = False
    own = float(getattr(engine.optimizer, '_global_grad_norm', float('nan')))
    dev = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    world = dist.get_world_size() if _dist_ok() else 1
    vals = torch.zeros(world, dtype=torch.float64, device=dev)
    vals[dist.get_rank() if _dist_ok() else 0] = own
    _sum_over(vals)
    if (dist.get_rank() if _dist_ok() else 0) == 0:
        uniq = sorted(set(round(v, 6) for v in vals.tolist()))
        print(f"[grad probe] the engine's own global grad norm after the step: {uniq[0]:.6f}" +
              (f'  (differs across ranks: {uniq})' if len(uniq) > 1 else '  (identical on every rank)'), flush=True)


def _probe(engine, k, n):
    opt = engine.optimizer
    mpu = engine.mpu
    dev = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    rank = dist.get_rank() if _dist_ok() else 0
    tp = bwc_tensor_model_parallel_world_size(mpu)
    tp_rank = bwc_tensor_model_parallel_rank(mpu)
    tp_group = bwc_tensor_model_parallel_group(mpu) if tp > 1 else None
    pp = bwc_pipeline_parallel_world_size(mpu) if mpu is not None else 1
    pp_group = bwc_pipeline_parallel_group(mpu) if pp > 1 else None
    dp_rank = mpu.get_data_parallel_rank() if mpu is not None else 0
    dp_group = mpu.get_data_parallel_group() if mpu is not None else None
    mp_group = mpu.get_model_parallel_group() if mpu is not None else None
    param_names = getattr(opt, 'param_names', None) or getattr(engine, 'param_names', {})

    fams = {}   # family -> dict of accumulators (float64 tensors on the device)

    def acc(fam):
        if fam not in fams:
            fams[fam] = dict(tensors=0, numel=0, true=torch.zeros((), dtype=torch.float64, device=dev),
                             eng_nonmoe=torch.zeros((), dtype=torch.float64, device=dev),
                             eng_moe={}, tp_d=torch.zeros((), dtype=torch.float64, device=dev),
                             tp_m=torch.zeros((), dtype=torch.float64, device=dev),
                             edp_d=torch.zeros((), dtype=torch.float64, device=dev),
                             edp_m=torch.zeros((), dtype=torch.float64, device=dev), n_true=0)
        return fams[fam]

    # the gradient tensors the optimizer is about to consume: the BF16_Optimizer holds every parameter's full fp32
    # gradient; the ZeRO-1 optimizer (DeepSpeedZeroOptimizer) holds this rank's partition of the data-parallel-averaged
    # gradient, one slice per parameter in the partition (the same slice of a replicated parameter on every TP rank)
    partitioned = hasattr(opt, 'averaged_gradients') and hasattr(opt, 'bit16_groups')
    if partitioned:
        def iter_grads():
            for i, params in enumerate(opt.params_in_partition):
                grads = opt.averaged_gradients.get(i) if isinstance(opt.averaged_gradients, dict) else opt.averaged_gradients[i]
                if grads is None:
                    continue
                for lp, g in zip(params, grads):
                    yield i, lp, g
    else:
        def iter_grads():
            for i, group in enumerate(opt.bf16_groups):
                for j, lp in enumerate(group):
                    if opt.fp32_groups_has_gradients[i][j]:
                        yield i, lp, opt.fp32_groups_gradients[i][j]

    for i, lp, g in iter_grads():
        pg = opt.optimizer.param_groups[i]
        moe_group = is_moe_param_group(pg)
        ep_name = pg.get('name') if moe_group else None
        edp_group = groups._get_expert_data_parallel_group(ep_name) if (moe_group and not partitioned) else None
        edp_rank = dist.get_rank(group=edp_group) if (edp_group is not None and _dist_ok()) else 0
        name = param_names.get(lp, f'group{i}.param?') if isinstance(param_names, dict) else f'group{i}.param?'
        fam = _family(name, lp)
        a = acc(fam)
        sliced = is_model_parallel_parameter(lp)
        moe = is_moe_param(lp)
        sq = g.detach().double().pow(2).sum()
        # the true count: each distinct element once
        if partitioned:
            count = moe or sliced or tp_rank == 0          # partitions are disjoint over dp / expert-dp; TP replicas hold the same slice
        elif moe:
            count = edp_rank == 0
        elif sliced:
            count = dp_rank == 0
        else:
            count = dp_rank == 0 and tp_rank == 0
        if count:
            a['true'] += sq
            a['numel'] += g.numel()
            a['n_true'] += 1
        a['tensors'] += 1
        # the engine's count (bf16_optimizer.get_grads_for_norm + get_global_norm_of_tensors); under ZeRO-1 the engine's rule
        # (stage_1_and_2.get_grad_norm_direct) is the same selection as the true count, so it is not repeated
        if not partitioned and (tp_rank == 0 or sliced or moe):
            if moe:
                a['eng_moe'].setdefault(ep_name, torch.zeros((), dtype=torch.float64, device=dev))
                a['eng_moe'][ep_name] += sq
            else:
                a['eng_nonmoe'] += sq
        # the spreads
        if moe:
            if edp_group is not None and _dist_ok() and dist.get_world_size(group=edp_group) > 1:
                d, m = _spread(g, edp_group)
                a['edp_d'] = torch.maximum(a['edp_d'], d)
                a['edp_m'] = torch.maximum(a['edp_m'], m)
        elif not sliced and tp > 1:
            d, m = _spread(g, tp_group)
            a['tp_d'] = torch.maximum(a['tp_d'], d)
            a['tp_m'] = torch.maximum(a['tp_m'], m)

    # reductions: the true sums over the world; the engine's over its groups; the spreads: max over the world. The family
    # list is unified across the world first: pipeline stages hold different families (the embedding on the first stage, the
    # head on the last, the dense MLP on the first), and a collective over per-rank lists of different lengths hangs.
    if _dist_ok():
        import torch.distributed as _td
        gathered = [None] * _td.get_world_size()
        _td.all_gather_object(gathered, sorted(fams))
        names = sorted(set().union(*[set(g) for g in gathered]))
    else:
        names = sorted(fams)
    for f in names:
        acc(f)
    true = torch.stack([fams[f]['true'] for f in names]) if names else torch.zeros(0, dtype=torch.float64, device=dev)
    _sum_over(true)
    numel = torch.tensor([fams[f]['numel'] for f in names], dtype=torch.float64, device=dev)
    _sum_over(numel)
    eng = []
    for f in names:
        a = fams[f]
        e = a['eng_nonmoe'].clone()
        if mp_group is not None:
            _sum_over(e, mp_group)
        for ep_name, sq in a['eng_moe'].items():
            s = sq.clone()
            if groups._get_expert_model_parallel_world_size() > 1:
                if mp_group is not None:
                    _sum_over(s, mp_group)
            elif pp_group is not None:
                _sum_over(s, pp_group)
            _sum_over(s, groups._get_expert_parallel_group(ep_name))
            e += s
        eng.append(e)
    eng = torch.stack(eng) if eng else torch.zeros(0, dtype=torch.float64, device=dev)
    spreads = torch.stack([torch.stack([fams[f]['tp_d'], fams[f]['tp_m'], fams[f]['edp_d'], fams[f]['edp_m']]) for f in names]) \
        if names else torch.zeros(0, 4, dtype=torch.float64, device=dev)
    _max_over(spreads)
    tensors = torch.tensor([fams[f]['tensors'] for f in names], dtype=torch.float64, device=dev)
    _sum_over(tensors)

    # the router bias buffers
    gates = [m for m in engine.module.modules() if hasattr(m, 'expert_bias') and hasattr(m, 'load_acc')]
    bias_dp = torch.zeros((), dtype=torch.float64, device=dev)
    bias_tp = torch.zeros((), dtype=torch.float64, device=dev)
    sign_diff = torch.zeros((), dtype=torch.float64, device=dev)
    n_signs = 0
    for gm in gates:
        b = gm.expert_bias.detach().double()
        bmax, bmin = _max_over(b.clone(), dp_group), _min_over(b.clone(), dp_group)
        bias_dp = torch.maximum(bias_dp, (bmax - bmin).max())
        if tp > 1:
            tmax, tmin = _max_over(b.clone(), tp_group), _min_over(b.clone(), tp_group)
            bias_tp = torch.maximum(bias_tp, (tmax - tmin).max())
        load_dp = _sum_over(gm.load_acc.detach().double().clone(), dp_group)
        load_full = _sum_over(load_dp.clone(), tp_group) if tp > 1 else load_dp
        s_dp = torch.sign(load_dp.mean() - load_dp)
        s_full = torch.sign(load_full.mean() - load_full)
        sign_diff += (s_dp != s_full).sum()
        n_signs += load_dp.numel()
    _max_over(sign_diff)      # the worst TP rank
    _max_over(bias_dp)
    _max_over(bias_tp)
    n_gates_total = torch.tensor(float(len(gates)), dtype=torch.float64, device=dev)
    if pp_group is not None:
        _sum_over(n_gates_total, pp_group)
    n_signs_t = torch.tensor(float(n_signs), dtype=torch.float64, device=dev)
    if pp_group is not None:
        _sum_over(n_signs_t, pp_group)
        _sum_over(sign_diff, pp_group)

    engine._xmoe_probe_pending = True
    if rank != 0:
        return
    dp = dist.get_world_size(group=dp_group) if (dp_group is not None and _dist_ok()) else 1
    ep_names = sorted({nm for f in names for nm in fams[f]['eng_moe']})
    lines = [f'[grad probe] probed step {k} of {n} of this run: tp {tp}, pp {pp}, dp {dp}, {"ZeRO-1 partitions" if partitioned else "full fp32 gradients"}, expert groups {ep_names or "none"}',
             f'  {"family":26s} {"tensors":>8s} {"numel":>14s} {"true norm":>14s} {"engine-counted":>14s} {"spread over TP":>15s} {"spread over expert-dp":>22s}']
    tot_true = tot_eng = 0.0
    for idx, f in enumerate(names):
        t, e = float(true[idx]), float(eng[idx])
        tot_true += t
        tot_eng += e
        d_tp, m_tp, d_e, m_e = spreads[idx].tolist()
        s_tp = '-' if (tp == 1 or m_tp == 0) else f'{d_tp / m_tp:.2e}'
        s_e = '-' if m_e == 0 else f'{d_e / m_e:.2e}'
        e_txt = '(= true)' if partitioned else f'{e ** 0.5:.6f}'
        lines.append(f'  {f:26s} {int(tensors[idx]):8d} {int(numel[idx]):14d} {t ** 0.5:14.6f} {e_txt:>14s} {s_tp:>15s} {s_e:>22s}')
    lines.append(f'  {"TOTAL":26s} {"":8s} {int(numel.sum()):14d} {tot_true ** 0.5:14.6f} ' + ('(= true)' if partitioned else f'{tot_eng ** 0.5:14.6f}'))
    lines.append(f'  router bias: {int(n_gates_total)} gates; spread over dp {float(bias_dp):.3e}, over tp {float(bias_tp):.3e}; '
                 f'update signs differing between loads summed over dp alone and over dp x tp: {int(sign_diff)} of {int(n_signs_t)}')
    print('\n'.join(lines), flush=True)
