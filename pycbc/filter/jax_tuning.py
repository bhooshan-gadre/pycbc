"""Measured, synchronized first-block tuning of the live filter.

The objective includes vetoes and host-ready results, not asynchronous dispatch.
A worst-trigger workload complements the actual first block. No tuning result
is reused across devices or banks. The whole bank remains resident: this is
batch sizing, not an out-of-core bank implementation.
"""
import gc
import time
import numpy as np


def memory_estimate(groups, batch, sg=None):
    """Surface, PSD/bins and padded SG allowance; not an XLA guarantee.

    SG tiles span at most 100 Hz in the production implementation. Use the
    largest descriptor count in a duration group so batch-dependent padding
    cannot make the estimate smaller than the final resident tile arrays.
    """
    resident, scratch = 0, 0
    for group, array in groups:
        f = array.shape[1]
        n = (f-1)*2
        q = 0
        if getattr(sg, 'do', False):
            q = max(len(sg.params.get(t.params.template_hash, '').split(','))
                    if t.params.template_hash in sg.params else 0 for t in group)
        width = min(f, int(np.ceil(100 / group[0].delta_f)) + 2) if q else 0
        tiles = q * (12*width + 4)
        # Power bins cannot have more edges than frequency samples. Allow
        # scalar weights/epochs and shared PSD/strain/k-index vectors as well.
        resident += len(group) * (8*(f+2*n) + 4*f + tiles + 128) + 32*f
        padded = 1 << (min(batch, len(group))-1).bit_length()
        scratch = max(scratch, padded * (n*8*4 + q*width*32))
    return resident + scratch


def choose_candidate(measurements, latency_ms=None):
    eligible = [m for m in measurements if m.get('status') == 'measured'
                and (latency_ms is None or m['p95_s'] * 1000 <= latency_ms)]
    if not eligible:
        raise RuntimeError('No batch candidate meets memory/latency constraints')
    return min(eligible, key=lambda m: m['objective_s'])['batch_size']


def install_layout(engine, batch):
    from .jax_matchedfilter import _DurationGroup, _import_jax
    jax, _ = _import_jax()
    engine._states = []
    gc.collect()
    engine.tgroups, engine._series, engine._lengths = [], [], []
    with jax.default_device(engine.device):
        for group, array in engine._bank_groups:
            for i in range(0, len(group), batch):
                ts, series = group[i:i+batch], array[i:i+batch]
                engine.tgroups.append(ts)
                engine._series.append(series)
                engine._lengths.append((series.shape[1]-1)*2)
                engine._states.append(_DurationGroup(series, ts, engine.device))
    engine.chunks = np.asarray([len(g) for g in engine.tgroups], dtype=np.uint32)


def tune_live_filter(engine, reader):
    started = time.perf_counter()
    maximum = max(len(g) for g, _ in engine._bank_groups)
    # Include the untuned layout's largest batch, so tuning also measures the
    # previous configuration rather than comparing only small alternatives.
    candidates = sorted({min(c, maximum) for c in engine.tuning_candidates} | {maximum})
    stats = engine.device.memory_stats() or {}
    if engine.memory_budget_mb is not None:
        budget = engine.memory_budget_mb * 2**20
    elif engine.device.platform == 'cpu':
        import psutil
        budget = psutil.virtual_memory().available * .5
    elif 'bytes_limit' in stats:
        budget = (stats['bytes_limit'] - stats.get('bytes_in_use', 0)) * .6
    else:
        raise RuntimeError('GPU memory limit unavailable; pass --jax-memory-budget-mb')
    measurements = []
    saved = {k: getattr(engine, k) for k in ('snr_threshold', 'snr_abort_threshold',
             'newsnr_threshold', 'max_triggers_in_batch')}
    sg_threshold = getattr(engine.sg_chisq, 'snr_threshold', 4)
    profiling = getattr(engine, 'profile_stages', False)
    # Select the production layout, not one optimized for artificial
    # per-stage profiling barriers. Host-ready results still synchronize each
    # complete bank evaluation used by the objective.
    engine.profile_stages = False
    engine._tuning = True
    try:
        for batch in candidates:
            estimate = memory_estimate(engine._bank_groups, batch, engine.sg_chisq)
            item = dict(batch_size=batch, estimated_bytes=estimate)
            measurements.append(item)
            if estimate > budget:
                item['status'] = 'memory_skipped'
                continue
            install_layout(engine, batch)
            workloads = {}
            for stress in (False, True):
                for key, value in saved.items():
                    setattr(engine, key, value)
                engine.sg_chisq.snr_threshold = sg_threshold
                if stress:
                    engine.snr_threshold = 0
                    engine.snr_abort_threshold = None
                    engine.newsnr_threshold = None
                    engine.max_triggers_in_batch = None
                    engine.sg_chisq.snr_threshold = 0
                # Exclude compilation and first-use PSD setup; process_data
                # returns NumPy results and therefore synchronizes all kernels.
                engine.process_data(reader)
                samples = []
                for _ in range(engine.tuning_repeats):
                    began = time.perf_counter()
                    engine.process_data(reader)
                    samples.append(time.perf_counter() - began)
                workloads['all_candidates' if stress else 'actual'] = samples
            item.update(status='measured', samples_s=workloads,
                        objective_s=max(float(np.median(x)) for x in workloads.values()),
                        p95_s=max(float(np.percentile(x, 95)) for x in workloads.values()))
        selected = choose_candidate(measurements, engine.max_batch_latency_ms)
        install_layout(engine, selected)
        engine.tuning_report = dict(batch_size=selected, candidates=measurements,
            memory_budget_bytes=budget, setup_s=time.perf_counter()-started,
            stage_profiling=False,
            objective='minimize worst median full-bank filtering+veto latency',
            latency_limit_scope='full bank, including forced all-template veto workload')
        import logging
        logging.info('JAX batch tuning: %s', engine.tuning_report)
    finally:
        for key, value in saved.items():
            setattr(engine, key, value)
        engine.sg_chisq.snr_threshold = sg_threshold
        engine.profile_stages = profiling
        engine._tuning = False
