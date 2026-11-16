#!/usr/bin/env python
"""Small live/offline search parity and synchronized warm-call benchmarks.

Run from the repository root with a PyCBC + JAX + ripplegw Python:
  python examples/search/jax/compare_searches.py --output /tmp/searches.json

Live uses the actual LiveBatchMatchedFilter.process_data contract. Offline
uses MatchedFilterControl (as pycbc_inspiral does), including symmetric
clustering and both vetoes at EVERY retained peak. Its JAX adapter reuses
the live surface, with CPU clustering and JAX vetoes. This does not execute
the MPI launcher, coincidence, or the Pegasus offline workflow.
"""
import argparse
import cProfile
import io
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'examples/live'))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/pycbc-matplotlib')

import numpy as np
from pycbc import events, vetoes
from pycbc.filter import LiveBatchMatchedFilter, MatchedFilterControl
from pycbc.filter.jax_matchedfilter import JaxLiveBatchMatchedFilter, _bucket_rows
from pycbc.scheme import CPUScheme
from pycbc.types import Array, FrequencySeries, zeros
from pycbc.vetoes.jax_chisq import (
    power_chisq_batch, power_chisq_selected, power_chisq_candidate_points,
    sg_chisq_candidate_points, sg_gate_mask)
from pycbc.vetoes.sgchisq import SingleDetSGChisq
import plot_batch_vs_serial as fixture


def timed(call, repeats):
    # Returned products are host arrays; process_data and offline() synchronize.
    start = time.perf_counter()
    result = call()
    first = time.perf_counter() - start
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        samples.append(time.perf_counter() - start)
    return result, dict(first_s=first, median_s=statistics.median(samples),
                        min_s=min(samples), max_s=max(samples), repeats=repeats)


def columns(rows):
    names = ('template_id', 'end_time', 'snr', 'coa_phase', 'chisq',
             'chisq_dof', 'sg_chisq')
    return {name: np.array([r[name] for r in rows]) for name in names}


def offline_cpu(templates, reader, sg, threshold, window):
    """Production offline MatchedFilterControl and the production CPU vetoes."""
    stilde = reader._stilde
    ntime = (len(stilde) - 1) * 2
    start, stop = fixture.valid_bounds(templates[0], reader)
    stilde.analyze = slice(start, stop)
    mem = zeros(ntime, dtype=np.complex64)
    control = MatchedFilterControl(
        fixture.F_LOW, None, threshold, ntime, stilde.delta_f, np.complex64,
        [stilde], mem, True, cluster_function='symmetric')
    power = vetoes.SingleDetPowerChisq('8', None)

    def run():
        rows = []
        for template in templates:
            mem.clear()
            mem[:len(template)] = template
            snr, norm, corr, indices, snrv = control.matched_filter_and_cluster(
                0, float(template.sigmasq(stilde.psd)), window, epoch=stilde.epoch)
            if not len(indices):
                continue
            indices = indices.astype(np.int64) + start
            raw, dof = power.values(corr, snrv, norm, stilde.psd, indices, template)
            sg_values = sg.values(stilde, template, stilde.psd, snrv, norm,
                                  raw, dof, indices)
            for j, index in enumerate(indices):
                value = snrv[j] * norm
                rows.append(dict(template_id=int(template.id),
                    end_time=reader.start_time + (index - start) / reader.sample_rate,
                    snr=float(abs(value)), coa_phase=float(np.angle(value)),
                    chisq=float(raw[j] / dof[j]), chisq_dof=int(dof[j]),
                    sg_chisq=float(sg_values[j])))
        return columns(rows)
    return run


def offline_jax(engine, reader, threshold, window):
    """Batched/serial JAX surface, CPU symmetric clustering, JAX peak vetoes."""
    def run():
        # Avoid computing the live peak vetoes; offline has different peaks.
        saved = engine.snr_threshold
        try:
            engine.snr_threshold = np.inf
            engine.process_data(reader)
        finally:
            engine.snr_threshold = saved
        rows = []
        for state in engine._states:
            start, stop = fixture.valid_bounds(state.templates[0], reader)
            surface = np.asarray(state.snr_window(start, stop - start))
            selected, indices, peaks = [], [], []
            for row in range(state.nbatch):
                values = Array(np.array(surface[row], copy=True))
                cluster = events.ThresholdCluster(values)
                snrv, idx = cluster.threshold_and_cluster(threshold, window)
                selected.extend([row] * len(idx))
                indices.extend(idx.astype(np.int64) + start)
                peaks.extend(snrv)
            if not selected:
                continue
            sel = np.asarray(selected, dtype=np.int32)
            indices = np.asarray(indices, dtype=np.int32)
            peaks = np.asarray(peaks, dtype=np.complex64)
            raw = np.asarray(power_chisq_candidate_points(
                state.q, state.norms, state.bins, state.num_bins_d,
                sel, indices, peaks, state.power_stop))
            dof = 2 * state.num_bins[sel] - 2
            reduced = raw / dof
            sg_values = np.ones(len(sel), dtype=np.float32)
            active = np.flatnonzero(state.sg['compute'][sel] & sg_gate_mask(
                np.abs(peaks), reduced, engine.sg_chisq.snr_threshold))
            if len(active):
                sg_values[active] = np.asarray(sg_chisq_candidate_points(
                    state.strain, state.sg, state.epochs, sel[active],
                    indices[active], 1 / reader.sample_rate))
            for j, row in enumerate(sel):
                rows.append(dict(template_id=int(state.templates[row].id),
                    end_time=reader.start_time + (indices[j] - start) / reader.sample_rate,
                    snr=float(abs(peaks[j])), coa_phase=float(np.angle(peaks[j])),
                    chisq=float(reduced[j]), chisq_dof=int(dof[j]),
                    sg_chisq=float(sg_values[j])))
        return columns(rows)
    return run


def compare(ref, actual, sample_rate):
    keys = lambda x: np.lexsort((x['end_time'], x['template_id']))
    a, b = keys(ref), keys(actual)
    np.testing.assert_array_equal(ref['template_id'][a], actual['template_id'][b])
    np.testing.assert_array_equal(ref['chisq_dof'][a], actual['chisq_dof'][b])
    delta = {}
    for name in ('snr', 'end_time', 'coa_phase', 'chisq', 'sg_chisq'):
        x, y = ref[name][a], actual[name][b]
        residual = y - x
        if name == 'coa_phase':
            residual = np.angle(np.exp(1j * residual))
        delta[name] = float(np.max(np.abs(residual), initial=0))
        if name == 'end_time':
            assert delta[name] <= 1 / sample_rate
        elif name == 'coa_phase':
            assert delta[name] < 1e-3
        else:
            # Same veto gate as live, with a quiet-SNR absolute tolerance.
            atol = 1e-3 if name != 'snr' else np.where(x < 5, 1e-3, 0)
            assert np.all(np.abs(residual) <= atol + 1e-4 * np.maximum(np.abs(x), 1)), (name, delta[name])
    delta['triggers'] = len(a)
    return delta


def make_engines(templates, bank, platform, threshold):
    def sg():
        return SingleDetSGChisq(bank, num_bins='8', snr_threshold=4,
                               chisq_locations=fixture.SG_LOCATIONS)
    common = dict(snr_threshold=threshold, chisq_bins='8')
    ntime = (len(templates[0]) - 1) * 2
    engines = {}
    setup = {}
    for name, constructor, extra in (
        ('cpu_live', LiveBatchMatchedFilter, {}),
        ('jax_serial', JaxLiveBatchMatchedFilter,
         dict(maxelements=ntime, jax_platform=platform)),
        ('jax_batch', JaxLiveBatchMatchedFilter, dict(jax_platform=platform))):
        start = time.perf_counter()
        engines[name] = constructor(list(templates), sg_chisq=sg(), **common, **extra)
        setup[name] = time.perf_counter() - start
    return engines, setup, sg()


def benchmark_sparse_vetoes(engine, repeats):
    """Isolate old full-group versus candidate-only power-veto work.

    The full-group result is the same for every retention policy. Vary the
    retained count to show how cost scales; these are kernel timings, not
    whole-search speedups.
    """
    result = []
    for state in engine._states:
        def full():
            return np.asarray(power_chisq_batch(
                state.q, state.peak_unnorm, state.norms, state.bins,
                state.index, state.num_bins_d))
        expected, old_timing = timed(full, repeats)
        loudest = np.argsort(state.host_abs_snr)[::-1]
        for count in sorted({1, min(4, state.nbatch), state.nbatch}):
            rows = loudest[:count].astype(np.int32)
            bucket = _bucket_rows(rows, state.nbatch)
            def selected():
                return np.asarray(power_chisq_selected(
                    state.q, state.peak_unnorm, state.norms, state.bins,
                    state.index, state.num_bins_d, bucket, state.power_stop))[:count]
            actual, new_timing = timed(selected, repeats)
            np.testing.assert_allclose(actual, expected[rows], rtol=1e-5, atol=1e-4)
            result.append(dict(templates=state.nbatch, retained=count,
                padded_candidates=len(bucket), fft_length=state.ntime,
                frequency_support=state.power_stop, full_group=old_timing,
                selected=new_timing, speedup=old_timing['median_s']/new_timing['median_s'],
                max_abs_difference=float(np.max(np.abs(actual - expected[rows])))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/jax-searches.json'))
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--bank-sizes', type=int, nargs='+', default=[10, 20])
    parser.add_argument('--jax-platform', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--skip-offline', action='store_true')
    parser.add_argument('--snr-threshold', type=float, default=4.5)
    parser.add_argument('--profile', action='store_true',
                        help='Also profile 20 warm batch calls per case')
    args = parser.parse_args()
    if args.repeats < 1 or min(args.bank_sizes) < 1:
        parser.error('repeats and bank sizes must be positive')
    import jax
    if args.jax_platform == 'gpu' and not any(d.platform == 'gpu' for d in jax.devices()):
        parser.error('No GPU visible; refusing to report CPU timings as GPU timings')
    # Keep generated fixtures together with this report, not the saved plots.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixture.OUT_DIR = str(args.output.parent / (args.output.stem + '_fixtures'))
    rows = fixture.draw_rows(max(args.bank_sizes), np.random.default_rng(fixture.SEED))
    report = dict(seed=fixture.SEED, devices=[str(d) for d in jax.devices()],
                  jax_version=jax.__version__, python=platform.python_version(),
                  snr_threshold=args.snr_threshold, valid_seconds=fixture.BLOCK_SECONDS,
                  sample_rate=fixture.SAMPLE_RATE, repeats=args.repeats,
                  workflows=[], scope=__doc__)
    profiles = []
    with CPUScheme(num_threads=1):
        bank, templates, df = fixture.build_templates(rows)
        ntime = (len(templates[0]) - 1) * 2
        psd = fixture.analytic_psd(len(templates[0]), df)
        noise = fixture.noise_from_string(fixture.PSD_NAME, ntime,
                    1 / fixture.SAMPLE_RATE, seed=fixture.SEED, low_frequency_cutoff=fixture.F_LOW)
        noise = np.asarray(noise)
        temporary = fixture.make_reader(fixture.overwhiten(noise, psd))
        start, _ = fixture.valid_bounds(templates[0], temporary)
        injection, _, _ = fixture.injection_waveform(templates[0], psd, ntime, start)
        for n in args.bank_sizes:
            engines, setup, sg = make_engines(
                templates[:n], bank, args.jax_platform, args.snr_threshold)
            for dataset, data in [('noise', noise), ('injection', noise + injection)]:
                reader = fixture.make_reader(fixture.overwhiten(data, psd))
                # Repeated calls reuse the block and PSD, measuring steady state.
                for workflow in ['live'] + ([] if args.skip_offline else ['offline']):
                    calls = {name: (lambda e=e: e.process_data(reader))
                             for name, e in engines.items()}
                    if workflow == 'offline':
                        calls = {'cpu_offline': offline_cpu(
                            templates[:n], reader, sg, args.snr_threshold, 2048)}
                        calls.update({name: offline_jax(e, reader, args.snr_threshold, 2048)
                                      for name, e in engines.items() if name.startswith('jax')})
                    results, timings = {}, {}
                    for name, call in calls.items():
                        results[name], timings[name] = timed(call, args.repeats)
                    reference = results[next(iter(calls))]
                    if dataset == 'injection':
                        assert 0 in reference['template_id'], 'Injection not recovered'
                    checks = {name: compare(reference, result, reader.sample_rate)
                              for name, result in results.items() if name.startswith('jax')}
                    item = dict(workflow=workflow, templates=n, dataset=dataset,
                                setup_s=setup, timings=timings, parity=checks)
                    item['triggers'] = {name: {key: value.tolist()
                                               for key, value in product.items()}
                                        for name, product in results.items()}
                    report['workflows'].append(item)
                    print(json.dumps({key: value for key, value in item.items()
                                      if key != 'triggers'}), flush=True)
                    if args.profile:
                        profile = cProfile.Profile()
                        profile.enable()
                        for _ in range(20):
                            calls['jax_batch']()
                        profile.disable()
                        stream = io.StringIO()
                        pstats.Stats(profile, stream=stream).sort_stats('cumulative').print_stats(25)
                        profiles.append(f'{workflow} {n} templates {dataset}\n' + stream.getvalue())
            report.setdefault('power_veto_microbenchmarks', []).extend(
                benchmark_sparse_vetoes(engines['jax_batch'], args.repeats))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if profiles:
        args.output.with_suffix('.profile.txt').write_text('\n'.join(profiles))
    print('PASS:', args.output)


if __name__ == '__main__':
    main()
