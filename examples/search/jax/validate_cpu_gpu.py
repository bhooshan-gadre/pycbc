#!/usr/bin/env python
"""Validate native/JAX serial and batched filtering, including real GPU execution.

Default: require a GPU, test IMRPhenomD (ripplegw) and regular LAL TaylorF2
on eight valid seconds of colored noise and noise plus an injection. See
GPU_TESTS.md for P100 installation and output interpretation. No MPI needed.
Native CPU transforms use FFTW by default; --fft-backend numpy is optional.
"""
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'examples/live'))

import numpy as np


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def command_output(command):
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return p.stdout.strip() if p.returncode == 0 else p.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)


def select_native_fft_backend(name):
    """Require the requested CPU FFT backend, without PyCBC's fallback list."""
    from pycbc.fft import backend_cpu, backend_support
    if name not in backend_cpu._alist:
        raise RuntimeError(
            f'Requested native CPU FFT backend {name!r} is unavailable. '
            'For FFTW, install both single- and double-precision FFTW libraries '
            '(micromamba install -p "$JAX_ENV_PREFIX" -c conda-forge fftw). '
            'Alternatively select --fft-backend numpy explicitly.')
    backend_support.set_backend([name])
    module = backend_support.get_backend().__name__
    expected = {'fftw': 'pycbc.fft.fftw', 'numpy': 'pycbc.fft.npfft'}[name]
    if module != expected:
        raise RuntimeError(f'FFT backend selection failed: expected {expected}, got {module}')
    return dict(name=name, module=module, cpu_threads=1)


def timed(call, repeats):
    started = time.perf_counter()
    result = call()
    first = time.perf_counter() - started
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = call()  # Host trigger/veto arrays synchronize device execution.
        samples.append(time.perf_counter() - started)
    return result, dict(first_s=first, median_s=statistics.median(samples),
        min_s=min(samples), p95_s=float(np.percentile(samples, 95)), repeats=repeats)


def aligned_result(result, ids):
    order = {int(tid): i for i, tid in enumerate(result['template_id'])}
    if set(order) != set(ids) or len(order) != len(result['template_id']):
        raise AssertionError('Missing or duplicate template IDs')
    return {key: np.asarray(result[key])[[order[int(t)] for t in ids]]
            for key in ['template_id', 'snr', 'end_time', 'coa_phase',
                        'chisq', 'chisq_dof', 'sg_chisq']}


def compare_products(reference, actual, rho_ref, rho, sample_rate, surface_atol=1e-3):
    """Gate scientific parity; retain failures instead of aborting plotting."""
    errors, metrics = [], {}
    if not np.isfinite(rho).all():
        errors.append('nonfinite SNR surface')
    delta = np.abs(rho - rho_ref)
    allowed = surface_atol + 1e-4 * np.maximum(np.abs(rho_ref), 1)
    metrics['complex_snr_absolute_tolerance'] = surface_atol
    metrics['max_complex_snr_error'] = float(np.max(delta))
    metrics['rms_complex_snr_error'] = float(np.sqrt(np.mean(delta**2)))
    metrics['max_snr_magnitude_error'] = float(np.max(np.abs(np.abs(rho)-np.abs(rho_ref))))
    if not np.all(delta <= allowed):
        errors.append(f'complex SNR surface exceeds {surface_atol:g} absolute + 1e-4 relative gate')
    for key in ['snr', 'chisq', 'sg_chisq']:
        x, y = reference[key], actual[key]
        d = np.abs(y-x)
        metrics['max_' + key + '_error'] = float(np.max(d))
        atol = np.where(x < 5, 1e-3, 0) if key == 'snr' else 1e-3
        if not np.isfinite(y).all() or not np.all(d <= atol + 1e-4*np.maximum(np.abs(x), 1)):
            errors.append(key + ' peak parity gate failed')
    phase_error = np.abs(np.angle(np.exp(1j*(actual['coa_phase']-reference['coa_phase']))))
    metrics['max_peak_phase_error_rad'] = float(np.max(phase_error))
    dt = np.abs(actual['end_time']-reference['end_time']) * sample_rate
    metrics['max_peak_time_error_samples'] = float(np.max(dt))
    if not np.isfinite(phase_error).all() or np.max(phase_error) > 1e-3:
        errors.append('peak phase exceeds 1e-3 rad')
    if not np.isfinite(dt).all() or np.max(dt) > 1.0001:
        errors.append('peak time differs by more than one sample')
    if not np.array_equal(actual['chisq_dof'], reference['chisq_dof']):
        errors.append('chi-squared degrees of freedom differ')
    metrics['passed'] = not errors
    return metrics, errors


def compare_curves(reference, actual):
    metrics, errors = {}, []
    for key in ['power', 'sg']:
        ref, values = reference[key], actual[key]
        delta = np.abs(values-ref)
        metrics['max_' + key + '_curve_error'] = float(np.max(delta))
        passed = bool(np.isfinite(values).all() and
                      np.all(delta <= 1e-3 + 1e-4*np.maximum(np.abs(ref), 1)))
        metrics[key + '_curve_passed'] = passed
        if not passed:
            errors.append('sampled ' + key + ' veto time-series gate failed')
    return metrics, errors


def build_fixture(args, model, rows, directory):
    import plot_batch_vs_serial as f
    from pycbc.waveform.bank import LiveFilterBank
    f.SAMPLE_RATE = args.sample_rate
    f.BLOCK_SECONDS = 8.0
    f.F_LOW = args.f_lower
    f.write_bank(str(directory / 'bank.hdf'), rows)
    bank = LiveFilterBank(str(directory / 'bank.hdf'), args.sample_rate,
        minimum_buffer=8, approximant=model, increment=8,
        low_frequency_cutoff=args.f_lower)
    df = min(bank.freq_resolution_for_template(i) for i in range(len(bank)))
    templates = [bank.get_template(i, delta_f=df) for i in range(len(bank))]
    ntime = (len(templates[0])-1)*2
    psd = f.analytic_psd(len(templates[0]), df)
    noise = f.noise_from_string(f.PSD_NAME, ntime, 1/args.sample_rate,
        seed=args.seed, low_frequency_cutoff=args.f_lower)
    stilde = f.overwhiten(np.asarray(noise), psd)
    reader = f.make_reader(stilde)
    start, stop = f.valid_bounds(templates[0], reader)
    # Native frequency-domain bank injection, independent of ripplegw/JAX.
    # Alpha sets the optimal SNR; noise changes its recovered value.
    from pycbc.types import FrequencySeries
    h = np.asarray(templates[0])
    sigma2 = float(templates[0].sigmasq(stilde.psd))
    alpha = args.injection_snr / np.sqrt(sigma2)
    target_index = start + int(5 * args.sample_rate)
    phase = np.exp(-2j*np.pi*np.arange(len(h))*target_index/ntime)
    addition = np.zeros(len(h), dtype=np.complex128)
    weight = np.asarray(stilde.psd)
    good = weight > 0
    addition[good] = alpha*h[good]*phase[good]/weight[good]
    injected = FrequencySeries((np.asarray(stilde)+addition).astype(np.complex64),
                              delta_f=df, epoch=stilde.epoch)
    injected.psd = stilde.psd
    readers = {'noise': reader, 'injection': f.make_reader(injected)}
    return bank, templates, readers, start, stop


class NativeSerial:
    def __init__(self, templates, reader, sg, sample_series=None):
        from pycbc import vetoes
        from pycbc.types import zeros
        self.templates, self.reader, self.sg = templates, reader, sg
        self.sample_series = sample_series
        self.power = vetoes.SingleDetPowerChisq('8', None)
        ntime = (len(templates[0])-1)*2
        self.out = zeros(ntime, dtype=np.complex64)
        self.corr = zeros(ntime, dtype=np.complex64)

    def run(self, collect=False):
        from pycbc.filter import matched_filter_core
        from pycbc.types import FrequencySeries
        import plot_batch_vs_serial as f
        stilde = self.reader._stilde
        rows, surfaces, packs = [], [], []
        for template in self.templates:
            target = template
            if self.sample_series is not None:
                target = FrequencySeries(self.sample_series[int(template.id)],
                    delta_f=template.delta_f, epoch=template.epoch)
            sigma2 = float(template.sigmasq(stilde.psd))
            unnorm, corr, norm = matched_filter_core(target, stilde,
                h_norm=sigma2, out=self.out, corr_out=self.corr)
            start, stop = f.valid_bounds(template, self.reader)
            raw = np.asarray(unnorm)
            index = start + int(np.argmax(np.abs(raw[start:stop])**2))
            snrv = np.array([raw[index]], dtype=np.complex64)
            indices = np.array([index], dtype=np.float64)
            chisq, dof = self.power.values(corr, snrv, norm, stilde.psd, indices, template)
            sg = self.sg.values(stilde, template, stilde.psd, snrv, norm, chisq, dof, indices)
            peak = snrv[0] * norm
            rows.append(dict(template_id=int(template.id), snr=float(abs(peak)),
                end_time=self.reader.start_time + (index-start)/self.reader.sample_rate,
                coa_phase=float(np.angle(peak)), chisq=float(chisq[0]/dof[0]),
                chisq_dof=int(dof[0]), sg_chisq=float(sg[0])))
            if collect:
                surfaces.append((raw[start:stop]*np.float32(norm)).copy())
                packs.append((raw.copy(), np.asarray(corr).copy(), float(norm)))
        result = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        return (result, np.stack(surfaces), packs) if collect else result


def assert_device(array, device, name):
    locations = array.devices()
    if locations != {device}:
        raise AssertionError(f'{name} is on {locations}, expected {device}')


def engine_surface(engine, ids, start, stop, device):
    rows, samples, links = {}, {}, {}
    for state in engine._states:
        for name in ['h', 'q', 'norms', 'bins', 'peak', 'index']:
            assert_device(getattr(state, name), device, name)
        window = state.snr_window(start, stop-start)
        assert_device(window, device, 'normalized SNR window')
        values = np.asarray(window)
        for i, template in enumerate(state.templates):
            tid = int(template.id)
            rows[tid] = values[i].copy()
            samples[tid] = np.asarray(state.h[i]).copy()
            links[tid] = (state, i)
    return np.stack([rows[int(t)] for t in ids]), samples, links


def native_curves(serial, pack, template, indices):
    from pycbc.types import Array
    raw, corr, norm = pack
    snrv = raw[indices]
    stilde = serial.reader._stilde
    chisq, dof = serial.power.values(Array(corr), snrv, norm, stilde.psd,
                                     indices.astype(float), template)
    sg = serial.sg.values(stilde, template, stilde.psd, snrv, norm,
                           chisq, dof, indices.astype(float))
    return np.asarray(chisq)/np.asarray(dof), np.asarray(sg)


def jax_curves(engine, link, indices, device, chunk=16):
    from pycbc.vetoes.jax_chisq import (
        power_chisq_candidate_points, sg_chisq_candidate_points, sg_gate_mask)
    state, row = link
    power, sg_out = [], []
    for begin in range(0, len(indices), chunk):
        selected = indices[begin:begin+chunk].astype(np.int32)
        rows = np.full(len(selected), row, dtype=np.int32)
        # The surface may be raw/lazy on CPU and normalized on GPU.
        peaks = state.snr_window(int(selected.min()), int(selected.max()-selected.min()+1))
        peaks = peaks[row, selected-selected.min()]
        raw = power_chisq_candidate_points(state.q, state.norms, state.bins,
            state.num_bins_d, rows, selected, peaks, state.power_stop)
        assert_device(raw, device, 'power veto time-series samples')
        dof = 2*state.num_bins[row]-2
        reduced = np.asarray(raw)/dof
        sg = np.ones(len(selected), dtype=np.float32)
        active = np.flatnonzero(sg_gate_mask(np.abs(np.asarray(peaks)), reduced, 4))
        if state.sg is not None and state.sg['compute'][row] and len(active):
            values = sg_chisq_candidate_points(state.strain, state.sg, state.epochs,
                rows[active], selected[active], 1/engine.data.sample_rate)
            assert_device(values, device, 'SG veto time-series samples')
            sg[active] = np.asarray(values)
        power.extend(reduced)
        sg_out.extend(sg)
    return np.array(power), np.array(sg_out)


def draw_plots(directory, model, dataset, ids, representative, times, products,
               curves, curve_times):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(products)
    reference = products['native_serial']['rho']
    row = ids.index(representative)
    fig, axes = plt.subplots(5, 2, figsize=(15, 17), constrained_layout=True)
    for name, product in products.items():
        rho = product['rho'][row]
        for k, transform in enumerate([np.abs, np.real, np.imag]):
            axes[k, 0].plot(times, transform(rho), lw=.8, label=name)
            if name != 'native_serial':
                axes[k, 1].plot(times, transform(rho)-transform(reference[row]), lw=.7, label=name)
        for k, field in enumerate(['power', 'sg'], 3):
            axes[k, 0].plot(curve_times, curves[name][field], '.-', ms=2, lw=.7)
            if name != 'native_serial':
                axes[k, 1].plot(curve_times, curves[name][field]-curves['native_serial'][field], '.-', ms=2, lw=.7)
    labels = ['SNR magnitude', 'Re(complex SNR)', 'Im(complex SNR)', 'Reduced power chi-squared', 'Sine-Gaussian chi-squared']
    for k, label in enumerate(labels):
        axes[k, 0].set_ylabel(label)
        axes[k, 1].set_ylabel('Difference from native serial')
        for ax in axes[k]:
            ax.set_xlabel('Time in valid block (s)')
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'{model}, {len(ids)} templates, {dataset}, template {representative}\n'
                 'SNR: full sample rate; vetoes: sampled points, including the peak')
    fig.savefig(directory/'timeseries.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    ref = products['native_serial']['result']
    for name, product in products.items():
        actual = product['result']
        axes[0, 0].plot(ids, actual['snr'], '.-', lw=.7, label=name)
        if name == 'native_serial':
            continue
        axes[0, 1].plot(ids, actual['snr']-ref['snr'], '.-')
        axes[1, 0].plot(ids, np.angle(np.exp(1j*(actual['coa_phase']-ref['coa_phase']))), '.-')
        axes[1, 1].plot(ids, actual['end_time']-ref['end_time'], '.-')
        axes[2, 0].plot(ids, actual['chisq']-ref['chisq'], '.-')
        axes[2, 1].plot(ids, actual['sg_chisq']-ref['sg_chisq'], '.-')
    for ax, label in zip(axes.flat, ['Time/phase maximized SNR', 'Peak SNR difference',
            'Wrapped peak phase difference (rad)', 'Peak time difference (s)',
            'Reduced power chi-squared difference', 'SG chi-squared difference']):
        ax.set_ylabel(label); ax.set_xlabel('Template ID'); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'{model}, {dataset}: all {len(ids)} templates')
    fig.savefig(directory/'peaks.png', dpi=140); plt.close(fig)
    shown = ['native_serial'] + [n for n in names if n.endswith('_batch')]
    fig, axes = plt.subplots(2, len(shown), figsize=(5*len(shown), 7), squeeze=False, constrained_layout=True)
    vmax = max(float(np.abs(products[n]['rho']).max()) for n in shown)
    error_max = max([float(np.abs(np.abs(products[n]['rho'])-np.abs(reference)).max()) for n in shown] + [1e-7])
    for i, name in enumerate(shown):
        values = products[name]['rho']
        kwargs = dict(origin='lower', aspect='auto', extent=[0, 8, -.5, len(ids)-.5], interpolation='nearest')
        image = axes[0, i].imshow(np.abs(values), vmin=0, vmax=vmax, **kwargs)
        diff = np.abs(values)-np.abs(reference)
        error = axes[1, i].imshow(diff, cmap='coolwarm', vmin=-error_max, vmax=error_max, **kwargs)
        axes[0, i].set_title(name)
        for ax in axes[:, i]:
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Template index')
    fig.colorbar(image, ax=list(axes[0]), label='SNR magnitude')
    fig.colorbar(error, ax=list(axes[1]), label='SNR magnitude difference from native serial')
    fig.suptitle(f'{model}, {dataset}: time/template surface')
    fig.savefig(directory/'surface.png', dpi=130); plt.close(fig)


def run_case(args, model, bank, templates, reader, start, stop, directory, devices):
    import jax
    from pycbc.filter import LiveBatchMatchedFilter
    from pycbc.filter.jax_matchedfilter import JaxLiveBatchMatchedFilter, _approximant_name
    from pycbc.vetoes.sgchisq import SingleDetSGChisq
    ids = [int(t.id) for t in templates]
    def sg():
        return SingleDetSGChisq(bank, num_bins='8', snr_threshold=4,
                               chisq_locations=['mtotal>0:10-0'])
    class ValidationFilter(JaxLiveBatchMatchedFilter):
        def _generate_and_compare(self, template):
            if _approximant_name(template) == 'TaylorF2':
                # Explicitly test regular LAL TF2, not the separate SPA model.
                return None, dict(template_id=int(template.id), approximant='TaylorF2',
                    implementation='lal-taylorf2', use_generated=False,
                    match_template=1.0, waveform_source='native LAL TaylorF2')
            generated, report = super()._generate_and_compare(template)
            if report['implementation'] != 'ripple' or not report['use_generated']:
                raise AssertionError('IMRPhenomD did not use ripplegw; refusing a fallback pass: ' + str(report))
            report['generation_device'] = str(jax.config.jax_default_device)
            return generated, report
    products, timings, metadata, failures = {}, {}, {}, []
    controls, control_curves = {}, {}
    serial = NativeSerial(templates, reader, sg())
    result, timing = timed(serial.run, args.repeats)
    result, rho, packs = serial.run(collect=True)
    products['native_serial'] = dict(result=aligned_result(result, ids), rho=rho)
    timings['native_serial'] = dict(setup_s=0, **timing)
    batch = LiveBatchMatchedFilter(list(templates), snr_threshold=0, chisq_bins='8', sg_chisq=sg())
    result, timing = timed(lambda: batch.process_data(reader), args.repeats)
    surfaces = {}
    for template in templates:
        norm = 4*template.delta_f/np.sqrt(template.sigmasq(reader._stilde.psd))
        surfaces[int(template.id)] = (np.asarray(template.out)[start:stop]*np.float32(norm)).copy()
    products['native_batch'] = dict(result=aligned_result(result, ids), rho=np.stack([surfaces[i] for i in ids]))
    timings['native_batch'] = dict(setup_s=0, **timing)
    native = products['native_serial']
    representative = ids[0] if directory.name == 'injection' else ids[int(np.argmax(native['result']['snr']))]
    ri = ids.index(representative)
    peak = start + int(round((native['result']['end_time'][ri]-reader.start_time)*args.sample_rate))
    points = np.unique(np.concatenate([np.linspace(start, stop-1, args.veto_samples).astype(int),
        np.arange(max(start, peak-128), min(stop, peak+129), 16), [peak]]))
    pcurve, scurve = native_curves(serial, packs[ri], templates[ri], points)
    curves = {'native_serial': dict(power=pcurve, sg=scurve)}
    # CPU batch has the same production CPU veto algorithms. Re-evaluate its
    # own correlation and SNR workspace for the representative template.
    norm = 4*templates[ri].delta_f/np.sqrt(templates[ri].sigmasq(reader._stilde.psd))
    pack = (np.asarray(templates[ri].out).copy(), np.asarray(templates[ri].cout).copy(), norm)
    pcurve, scurve = native_curves(serial, pack, templates[ri], points)
    curves['native_batch'] = dict(power=pcurve, sg=scurve)
    ntime = (len(templates[0])-1)*2
    for platform_name, device in devices.items():
        for mode in ['serial', 'batch']:
            name = 'jax_' + platform_name + '_' + mode
            with jax.default_device(device):
                began = time.perf_counter()
                engine = ValidationFilter(list(templates), snr_threshold=0,
                    chisq_bins='8', sg_chisq=sg(), jax_platform=platform_name,
                    processing_device_id=device.id, maxelements=ntime if mode == 'serial' else 2**27)
                setup = time.perf_counter()-began
                if engine.device != device:
                    raise AssertionError('Unexpected device fallback')
                result, timing = timed(lambda: engine.process_data(reader), args.repeats)
                rho, chosen, links = engine_surface(engine, ids, start, stop, device)
                products[name] = dict(result=aligned_result(result, ids), rho=rho)
                timings[name] = dict(setup_s=setup, **timing)
                metadata[name] = dict(device=str(device), device_kind=device.device_kind,
                    chunks=[int(x) for x in engine.chunks], waveforms=engine.startup_comparisons)
                pcurve, scurve = jax_curves(engine, links[representative], points, device)
                curves[name] = dict(power=pcurve, sg=scurve)
                # Independent CPU filtering of the exact chosen waveform
                # separates GPU kernel errors from waveform-generator differences.
                control = NativeSerial(templates, reader, sg(), sample_series=chosen)
                control_result, control_rho, control_packs = control.run(collect=True)
                control_result = aligned_result(control_result, ids)
                controls[name] = dict(result=control_result, rho=control_rho)
                pc, sc = native_curves(control, control_packs[ri], templates[ri], points)
                control_curves[name] = dict(power=pc, sg=sc)
                filter_metrics, errors = compare_products(control_result,
                    products[name]['result'], control_rho, rho, args.sample_rate)
                curve_metrics, curve_errors = compare_curves(control_curves[name], curves[name])
                filter_metrics.update(curve_metrics)
                errors.extend(curve_errors)
                filter_metrics['passed'] = not errors
                metadata[name]['filter_only'] = filter_metrics
                failures.extend([name + ' filter-only: ' + e for e in errors])
                del control, engine
    metrics = {}
    for name, product in products.items():
        model_comparison = model == 'IMRPhenomD' and name.startswith('jax_')
        m, errors = compare_products(native['result'], product['result'], native['rho'], product['rho'],
            args.sample_rate, surface_atol=5e-3 if model_comparison else 1e-3)
        metrics[name] = m
        failures.extend([name + ': ' + e for e in errors])
        cm, ce = compare_curves(curves['native_serial'], curves[name])
        m.update(cm)
        # Different PhenomD implementations are not bit-identical waveforms.
        # Keep strict cross-model peak/surface gates; off-peak veto differences
        # are reported, while kernel veto accuracy uses the exact-waveform
        # independent CPU control above. TaylorF2 uses identical waveforms.
        if model != 'IMRPhenomD' or not name.startswith('jax_'):
            errors.extend(ce)
            failures.extend([name + ': ' + e for e in ce])
        m['cross_model_veto_curves_gated'] = model != 'IMRPhenomD' or not name.startswith('jax_')
        m['passed'] = not errors and metadata.get(name, {}).get('filter_only', {}).get('passed', True)
    # Direct serial/batch and CPU/GPU comparisons include independently
    # generated waveforms on the respective devices, not a shared host bank.
    consistency = {}
    baseline = products['jax_cpu_serial']
    for name, product in products.items():
        if not name.startswith('jax_'):
            continue
        m, errors = compare_products(baseline['result'], product['result'],
            baseline['rho'], product['rho'], args.sample_rate)
        cm, ce = compare_curves(curves['jax_cpu_serial'], curves[name])
        m.update(cm); errors.extend(ce); m['passed'] = not errors
        consistency[name] = m
        failures.extend([name + ' vs JAX CPU serial: ' + e for e in errors])
    if directory.name == 'injection' and native['result']['snr'][0] < args.injection_snr*.7:
        failures.append('Injected template was not recovered with the expected SNR')
    times = np.arange(stop-start)/args.sample_rate
    curve_times = (points-start)/args.sample_rate
    draw_plots(directory, model, directory.name, ids, representative, times,
               products, curves, curve_times)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
    for name, control in controls.items():
        axes[0].plot(times, np.abs(products[name]['rho'][ri]-control['rho'][ri]), lw=.7, label=name)
        for ax, key in zip(axes[1:], ['power', 'sg']):
            ax.plot(curve_times, curves[name][key]-control_curves[name][key], '.-', ms=2, lw=.7)
    for ax, label in zip(axes, ['Absolute complex SNR error', 'Reduced power chi-squared error', 'SG chi-squared error']):
        ax.set_ylabel(label); ax.set_xlabel('Time in valid block (s)'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'{model}, {directory.name}: JAX minus native CPU filtering of the exact same waveform')
    fig.savefig(directory/'filter_residuals.png', dpi=140); plt.close(fig)
    if not args.no_arrays:
        arrays = dict(time_s=times, template_ids=ids, veto_time_s=curve_times,
                      representative_template_id=representative)
        for name, product in products.items():
            arrays[name+'_rho'] = product['rho']
            for key, values in product['result'].items():
                arrays[name+'_'+key] = values
            arrays[name+'_power_curve'] = curves[name]['power']
            arrays[name+'_sg_curve'] = curves[name]['sg']
            if name in controls:
                arrays[name+'_control_rho'] = controls[name]['rho']
                for key in ['power', 'sg']:
                    arrays[name+'_control_'+key+'_curve'] = control_curves[name][key]
        np.savez_compressed(directory/'arrays.npz', **arrays)
    report = dict(model=model, templates=len(templates), dataset=directory.name,
        native_fft_backend=args.fft_backend,
        fft_length=ntime, valid_length=stop-start, waveform_sources=dict(
            TaylorF2='regular LAL TaylorF2, copied to the selected filter device',
            IMRPhenomD='ripplegw generated under the selected JAX device context'),
        metrics=metrics, timings=timings, implementations=metadata,
        jax_cpu_serial_consistency=consistency,
        representative_template_id=representative, failures=failures, passed=not failures)
    for name, timing in timings.items():
        timing['speedup_vs_native_serial'] = timings['native_serial']['median_s']/timing['median_s']
        timing['speedup_vs_native_batch'] = timings['native_batch']['median_s']/timing['median_s']
        serial_name = name.replace('_batch', '_serial')
        timing['speedup_vs_same_backend_serial'] = timings[serial_name]['median_s']/timing['median_s']
    save_json(directory/'case.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', choices=['required', 'optional', 'off'], default='required')
    parser.add_argument('--gpu-index', type=int, default=0, help='Index within CUDA_VISIBLE_DEVICES')
    parser.add_argument('--fft-backend', choices=['fftw', 'numpy'], default='fftw',
        help='Native CPU FFT backend for serial/batch filtering and data preparation (default: fftw); JAX FFTs are unchanged')
    parser.add_argument('--approximants', nargs='+', choices=['IMRPhenomD', 'TaylorF2'], default=['IMRPhenomD', 'TaylorF2'])
    parser.add_argument('--bank-sizes', type=int, nargs='+', default=[10, 20])
    parser.add_argument('--repeats', type=int, default=11)
    parser.add_argument('--veto-samples', type=int, default=65)
    parser.add_argument('--seed', type=int, default=20261116)
    parser.add_argument('--sample-rate', type=int, default=2048)
    parser.add_argument('--f-lower', type=float, default=20)
    parser.add_argument('--injection-snr', type=float, default=20)
    parser.add_argument('--no-arrays', action='store_true')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'validation_results')
    args = parser.parse_args()
    if min(args.bank_sizes) < 1 or args.repeats < 1 or args.veto_samples < 2 or args.gpu_index < 0:
        parser.error('positive bank sizes/repeats, >=2 veto samples, and nonnegative GPU index required')
    if args.sample_rate < 512 or args.f_lower <= 0 or args.f_lower >= args.sample_rate/2 or args.injection_snr < 8:
        parser.error('invalid sample rate, f-lower, or injection SNR (minimum 8)')
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(args.output.resolve()/'.matplotlib'))
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    if args.gpu == 'off':
        os.environ['JAX_PLATFORMS'] = 'cpu'
    report = dict(arguments=vars(args).copy(), cases=[], failures=[],
        gpu_tested=False, status='STARTED', python=platform.python_version(),
        hardware=command_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']),
        git_commit=command_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']))
    report['arguments']['output'] = str(args.output)
    try:
        import jax
        from pycbc.scheme import CPUScheme
        from pycbc.waveform.jax_phenom import phenomd_implementation
        import plot_batch_vs_serial as fixture
        report['versions'] = {}
        for package in ['pycbc', 'jax', 'jaxlib', 'jax-cuda12-plugin', 'jax-cuda13-plugin', 'ripplegw', 'numpy', 'scipy']:
            try:
                report['versions'][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                report['versions'][package] = None
        devices = {'cpu': jax.devices('cpu')[0]}
        if args.gpu != 'off':
            try:
                gpus = jax.devices('gpu')
                devices['gpu'] = gpus[args.gpu_index]
            except (RuntimeError, IndexError) as exc:
                report['gpu_skip_reason'] = str(exc)
                if args.gpu == 'required':
                    raise RuntimeError('GPU required but unavailable. Install a CUDA-12 JAX build for P100; no CPU fallback is accepted.') from exc
        report['devices'] = {k: dict(name=str(d), kind=d.device_kind) for k, d in devices.items()}
        # Actually compile/execute FP32 FFT and FP64 arithmetic on every device.
        import jax.numpy as jnp
        for device in devices.values():
            with jax.default_device(device):
                fft = jax.jit(lambda x: jnp.fft.ifft(x))(jnp.ones(32, dtype=jnp.complex64))
                assert_device(fft, device, 'FFT preflight'); fft.block_until_ready()
                with jax.enable_x64(True):
                    value = jax.jit(lambda x: jnp.sum(jnp.sin(x)))(jnp.ones(32, dtype=jnp.float64))
                    assert_device(value, device, 'FP64 preflight'); value.block_until_ready()
        if 'IMRPhenomD' in args.approximants and phenomd_implementation() != 'ripple':
            raise RuntimeError('ripplegw IMRPhenomD is required; install ripplegw==0.5.1')
        rows = fixture.draw_rows(max(args.bank_sizes), np.random.default_rng(args.seed))
        with CPUScheme(num_threads=1):
            report['native_fft_backend'] = select_native_fft_backend(args.fft_backend)
            print('Native CPU FFT backend: ' + report['native_fft_backend']['module'], flush=True)
            for model in args.approximants:
                for count in args.bank_sizes:
                    directory = args.output / model / ('n'+str(count))
                    directory.mkdir(parents=True, exist_ok=True)
                    bank, templates, readers, start, stop = build_fixture(args, model, rows[:count], directory)
                    for dataset, reader in readers.items():
                        output = directory/dataset
                        output.mkdir(exist_ok=True)
                        case = run_case(args, model, bank, templates, reader, start, stop, output, devices)
                        report['cases'].append(case)
                        report['failures'].extend(case['failures'])
                        print(f'{model} n={count} {dataset}: ' + ('PASS' if case['passed'] else 'FAIL'), flush=True)
                        for name, timing in case['timings'].items():
                            metric = case['metrics'][name]
                            print(f"  {name:18s} median={1000*timing['median_s']:8.3f} ms "
                                  f"max complex dSNR={metric['max_complex_snr_error']:.3e} "
                                  f"power/SG dchi2={metric['max_chisq_error']:.3e}/{metric['max_sg_chisq_error']:.3e} "
                                  f"native-batch speedup={timing['speedup_vs_native_batch']:.2f}x "
                                  f"peak dt={metric['max_peak_time_error_samples']:.0f} samples", flush=True)
                        save_json(args.output/'summary.json', report)
                        jax.clear_caches()  # Bound accumulation of compiled shapes across cases.
        report['gpu_tested'] = 'gpu' in devices
        report['status'] = 'FAIL' if report['failures'] else ('PASS_CPU_GPU' if report['gpu_tested'] else 'PASS_CPU_ONLY_GPU_NOT_TESTED')
        for kind in ['metrics', 'timings']:
            rows = []
            for case in report['cases']:
                for mode, values in case[kind].items():
                    rows.append(dict(model=case['model'], templates=case['templates'],
                        native_fft_backend=case['native_fft_backend'],
                        dataset=case['dataset'], mode=mode, **values))
            if rows:
                keys = sorted(set().union(*(r.keys() for r in rows)))
                with (args.output/(kind+'.csv')).open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=keys, lineterminator='\n')
                    writer.writeheader(); writer.writerows(rows)
        save_json(args.output/'summary.json', report)
        print(report['status'], args.output.resolve(), flush=True)
        return 1 if report['failures'] else 0
    except Exception as exc:
        report['status'] = 'ERROR'
        report['error'] = str(exc)
        report['traceback'] = traceback.format_exc()
        save_json(args.output/'summary.json', report)
        print(report['traceback'], file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
