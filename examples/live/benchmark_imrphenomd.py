#!/usr/bin/env python
"""Run real bin/pycbc_live MPI jobs on identical IMRPhenomD frames and bank.

No GraceDB uploads, coincidence background, or Bayestar are enabled. Wall time
includes MPI startup, conditioning, waveform preparation, compilation, filtering,
vetoes and output. Use validate_cpu_gpu.py --benchmark-components for isolated
stage timings and full eight-second SNR/veto surfaces.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def fixture(directory, count, rate, seed):
    import h5py
    import numpy as np
    from pycbc.frame import write_frame
    from pycbc.noise import noise_from_psd
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.waveform import get_fd_waveform
    from pycbc.types import TimeSeries
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = np.column_stack([rng.uniform(30, 60, count), rng.uniform(20, 30, count),
                            rng.uniform(-.5, .5, count), rng.uniform(-.5, .5, count)])
    with h5py.File(directory/'bank.hdf', 'w') as f:
        for i, key in enumerate(['mass1', 'mass2', 'spin1z', 'spin2z']):
            f[key] = rows[:, i]
        f['f_lower'] = np.full(count, 20.)
    gps, duration = 1272790000, max(128, 64+8*count)
    n = duration*rate
    psd = aLIGOZeroDetHighPower(n//2+1, 1/duration, 20)
    signal = np.zeros(n//2+1, dtype=np.complex128)
    injections = []
    for tid, row in enumerate(rows):
        hp, _ = get_fd_waveform(approximant='IMRPhenomD', mass1=row[0], mass2=row[1],
            spin1z=row[2], spin2z=row[3], distance=100, f_lower=20,
            delta_f=1/duration, f_final=rate/2)
        hp.resize(n//2+1)
        h = np.asarray(hp).copy(); h[-1] = 0
        good = np.asarray(psd) > 0
        sigmasq = 4/duration*np.sum(np.abs(h[good])**2/np.asarray(psd)[good])
        tc = 40+8*tid
        # Loud injections allow a recovery gate after live PSD estimation.
        signal += h*(40/np.sqrt(sigmasq))*np.exp(-2j*np.pi*np.arange(len(h))*tc/duration)
        injections.append(dict(template_id=tid, gps=gps+tc, optimal_snr=40))
    signal_td = np.fft.irfft(signal, n=n)*rate
    for i, ifo in enumerate(['H1', 'L1']):
        noise = noise_from_psd(n, 1/rate, psd, seed=seed+i)
        strain = TimeSeries(np.asarray(noise)+signal_td, delta_t=1/rate, epoch=gps)
        write_frame(str(directory/f'{ifo}-JAX-{gps}-{duration}.gwf'),
                    f'{ifo}:SIMULATED_STRAIN', strain)
    meta = dict(gps=gps, duration=duration, sample_rate=rate, templates=count,
                injections=injections, seed=seed)
    (directory/'fixture.json').write_text(json.dumps(meta, indent=2)+'\n')
    return meta


def live_args(directory, output, meta):
    gps, rate = meta['gps'], meta['sample_rate']
    return ['--bank-file', str(directory/'bank.hdf'), '--approximant', 'IMRPhenomD',
        '--sample-rate', str(rate), '--analysis-chunk', '8', '--increment', '8',
        '--low-frequency-cutoff', '20', '--chisq-bins', '8', '--snr-threshold', '0',
        '--newsnr-threshold', '0', '--max-length', '64', '--pvalue-lookback-time', '8',
        '--psd-samples', '7', '--psd-segment-length', '4', '--psd-inverse-length', '2',
        '--trim-padding', '.25', '--highpass-frequency', '15',
        '--highpass-bandwidth', '5', '--highpass-reduction', '100',
        '--channel-name', 'H1:SIMULATED_STRAIN', 'L1:SIMULATED_STRAIN',
        '--frame-src', f'H1:{directory}/H1-*.gwf', f'L1:{directory}/L1-*.gwf',
        '--frame-read-timeout', '1', '--processing-scheme', 'cpu:1',
        '--fft-backends', 'fftw', '--fftw-measure-level', '0',
        '--sgchisq-snr-threshold', '0', '--sgchisq-locations', 'mtotal>0:10-0',
        '--sngl-ranking', 'newsnr_sgveto', '--ranking-statistic', 'quadsum',
        '--src-class-mchirp-to-delta', '.01', '--src-class-eff-to-lum-distance', '.74899',
        '--src-class-lum-distance-to-delta', '-.51557', '-.32195', '--ifar-double-followup-threshold', '1',
        '--pvalue-combination-livetime', '.0005', '--ifar-upload-threshold', '1',
        '--start-time', str(gps+32), '--end-time', str(gps+meta['duration']),
        '--output-path', str(output), '--store-psd', '--verbose']


def collect(directory):
    import h5py
    import numpy as np
    records = {}
    for path in sorted(directory.rglob('*.hdf')):
        with h5py.File(path, 'r') as f:
            for ifo in ['H1', 'L1']:
                if ifo not in f or 'template_id' not in f[ifo]:
                    continue
                g = f[ifo]
                for i, tid in enumerate(g['template_id'][:]):
                    # File identifies the analysis block; template id identifies
                    # bank row. No nearest-SNR pairing or dropped unmatched rows.
                    key = (str(path.relative_to(directory)), ifo, int(tid))
                    if key in records:
                        raise RuntimeError(f'Duplicate trigger key {key}')
                    records[key] = {k: float(g[k][i]) for k in
                        ['snr', 'chisq', 'sg_chisq', 'end_time', 'coa_phase']}
    if not records:
        raise RuntimeError(f'No live trigger products in {directory}')
    return records


def compare(reference, actual, rate, directory, label):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys = sorted(reference.keys() & actual.keys())
    report = dict(missing=len(reference.keys()-actual.keys()),
                  extra=len(actual.keys()-reference.keys()), matched=len(keys), metrics={})
    dt = np.array([(actual[k]['end_time']-reference[k]['end_time'])*rate for k in keys])
    same_time = np.abs(dt) < .1
    report['same_peak_time_rows'] = int(same_time.sum())
    report['different_peak_time_rows'] = int((~same_time).sum())
    csv_rows = [dict(block=k[0], detector=k[1], template_id=k[2], peak_dt_samples=float(d))
                for k,d in zip(keys, dt)]
    fig, axes = plt.subplots(5, 2, figsize=(14, 14))
    for row, name in enumerate(['snr', 'chisq', 'sg_chisq', 'end_time', 'coa_phase']):
        a = np.array([reference[k][name] for k in keys]); b = np.array([actual[k][name] for k in keys])
        diff = b-a
        if name == 'coa_phase':
            diff = np.angle(np.exp(1j*diff))
        tolerance = (1.01/rate if name == 'end_time' else .001 if name == 'coa_phase' else .002 + .0002*np.abs(a))
        passed = bool(len(keys) and np.all(np.isfinite(diff)) and np.all(np.abs(diff) <= tolerance))
        report['metrics'][name] = dict(max_absolute_error=float(np.max(np.abs(diff))) if len(keys) else None,
            max_same_peak_time_error=float(np.max(np.abs(diff[same_time]))) if same_time.any() else None,
            max_shifted_peak_time_error=float(np.max(np.abs(diff[~same_time]))) if (~same_time).any() else None,
            passed=passed)
        for item, av, bv, dv in zip(csv_rows, a, b, diff):
            item.update({name+'_native':float(av), name+'_'+label:float(bv), name+'_difference':float(dv)})
        axes[row,0].plot(a, '.', label='native'); axes[row,0].plot(b, '.', alpha=.6, label=label)
        axes[row,1].plot(diff, '.'); axes[row,0].set_ylabel(name); axes[row,1].set_ylabel('Difference')
    axes[0,0].legend()
    for ax in axes.flat:
        ax.set_xlabel('Analysis block / detector / template row'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(directory/(label+'_live_comparison.png'), dpi=140); plt.close(fig)
    if csv_rows:
        import csv
        with (directory/(label+'_live_rows.csv')).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(csv_rows)
    report['passed'] = not report['missing'] and not report['extra'] and all(x['passed'] for x in report['metrics'].values())
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--templates', type=int, default=10)
    p.add_argument('--workers', type=int, default=1, help='Filtering ranks; one additional CPU coordinator')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--sample-rate', type=int, default=2048)
    p.add_argument('--seed', type=int, default=20261117)
    p.add_argument('--gpu', choices=['required', 'off'], default='required')
    p.add_argument('--autotune', action='store_true')
    p.add_argument('--memory-budget-mb', type=float)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    if min(args.templates, args.workers, args.repeats) < 1 or args.templates < args.workers:
        p.error('positive counts and at least one template per filtering rank required')
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Choose a fresh or empty --output directory; existing fixtures/runs are preserved')
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=str(ROOT)+os.pathsep+env.get('PYTHONPATH',''),
        XLA_PYTHON_CLIENT_PREALLOCATE='false')
    fixture_dir = args.output/'fixture'
    meta = fixture(fixture_dir, args.templates, args.sample_rate, args.seed)
    if args.prepare_only:
        return 0
    modes = [('native', ['--filter-backend', 'cpu']),
             ('jax_cpu', ['--filter-backend', 'jax', '--jax-platform', 'cpu'])]
    if args.gpu == 'required':
        # MPI binding will reject absent/insufficient GPUs; never silently skip.
        modes.append(('jax_gpu', ['--filter-backend', 'jax', '--jax-platform', 'gpu']))
    report = dict(scope='real pycbc_live wall time including setup and MPI; background/uploads disabled',
                  gpu_tested=False, fixture=meta, runs=[], comparisons={}, failures=[])
    try:
        reference = None
        for label, flags in modes:
            for rep in range(args.repeats):
                out = args.output/f'{label}_{rep}'
                if out.exists():
                    raise RuntimeError(f'Refusing to overwrite existing run {out}; choose a fresh --output')
                command = ['mpiexec', '-n', str(args.workers+1), sys.executable,
                    '-m', 'mpi4py', str(ROOT/'bin/pycbc_live'), *live_args(fixture_dir, out, meta), *flags]
                if label.startswith('jax') and args.autotune:
                    command += ['--jax-autotune']
                    if args.memory_budget_mb:
                        command += ['--jax-memory-budget-mb', str(args.memory_budget_mb)]
                started = time.perf_counter()
                with (args.output/f'{label}_{rep}.log').open('w') as log:
                    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                wall = time.perf_counter()-started
                report['runs'].append(dict(mode=label, repeat=rep, wall_s=wall, command=command,
                                           returncode=result.returncode))
                if result.returncode:
                    raise RuntimeError(f'{label} failed; see {label}_{rep}.log')
                if label.startswith('jax'):
                    import re
                    text = (args.output/f'{label}_{rep}.log').read_text()
                    summaries = re.findall(r'JAX waveform check summary: (\d+)/(\d+) templates use the JAX generator \(phenomd=(\w+), platform=(\w+)', text)
                    report['runs'][-1]['waveform_checks'] = summaries
                    if len(summaries) != args.workers or any(
                            used != total or impl != 'ripple' or platform != label[4:]
                            for used, total, impl, platform in summaries):
                        raise RuntimeError('Missing ripplegw/device verification or waveform fallback in live log')
                products = collect(out)
                # Every injected template must be recovered near its own time.
                for injection in meta['injections']:
                    for ifo in ['H1', 'L1']:
                        candidates = [v['snr'] for k,v in products.items() if k[1] == ifo
                            and k[2] == injection['template_id'] and abs(v['end_time']-injection['gps']) <= .1]
                        if not candidates or max(candidates) < 10:
                            report['failures'].append(f'{label}/{rep}/{ifo}: injection {injection["template_id"]} not recovered with SNR >=10')
                if reference is None:
                    reference = products
                else:
                    comparison = compare(reference, products, args.sample_rate, args.output, f'{label}_{rep}')
                    report['comparisons'][f'{label}_{rep}'] = comparison
                    if not comparison['passed']:
                        report['failures'].append(f'{label}/{rep} trigger parity failed')
                if label == 'jax_gpu':
                    report['gpu_tested'] = True
                print(f'{label} repeat={rep} wall={wall:.3f}s triggers={len(products)}', flush=True)
        import numpy as np
        report['timings'] = {}
        baseline = np.median([r['wall_s'] for r in report['runs'] if r['mode'] == 'native'])
        for label, _ in modes:
            values = [r['wall_s'] for r in report['runs'] if r['mode'] == label]
            report['timings'][label] = dict(median_wall_s=float(np.median(values)),
                speedup_vs_native=float(baseline/np.median(values)), samples_s=values)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        names = list(report['timings'])
        ax.bar(names, [report['timings'][n]['median_wall_s'] for n in names])
        ax.set_ylabel('Median process wall time (s)')
        ax.set_title('Real pycbc_live: includes MPI, setup, tuning and output')
        fig.tight_layout(); fig.savefig(args.output/'live_runtime.png', dpi=140); plt.close(fig)
        report['status'] = 'FAIL' if report['failures'] else 'PASS'
    except Exception as exc:
        report['status'] = 'ERROR'; report['error'] = str(exc)
    (args.output/'live_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status', 'gpu_tested', 'failures']}))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
