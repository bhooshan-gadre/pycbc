#!/usr/bin/env python
"""Run native/JAX CPU/JAX GPU on the unchanged native live-example fixtures."""
import argparse
import glob
import hashlib
import json
import importlib.metadata
import platform
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def options(path=HERE/'live_example_options.sh'):
    result = subprocess.run(['bash', '-e', '-c', 'source "$1"; printf "%s\\0" "${live_example_args[@]}"',
                             'bash', str(path)], check=True, capture_output=True)
    return [s.decode() for s in result.stdout.split(b'\0') if s]


def values(args, flag):
    if flag not in args:
        return []
    i = args.index(flag)+1
    end = i
    while end < len(args) and not args[end].startswith('--'):
        end += 1
    return args[i:end]


def replace(args, flag, value=None):
    args = list(args)
    while flag in args:
        i = args.index(flag)
        # A quoted option value can itself start with -- (optimizer options).
        # These flags have known arity; do not mistake such a value for a flag.
        end = i+1+(0 if flag == '--run-snr-optimization' else 1)
        del args[i:end]
    if value is not None:
        args += [flag, str(value)]
    return args


def input_files(args, directory):
    paths = set()
    for flag in ['--bank-file', '--statistic-files', '--single-fit-file']:
        for name in values(args, flag):
            paths.add((directory/name).resolve())
    for entry in values(args, '--frame-src'):
        pattern = entry.split(':', 1)[1]
        found = glob.glob(str(directory/pattern))
        if not found:
            raise ValueError('No frames match '+entry)
        paths.update(Path(f).resolve() for f in found if Path(f).is_file())
    if not values(args, '--frame-src'):
        raise ValueError('Comparison requires fixed --frame-src files')
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise ValueError('Missing native fixtures: '+', '.join(missing))
    return sorted(paths)


def fingerprints(paths):
    rows = []
    for path in paths:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(4*2**20), b''):
                digest.update(block)
        rows.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest()))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--example-dir', type=Path, default=HERE,
                   help='Directory holding native template_bank.hdf, strain/ and statistic files')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--prepare', action='store_true', help='Generate missing native fixtures with run.sh, once')
    p.add_argument('--modes', nargs='+', choices=['native','jax_cpu','jax_gpu'],
                   default=['native','jax_cpu','jax_gpu'])
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--mpi-ranks', type=int, default=2, help='Same total ranks for all modes; native example uses 2')
    p.add_argument('--launcher', default='mpirun')
    p.add_argument('--launcher-arg', action='append', default=[], help='Repeat; use --launcher-arg=VALUE for dash-prefixed values')
    p.add_argument('--coordinator-gpu', action='store_true', help='Also reserve a GPU for rank-0 conditioning (condition mode)')
    p.add_argument('--allow-gpu-sharing', action='store_true', help='Explicitly permit coordinator/worker GPU sharing')
    p.add_argument('--profile-stages', action='store_true', help='Record per-rank stage timings with GPU synchronization')
    p.add_argument('--jax-preprocessing', choices=['off','whiten','condition'], default='off')
    p.add_argument('--waveform-model', choices=['native','IMRPhenomD'], default='native',
                   help='IMRPhenomD uses the same bank parameters/data with LAL native and compiled ripple JAX waveforms')
    p.add_argument('--jax-waveform-source',choices=['bank','ripple'],default=None)
    p.add_argument('--reference-waveforms',choices=['lal','ripple'],default='lal',
                   help='ripple uses identical-model CPU reference banks for FFTW/JAX; lal retains the cross-model comparison')
    p.add_argument('--autotune', action=argparse.BooleanOptionalAction, default=None,
                   help='Tune GPU batches by default; --autotune also tunes CPU, --no-autotune disables tuning')
    p.add_argument('--tuning-candidates', nargs='+', type=int, default=[1,2,4,8,16,32,64,128,256])
    p.add_argument('--tuning-repeats', type=int, default=5)
    p.add_argument('--memory-budget-mb', type=float)
    p.add_argument('--args-json', type=Path, help='Advanced: fixed native search argv JSON, instead of shared example options')
    p.add_argument('--plan-only', action='store_true', help='Validate fixtures and save exact commands without launching MPI')
    args = p.parse_args()
    if args.repeats < 1 or args.mpi_ranks < 2 or len(set(args.modes)) != len(args.modes) or 'native' not in args.modes:
        p.error('positive repeats, >=2 MPI ranks and unique modes including native required')
    if args.coordinator_gpu and args.jax_preprocessing != 'condition':
        p.error('--coordinator-gpu requires --jax-preprocessing condition')
    if min(args.tuning_candidates) < 1 or args.tuning_repeats < 1:
        p.error('Tuning candidates and repeats must be positive')
    out = args.output.resolve(); directory = args.example_dir.resolve()
    if out.exists() and any(out.iterdir()):
        p.error('Use a new or empty output directory')
    if out == directory or directory in out.parents and out.name in ['strain','output']:
        p.error('Use a separate comparison output directory')
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HDF5_USE_FILE_LOCKING='FALSE')
    env.setdefault('OMP_NUM_THREADS','4')
    env.setdefault('PYTHONHASHSEED','0')
    env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    env['PYTHONPATH'] = str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    if args.prepare:
        prep = dict(env, PYCBC_LIVE_PREPARE_ONLY='1')
        subprocess.run(['bash', str(HERE/'run.sh')], cwd=directory, env=prep, check=True)
    base = json.loads(args.args_json.read_text()) if args.args_json else options()
    if not isinstance(base,list) or not all(isinstance(x,str) for x in base):
        p.error('--args-json must contain an argv array of strings')
    for flag in ['--filter-backend','--jax-platform','--rank-backend','--latency-log']:
        if flag in base:
            p.error('Native options must not contain '+flag)
    # Async SNR optimization and Bayestar are followup stages, timed separately
    # by the original example. Exclude optimizer uniformly from this comparison.
    for flag in ['--run-snr-optimization','--snr-opt-extra-opts']:
        base = replace(base, flag)
    if args.waveform_model == 'IMRPhenomD':
        if '--approximant' not in base:p.error('Native options need --approximant')
        start=base.index('--approximant')+1
        base[start:start+len(values(base,'--approximant'))]=['IMRPhenomD']
    waveform_source=args.jax_waveform_source or ('ripple' if args.waveform_model == 'IMRPhenomD' or args.reference_waveforms == 'ripple' else 'bank')
    if waveform_source == 'ripple' and values(base,'--approximant') != ['IMRPhenomD']:
        p.error('ripple requires --waveform-model IMRPhenomD or IMRPhenomD in --args-json')
    if args.reference_waveforms == 'ripple' and values(base,'--approximant') != ['IMRPhenomD']:
        p.error('ripple reference banks require IMRPhenomD')
    try:
        paths = input_files(base, directory)
        before = fingerprints(paths)
    except ValueError as exc:
        p.error(str(exc)+'; use --prepare or point --example-dir at your native fixtures')
    out.mkdir(parents=True, exist_ok=True)
    versions = {}
    for package in ['pycbc','numpy','jax','jaxlib','mpi4py','ripplegw']:
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = None
    manifest = dict(versions=versions,python=platform.python_version(),host=platform.node(),
        cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES'),schema=1, example_dir=str(directory), native_options=base,
        native_options_source=str(args.args_json.resolve() if args.args_json else HERE/'live_example_options.sh'),
        inputs=before, input_integrity=None, runs=[], plan_only=args.plan_only,
        threads=env['OMP_NUM_THREADS'],pythonhashseed=env['PYTHONHASHSEED'],
        waveform_model=args.waveform_model,jax_waveform_source=waveform_source,
        reference_waveforms=args.reference_waveforms,
        scope='pycbc_live through background/coincidence/trigger output; async SNR optimization and Bayestar excluded')
    failed = False
    try:
        # Rotate ordering across repeats to reduce a fixed first-backend bias.
        for rep in range(args.repeats):
            modes = args.modes[rep % len(args.modes):]+args.modes[:rep % len(args.modes)]
            for mode in modes:
                run = out/f'{mode}_{rep}'; run.mkdir()
                argv = replace(base, '--output-path', run/'output')
                argv += ['--latency-log', str(run/'latency.jsonl'),
                         '--bank-waveform-source',args.reference_waveforms,
                         '--filter-backend','cpu' if mode == 'native' else 'jax']
                if args.profile_stages:
                    argv += ['--stage-profile-dir',str(run/'stages')]
                if mode != 'native':
                    argv += ['--jax-platform', mode[4:], '--jax-waveform-source',waveform_source,
                             '--jax-preprocessing',args.jax_preprocessing]
                    if mode == 'jax_gpu' and args.coordinator_gpu:
                        argv += ['--jax-coordinator-preprocessing']
                    if mode == 'jax_gpu' and args.allow_gpu_sharing:
                        argv += ['--jax-allow-gpu-sharing']
                    if args.autotune is True or (args.autotune is None and mode == 'jax_gpu'):
                        argv += ['--jax-autotune','--jax-tuning-candidates',
                                 *map(str,args.tuning_candidates),
                                 '--jax-tuning-repeats',str(args.tuning_repeats)]
                    if args.memory_budget_mb is not None:
                        argv += ['--jax-memory-budget-mb',str(args.memory_budget_mb)]
                command = [args.launcher,*args.launcher_arg,'-n',str(args.mpi_ranks),
                           sys.executable,'-m','mpi4py',str(ROOT/'bin/pycbc_live'),*argv]
                record = dict(mode=mode,repeat=rep,directory=str(run),command=command,
                              returncode=None,wall_s=None)
                manifest['runs'].append(record)
                if not args.plan_only:
                    run_env = dict(env)
                    if mode == 'jax_cpu':
                        run_env['JAX_PLATFORMS'] = 'cpu'
                    elif mode == 'jax_gpu':
                        run_env.pop('JAX_PLATFORMS',None)
                    started = time.perf_counter()
                    with (run/'search.log').open('w') as log:
                        result = subprocess.run(command,cwd=directory,env=run_env,
                                                stdout=log,stderr=subprocess.STDOUT)
                    record.update(returncode=result.returncode,wall_s=time.perf_counter()-started)
                    failed |= result.returncode != 0
                    print(f'{mode} repeat={rep} rc={result.returncode} wall={record["wall_s"]:.3f}s',flush=True)
                (out/'runs.json').write_text(json.dumps(manifest,indent=2)+'\n')
        try:
            manifest['input_integrity'] = before == fingerprints(input_files(base,directory))
        except Exception as exc:
            manifest['input_integrity'] = False
            manifest['input_integrity_error'] = str(exc)
        failed |= not manifest['input_integrity']
    finally:
        (out/'runs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if args.plan_only:
        return 0
    result = subprocess.run([sys.executable,str(HERE/'compare_backend_runs.py'),
                             '--run-dir',str(out)],env=env)
    return 1 if failed else result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
