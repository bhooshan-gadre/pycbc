"""Acceptance gates and portable smoke test for the P100 validation runner."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip('jax')

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'examples/search/jax/validate_cpu_gpu.py'
spec = importlib.util.spec_from_file_location('gpu_validation', SCRIPT)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


@pytest.mark.parametrize('corruption', ['surface', 'peak', 'phase', 'time', 'dof'])
def test_gates_reject_corrupted_results(corruption):
    reference = dict(snr=np.array([20.]), chisq=np.array([1.]),
        sg_chisq=np.array([1.]), coa_phase=np.array([0.]),
        end_time=np.array([5.]), chisq_dof=np.array([14]))
    actual = {k: v.copy() for k, v in reference.items()}
    rho_ref = np.ones((1, 32), dtype=np.complex64)
    rho = rho_ref.copy()
    if corruption == 'surface':
        rho[0, 0] += .1j
    else:
        field = dict(peak='snr', phase='coa_phase', time='end_time', dof='chisq_dof')[corruption]
        actual[field] += 1
    metrics, failures = validation.compare_products(reference, actual, rho_ref, rho, 2048)
    assert failures and not metrics['passed']


def test_curve_gate_rejects_veto_error_and_nonfinite():
    ref = dict(power=np.ones(3), sg=np.ones(3))
    for bad in [2., np.nan]:
        actual = dict(power=np.array([1., bad, 1.]), sg=np.ones(3))
        metrics, failures = validation.compare_curves(ref, actual)
        assert failures and not metrics['power_curve_passed']
        assert metrics['sg_curve_passed']


def run(tmp_path, *args):
    env = dict(os.environ, JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MPLCONFIGDIR=str(tmp_path/'mpl'))
    return subprocess.run([sys.executable, str(SCRIPT), '--output', str(tmp_path),
                           *args], cwd=ROOT, env=env, text=True,
                          capture_output=True, timeout=240)


def test_required_gpu_cannot_pass_on_cpu(tmp_path):
    result = run(tmp_path, '--gpu', 'required')
    assert result.returncode == 2, result.stdout + result.stderr
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert summary['status'] == 'ERROR'
    assert not summary['gpu_tested']
    assert 'GPU required but unavailable' in summary['error']


@pytest.mark.parametrize('backend', ['fftw', 'numpy'])
def test_cpu_smoke_outputs_eight_second_arrays_and_plots(tmp_path, backend):
    from pycbc.fft import backend_cpu
    if backend not in backend_cpu._alist:
        pytest.skip(f'{backend} libraries unavailable')
    result = run(tmp_path, '--gpu', 'off', '--approximants', 'TaylorF2',
                 '--bank-sizes', '2', '--repeats', '1', '--veto-samples', '3',
                 '--fft-backend', backend)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert summary['status'] == 'PASS_CPU_ONLY_GPU_NOT_TESTED'
    assert len(summary['cases']) == 2
    assert summary['native_fft_backend']['name'] == backend
    assert summary['native_fft_backend']['module'] == 'pycbc.fft.' + {'fftw': 'fftw', 'numpy': 'npfft'}[backend]
    assert 'native_fft_backend' in (tmp_path/'timings.csv').read_text().splitlines()[0]
    assert (tmp_path/'metrics.csv').is_file() and (tmp_path/'timings.csv').is_file()
    for case in summary['cases']:
        assert case['native_fft_backend'] == backend
        assert case['passed'] and case['valid_length'] == 8*2048
        for name, metadata in case['implementations'].items():
            assert name.startswith('jax_cpu_')
            assert metadata['filter_only']['passed']
            assert all(w['implementation'] == 'lal-taylorf2' for w in metadata['waveforms'])
        directory = tmp_path/'TaylorF2/n2'/case['dataset']
        for name in ['timeseries', 'peaks', 'surface', 'filter_residuals']:
            assert (directory/(name+'.png')).stat().st_size > 1000
        with np.load(directory/'arrays.npz') as arrays:
            assert arrays['jax_cpu_batch_rho'].shape == (2, 8*2048)
            assert arrays['time_s'][0] == 0
            assert arrays['time_s'][-1] == 8-1/2048


def test_unavailable_fftw_is_an_error_not_a_numpy_fallback(monkeypatch):
    from pycbc.fft import backend_cpu
    monkeypatch.setattr(backend_cpu, '_alist', ['numpy'])
    with pytest.raises(RuntimeError, match='FFTW.*install both single- and double-precision'):
        validation.select_native_fft_backend('fftw')
