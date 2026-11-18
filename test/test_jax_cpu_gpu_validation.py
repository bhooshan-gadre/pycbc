"""Acceptance gates and portable smoke test for the P100 validation runner."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

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


def test_compiled_ripple_identical_model_reference(tmp_path):
    pytest.importorskip('ripplegw')
    result=run(tmp_path,'--gpu','off','--approximants','IMRPhenomD',
        '--reference-waveforms','ripple','--bank-sizes','2','--repeats','1','--veto-samples','3')
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads((tmp_path/'summary.json').read_text())
    assert report['status']=='PASS_CPU_ONLY_GPU_NOT_TESTED'
    for case in report['cases']:
        assert case['reference_waveforms']=='ripple'
        assert case['passed']
        for name,metadata in case['implementations'].items():
            assert metadata['filter_only']['passed']
            assert all(w['implementation']=='ripple-batched' for w in metadata['waveforms'])
            assert case['metrics'][name]['cross_model_veto_curves_gated']
    assert (tmp_path/'IMRPhenomD/n2/reference_waveforms.json').is_file()


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
    assert len(summary['cases']) == 3
    injections = [case for case in summary['cases'] if case['injection']]
    assert {case['injection']['template_id'] for case in injections} == {0, 1}
    for case in injections:
        assert case['representative_template_id'] == case['injection']['template_id']
        assert all(t['engine_reused'] for t in case['timings'].values())
        for check in case['injection_checks'].values():
            assert check['passed']
            assert check['snr_at_injection_time'] >= 10
            assert check['recovered_peak_snr'] >= 10
    assert 'N/A (reference compared with itself)' in result.stdout
    assert 'exactly_equal=True' in result.stdout
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


def test_zero_residual_is_exact_equality_of_independent_arrays():
    result = dict(snr=np.array([1.]), chisq=np.array([1.]),
        sg_chisq=np.array([1.]), coa_phase=np.array([0.]),
        end_time=np.array([5.]), chisq_dof=np.array([14]))
    reference = np.ones((1, 32), dtype=np.complex128)
    actual = reference.copy()
    metrics, errors = validation.compare_products(result, result, reference, actual, 2048)
    assert not errors
    assert metrics['max_complex_snr_error'] == 0
    assert metrics['surface_exactly_equal']
    assert metrics['unequal_complex_snr_samples'] == 0
    assert not metrics['surface_shares_memory_with_reference']
    actual[0, 1] += 1e-10j
    metrics, errors = validation.compare_products(result, result, reference, actual, 2048)
    assert not errors
    assert not metrics['surface_exactly_equal']
    assert metrics['unequal_complex_snr_samples'] == 1
    assert metrics['max_imag_snr_error'] == pytest.approx(1e-10)
    assert metrics['max_real_snr_error'] == 0


@pytest.mark.parametrize('bad_snr', [9.99, np.nan])
def test_injection_gate_checks_injected_id_and_every_mode(bad_snr):
    # Row 0 is deliberately loud: it must not mask a weak injected row 1.
    good = dict(result=dict(snr=np.array([20., 15.])),
                rho=np.array([[20., 20.], [15., 15.]], dtype=np.complex64))
    bad = dict(result=dict(snr=np.array([20., bad_snr])),
               rho=np.array([[20., 20.], [bad_snr, bad_snr]], dtype=np.complex64))
    checks, failures = validation.check_injection(
        dict(native_serial=good, jax_gpu_batch=bad), [42, 7],
        dict(template_id=7, valid_sample_index=1))
    assert checks['native_serial']['passed']
    assert not checks['jax_gpu_batch']['passed']
    assert len(failures) == 1 and 'jax_gpu_batch' in failures[0]


def test_minimum_requested_snr_is_boosted_and_recovered(tmp_path):
    result = run(tmp_path, '--gpu', 'off', '--approximants', 'TaylorF2',
                 '--bank-sizes', '1', '--repeats', '1', '--veto-samples', '3',
                 '--injection-snr', '10', '--fft-backend', 'numpy', '--no-arrays')
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((tmp_path/'summary.json').read_text())
    case = summary['cases'][1]
    assert case['injection']['requested_optimal_snr'] == 10
    assert case['injection']['amplitude_boosted']
    assert case['injection']['optimal_snr_used'] >= 11
    assert all(c['passed'] for c in case['injection_checks'].values())


def test_reject_requested_snr_below_ten(tmp_path):
    result = run(tmp_path, '--injection-snr', '9')
    assert result.returncode == 2 and 'minimum 10' in result.stderr


def test_adjacent_peak_vetoes_are_compared_at_the_same_sample():
    reference = dict(snr=np.array([1.]), chisq=np.array([1.]),
        sg_chisq=np.array([1.]), coa_phase=np.array([.2]),
        end_time=np.array([4/2048]), chisq_dof=np.array([14]))
    actual = {k: v.copy() for k, v in reference.items()}
    actual['end_time'][0] = 5/2048
    actual['coa_phase'][0] = .4
    actual['chisq'][0] = 2
    actual['sg_chisq'][0] = 3
    rho = np.ones((1, 32), dtype=np.complex128)
    rho[0, 4:6] = np.exp(1j*np.array([.2, .4]))
    called = []
    def evaluator(row, index):
        called.append((row, index))
        return 1., 1.
    aligned, changed = validation.align_peak_vetoes(
        dict(result=actual, rho=rho), reference, [7],
        SimpleNamespace(start_time=0, sample_rate=2048), 100, evaluator)
    assert changed == [7] and called == [(0, 104)]
    metrics, failures = validation.compare_products(reference, actual,
        rho.copy(), rho, 2048, actual_at_reference_peak=aligned)
    assert not failures
    assert metrics['max_chisq_error'] == 1
    assert metrics['max_same_time_chisq_error'] == 0
    assert metrics['max_peak_phase_error_rad'] == pytest.approx(.2)
    assert metrics['max_same_time_phase_error_rad'] < 1e-12
    assert metrics['max_peak_time_error_samples'] == 1
    assert actual['chisq'][0] == 2  # Original peak outputs are retained.


@pytest.mark.parametrize('field', ['chisq', 'coa_phase'])
def test_same_time_comparison_cannot_hide_nonfinite_original_peaks(field):
    reference = dict(snr=np.array([1.]), chisq=np.array([1.]),
        sg_chisq=np.array([1.]), coa_phase=np.array([0.]),
        end_time=np.array([5.]), chisq_dof=np.array([14]))
    actual = {key: value.copy() for key, value in reference.items()}
    actual[field][0] = np.nan
    rho = np.ones((1, 32), dtype=np.complex64)
    metrics, errors = validation.compare_products(reference, actual,
        rho.copy(), rho, 2048, actual_at_reference_peak=reference)
    assert not metrics['passed'] and any('nonfinite original' in error for error in errors)
