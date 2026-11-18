"""Regress candidate selection, FFT support, and block caching against PyCBC."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('jax')

from pycbc.filter import LiveBatchMatchedFilter
from pycbc.filter.jax_matchedfilter import JaxLiveBatchMatchedFilter
from pycbc.scheme import CPUScheme
from pycbc.types import Array, FrequencySeries
from pycbc.vetoes.chisq import power_chisq_at_points_from_precomputed
from pycbc.vetoes.jax_chisq import (
    power_chisq_batch, power_chisq_selected, power_chisq_candidate_points)
from pycbc.vetoes.sgchisq import SingleDetSGChisq
from pycbc.waveform.bank import LiveFilterBank

spec = importlib.util.spec_from_file_location(
    'live_parity_fixture', Path(__file__).resolve().parents[1]
    / 'examples/live/compare_jax_live.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


@pytest.fixture
def bank_and_reader(tmp_path):
    path = tmp_path / 'bank.hdf'
    fixture.write_bank(path, [(30, 30), (35, 29), (28, 22), (40, 15)],
                       [(0.1, 0), (0.2, -0.1), (0, 0.15), (-0.2, 0.05)], 20)
    bank = LiveFilterBank(str(path), 2048, minimum_buffer=8,
                          approximant='SPAtmplt', increment=8,
                          low_frequency_cutoff=20)
    reader, _ = fixture.make_reader(bank, 0, 2048)
    with CPUScheme(num_threads=1):
        yield bank, reader


def engines(bank, **overrides):
    options = dict(snr_threshold=4, chisq_bins='4')
    options.update(overrides)
    def sg():
        return SingleDetSGChisq(bank, num_bins='4', snr_threshold=4,
                               chisq_locations=['mtotal>0:10-0'])
    return (LiveBatchMatchedFilter(list(bank), sg_chisq=sg(), **options),
            JaxLiveBatchMatchedFilter(list(bank), sg_chisq=sg(), **options))


@pytest.mark.parametrize('limits', [dict(), dict(max_triggers_in_batch=1),
    dict(newsnr_threshold=5), dict(snr_abort_threshold=1),
    dict(snr_threshold=100)])
def test_live_limits_and_disabled_veto(bank_and_reader, limits):
    bank, reader = bank_and_reader
    cpu, jax = engines(bank, **limits)
    expected, actual = cpu.process_data(reader), jax.process_data(reader)
    if expected is False:
        assert actual is False
    elif not len(expected['snr']):
        assert len(actual['snr']) == 0
    else:
        fixture.assert_parity(expected, actual, 'live limits')


def test_jax_disabled_veto(bank_and_reader):
    bank, reader = bank_and_reader
    _cpu, engine = engines(bank, chisq_bins='0')
    engine.sg_chisq.do = False
    actual = engine.process_data(reader)
    assert len(actual['snr']) > 0
    np.testing.assert_array_equal(actual['chisq'], 0)
    np.testing.assert_array_equal(actual['chisq_dof'], 1)
    np.testing.assert_array_equal(actual['sg_chisq'], 0)


@pytest.mark.parametrize('threshold',[4,100])
def test_deferred_bank_download_matches_serial_batches(bank_and_reader, monkeypatch, threshold):
    import jax
    bank, reader = bank_and_reader
    _, engine = engines(bank, snr_threshold=threshold)
    from pycbc.filter.jax_tuning import install_layout
    install_layout(engine, 1)
    expected = engine.process_data(reader)
    engine.set_data(reader)
    calls = []
    original = jax.device_get
    def observe(value):
        calls.append(value)
        return original(value)
    monkeypatch.setattr(jax, 'device_get', observe)
    batches = list(engine._collect_device_batches())
    assert len(engine.tgroups) == 4
    assert len(calls) == 1 and len(calls[0]) == 4
    results = engine.combine_results([r for r,_ in batches])
    links = [link for _,batch_links in batches for link in batch_links]
    actual = engine._process_vetoes(results, links)
    for key in expected:
        np.testing.assert_array_equal(actual[key],expected[key])


def test_cpu_surface_is_lazy_and_window_refreshes(bank_and_reader):
    bank, reader = bank_and_reader
    _cpu, engine = engines(bank)
    engine.process_data(reader)
    for state in engine._states:
        assert state._snr is None
        assert state.unnorm is not None
        start = 1024
        window = np.asarray(state.snr_window(start, 4096))
        assert state._snr is None
        full = np.asarray(state.snr)
        assert state.unnorm is None
        np.testing.assert_array_equal(window, full[:, start:start + 4096])
        np.testing.assert_array_equal(
            np.asarray(state.snr_window(start, 4096)), window)
        assert state.snr is state.snr
        with pytest.raises(ValueError):
            state.snr_window(-1, 4096)
    engine.process_data(reader)
    assert all(state._snr is None for state in engine._states)


@pytest.mark.parametrize('batch', [1, 2, 5])
def test_split_surface_matches_fused_for_partial_windows(batch):
    import jax
    import jax.numpy as jnp
    from pycbc.filter.jax_matchedfilter import (
        _cpu_surface, _surface_kernel, _normalize_surface_kernel)
    rng = np.random.default_rng(44)
    ntime, length, start = 256, 37, np.int32(83)
    h = jnp.asarray((rng.normal(size=(batch, 129))
                     + 1j * rng.normal(size=(batch, 129))).astype(np.complex64))
    strain = h[0]
    norms = jnp.asarray(rng.uniform(.01, 2, batch).astype(np.float32))
    q = jnp.zeros((batch, ntime), dtype=jnp.complex64)
    surface = jnp.zeros_like(q)
    expected = jax.device_get(_surface_kernel(length)(h, strain, norms, q, surface, start))
    product, raw, *peaks = _cpu_surface(h, strain, norms, start, length)
    actual = jax.device_get((product, _normalize_surface_kernel()(raw, norms), *peaks))
    np.testing.assert_array_equal(actual[4], expected[4])
    for i in [0, 1, 2, 3, 5]:
        np.testing.assert_allclose(actual[i], expected[i], rtol=1e-5, atol=1e-5)


def test_raw_peak_keeps_first_equal_maximum_in_valid_window():
    from pycbc.filter.jax_matchedfilter import _peak_kernel
    unnorm = np.zeros((2, 64), dtype=np.complex64)
    unnorm[:, 0] = 100  # Outside the valid window.
    unnorm[0, [8, 10]] = 5 + 3j
    unnorm[1, [20, 25]] = 2 - 4j
    norms = np.array([.25, 2], dtype=np.float32)
    peak, raw, indices, magnitude = _peak_kernel(37)(unnorm, norms, np.int32(5))
    np.testing.assert_array_equal(indices, [8, 20])
    np.testing.assert_array_equal(raw, [5 + 3j, 2 - 4j])
    np.testing.assert_allclose(peak, np.asarray(raw) * norms)
    np.testing.assert_allclose(magnitude, np.abs(peak))


def test_vetoes_only_process_retained_and_activated_rows(bank_and_reader, monkeypatch):
    import pycbc.filter.jax_matchedfilter as module
    bank, reader = bank_and_reader
    cpu, engine = engines(bank, max_triggers_in_batch=1)
    seen = []
    original = module._cpu_point_chisq
    def observe(q, snr, norms, bins, shifts, num_bins, rows, support):
        seen.append((len(rows), support, q.shape))
        return original(q, snr, norms, bins, shifts, num_bins, rows, support)
    monkeypatch.setattr(module, '_cpu_point_chisq', observe)
    # Force all SG candidates below its activation threshold: no SG kernel call.
    engine.sg_chisq.snr_threshold = cpu.sg_chisq.snr_threshold = 100
    def unexpected(*args, **kwargs):
        raise AssertionError('Inactive SG candidates must not execute the tile kernel')
    monkeypatch.setattr(engine, '_sg_for_group', unexpected)
    fixture.assert_parity(cpu.process_data(reader), engine.process_data(reader), 'selected veto')
    assert seen and all(n == 1 and support <= shape[1] // 2 + 1
                        for n, support, shape in seen)


def test_serial_chunks_reuse_strain_but_refresh_each_block(bank_and_reader):
    bank, reader = bank_and_reader
    ntime = (len(bank[0]) - 1) * 2
    cpu, engine = engines(bank, maxelements=ntime)
    assert len(engine._states) > 1
    original = reader.overwhitened_data
    calls = []
    def counted(df):
        calls.append(df)
        return original(df)
    reader.overwhitened_data = counted
    for iteration in range(3):
        if iteration:
            old = reader._stilde
            reader._stilde = FrequencySeries(
                np.asarray(old) * np.complex64(0.9), delta_f=old.delta_f,
                epoch=old.epoch)
            reader._stilde.psd = old.psd if iteration == 1 else FrequencySeries(
                np.asarray(old.psd) * np.float32(1.3), delta_f=old.delta_f,
                epoch=old.psd.epoch)
        expected = cpu.process_data(reader)
        calls.clear()
        actual = engine.process_data(reader)
        assert len(calls) == len(set(calls)) == 1
        fixture.assert_parity(expected, actual, 'fresh block and PSD')


def test_three_detector_psd_weights_are_reused(bank_and_reader):
    bank,reader=bank_and_reader
    _,engine=engines(bank)
    original=reader._stilde.psd
    expected=engine.process_data(reader)
    norms=[s.norms for s in engine._states]
    for factor in [1.1,1.3]:
        reader._psd=original*factor
        engine.process_data(reader)
    reader._psd=original
    actual=engine.process_data(reader)
    assert all(s.norms is n for s,n in zip(engine._states,norms))
    assert all(len(s._psd_cache)==3 for s in engine._states)
    fixture.assert_parity(expected,actual,'three detector PSD cache')


def test_long_bin_veto_matches_native_rotation_arithmetic():
    rng=np.random.default_rng(681)
    ntime=262144
    q=(rng.normal(size=(2,ntime))+1j*rng.normal(size=(2,ntime))).astype('complex64')*.01
    q[:,ntime//2+1:]=0
    bins=np.array([1000,5000,18000,60000,120000],dtype=np.int32)
    shifts=np.array([127999,188001],dtype=np.float32)
    snr=np.array([.5+.2j,-.3+.4j],dtype=np.complex64)
    norms=np.array([.3,.2],dtype=np.float32)
    actual=np.asarray(power_chisq_batch(q,snr,norms,bins,shifts,np.full(2,4)))
    with CPUScheme(num_threads=1):
        expected=np.array([power_chisq_at_points_from_precomputed(
            Array(q[i]),snr[i:i+1],float(norms[i]),bins,[shifts[i]])[0] for i in range(2)])
    np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=1e-5)


def test_sg_shift_matches_native_at_late_times():
    from pycbc.vetoes.jax_chisq import sg_chisq_batch
    from pycbc.waveform.utils import apply_fseries_time_shift
    rng=np.random.default_rng(92)
    data=(rng.normal(size=65537)+1j*rng.normal(size=65537)).astype('complex64')
    indices=np.stack([np.arange(10037,10837),np.arange(20481,21281)])[:,None,:].astype('int32')
    tiles=(rng.normal(size=indices.shape)+1j*rng.normal(size=indices.shape)).astype('complex64')
    times=np.array([31.96728515625,63.99951171875],dtype='float32')
    gsigma=np.full((2,1),10.,dtype='float32');df=1/64
    actual=np.asarray(sg_chisq_batch(data,tiles,indices,gsigma,times,df))
    expected=[]
    with CPUScheme(num_threads=1):
        spectrum=FrequencySeries(data,delta_f=df)
        for row,t in enumerate(times):
            shifted=np.asarray(apply_fseries_time_shift(spectrum,-float(t)))
            z=(tiles[row,0]*shifted[indices[row,0]]).sum()*(4*df/10.)
            expected.append(abs(z)**2/2)
    np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=1e-6)


def test_truncated_selected_chisq_keeps_original_fft_phase_length():
    rng = np.random.default_rng(20261116)
    ntime = 4096
    q = (rng.normal(size=(3, ntime)) + 1j * rng.normal(size=(3, ntime))).astype(np.complex64)
    q[:, ntime // 2 + 1:] = 0
    bins = np.array([[8, 40, 100, 400, 800, 1500],
                     [8, 40, 100, 400, 400, 400],
                     [8, 80, 200, 700, 1000, 1500]], dtype=np.int32)
    counts = np.array([5, 3, 5], dtype=np.float32)
    shifts = np.array([10, 100, 2000], dtype=np.float32)
    snr = np.array([1 + 1j, 2 - 1j, 3j], dtype=np.complex64)
    norms = np.full(3, 0.013, dtype=np.float32)
    full = np.asarray(power_chisq_batch(q, snr, norms, bins, shifts, counts))
    truncated = np.asarray(power_chisq_batch(
        q[:, :1500], snr, norms, bins, shifts, counts, ntime=ntime))
    np.testing.assert_allclose(truncated, full, rtol=1e-6)
    selected = np.array([2, 0, 2, 2], dtype=np.int32)
    got = np.asarray(power_chisq_selected(
        q, snr, norms, bins, shifts, counts, selected, 1500))
    np.testing.assert_allclose(got, full[selected], rtol=1e-6)
    # Two retained peaks for the same template must keep distinct time shifts.
    point_indices = np.array([2000, 10, 500, 800], dtype=np.int32)
    peaks = np.array([1j, 2, 1 + 1j, 2j], dtype=np.complex64)
    expected = np.asarray(power_chisq_batch(
        q[selected], peaks / norms[selected], norms[selected], bins[selected],
        point_indices, counts[selected]))
    got = np.asarray(power_chisq_candidate_points(
        q, norms, bins, counts, selected, point_indices, peaks, 1500))
    np.testing.assert_allclose(got, expected, rtol=1e-6)
    with CPUScheme(num_threads=1):
        for i in range(3):
            edges = bins[i, :int(counts[i]) + 1].astype(np.uint32)
            expected = power_chisq_at_points_from_precomputed(
                Array(q[i]), snr[i:i+1], float(norms[i]), edges,
                shifts[i:i+1].astype(np.float64))
            np.testing.assert_allclose(full[i], np.asarray(expected)[0], rtol=1e-4)


def test_live_and_offline_example(tmp_path):
    import os
    import subprocess
    import sys
    script = Path(__file__).resolve().parents[1] / 'examples/search/jax/compare_searches.py'
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    subprocess.run([sys.executable, str(script), '--bank-sizes', '4',
                    '--repeats', '1', '--output', str(tmp_path / 'searches.json')],
                   check=True, env=env, timeout=180)
