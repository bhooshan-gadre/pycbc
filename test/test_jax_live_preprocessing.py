"""Native FFTW versus device preprocessing and resident-buffer parity."""
from types import SimpleNamespace
import os
import numpy as np
import pytest

jax=pytest.importorskip('jax')
from pycbc.scheme import CPUScheme
from pycbc.types import TimeSeries,FrequencySeries
from pycbc.strain import StrainBuffer
from pycbc.strain.jax_live import JaxLiveConditioner


def device():
    return jax.devices(os.environ.get('PYCBC_JAX_TEST_PLATFORM','cpu'))[0]


@pytest.fixture(autouse=True)
def cpu_scheme():
    from pycbc.fft import backend_cpu
    with CPUScheme(num_threads=1), jax.default_device(device()):
        backend_cpu.set_backend(['fftw'])
        yield


def reader():
    r=object.__new__(StrainBuffer)
    r.sample_rate=512;r.corruption=16;r.trim_padding=32;r.reduced_pad=256
    r.psd_inverse_length=.25;r.low_frequency_cutoff=20;r.gate_params=[]
    r.strain=TimeSeries(np.random.default_rng(3).normal(size=8192).astype('float32'),
                        delta_t=1/512,epoch=1272790000)
    freq=np.arange(4097)/16
    r.psd=FrequencySeries((1+freq/256).astype('float32'),delta_f=1/16)
    r.psds={};r.segments={}
    return r


@pytest.mark.parametrize('mode',['whiten','condition'])
def test_whitening_and_incremental_ring_match_native(mode):
    native=reader(); device_reader=reader()
    backend=JaxLiveConditioner(device_reader,device(),mode)
    for step in range(3):
        if step:
            for r in [native,device_reader]:
                r.strain.roll(-512)
                r.strain.numpy()[-528:]=np.random.default_rng(step).normal(size=528).astype('float32')
                r.strain.start_time += 1
                r.segments={}
        expected=native.overwhitened_data(1/8)
        metadata,actual=backend.overwhitened(1/8)
        np.testing.assert_allclose(np.asarray(actual),np.asarray(expected),rtol=3e-5,atol=3e-6)
        # Stopband PSD reciprocals amplify insignificant FFT roundoff. Check
        # the inverse weights everywhere and the PSD itself in the search band.
        np.testing.assert_allclose(1/np.asarray(metadata.psd),1/np.asarray(expected.psd),rtol=3e-5,atol=1e-6)
        cut=int(20/metadata.delta_f)
        np.testing.assert_allclose(np.asarray(metadata.psd)[cut:],np.asarray(expected.psd)[cut:],rtol=1e-5,atol=1e-6)
        assert metadata.epoch==expected.epoch and metadata.delta_f==expected.delta_f
    assert backend.buffer.devices()=={device()}
    assert backend.metrics['full_buffer_uploads']==1
    assert backend.metrics['incremental_uploads']==2
    assert backend.metrics['uploaded_bytes']==(8192+528*2)*4


@pytest.mark.parametrize('factor',[1,8])
def test_fir_and_ldas_resampling_match_native(factor):
    import pycbc.filter
    r=SimpleNamespace(sample_rate=512,factor=factor,highpass_samples=32,
                      highpass_frequency=15,beta=5.,dyn_range_fac=1e20)
    x=TimeSeries(np.random.default_rng(4).normal(size=32768)*1e-20,
                 delta_t=1/(512*factor),epoch=1272790000)
    expected=pycbc.filter.highpass_fir(x,15,32,beta=5.)
    expected=(expected*1e20).astype(np.float32)
    expected=pycbc.filter.resample_to_delta_t(expected,1/512,method='ldas')
    backend=JaxLiveConditioner(r,device(),'condition')
    actual=backend.condition_raw(x)
    np.testing.assert_allclose(np.asarray(actual),np.asarray(expected),rtol=3e-5,atol=3e-6)
    assert actual.start_time==x.start_time


def test_batched_welch_and_glitch_detection_match_native():
    import pycbc.psd
    from pycbc.strain import detect_loud_glitches
    x=TimeSeries(np.random.default_rng(8).normal(size=8192).astype('float32'),
                 delta_t=1/512,epoch=1272790000)
    backend=JaxLiveConditioner(SimpleNamespace(),device(),'condition')
    expected=pycbc.psd.welch(x,seg_len=1024,seg_stride=512)
    actual=backend.welch(x,1024,512)
    np.testing.assert_allclose(np.asarray(actual),np.asarray(expected),rtol=5e-6,atol=1e-8)
    x[4000]+=400
    kwargs=dict(psd_duration=2.,psd_stride=1.,threshold=20.,cluster_window=1.,
                low_freq_cutoff=15.,corrupt_time=.5)
    expected=detect_loud_glitches(x,**kwargs)
    actual=backend.detect_loud_glitches(x,**kwargs)
    np.testing.assert_array_equal(actual,expected)
    assert len(expected)>0


def test_gpu_peak_kernel_does_not_need_full_normalized_surface():
    import jax.numpy as jnp
    from pycbc.filter.jax_matchedfilter import _surface_kernel,_cpu_surface
    rng=np.random.default_rng(9)
    h=jnp.asarray((rng.normal(size=(3,65))+1j*rng.normal(size=(3,65))).astype('complex64'))
    strain=jnp.ones(65,dtype=jnp.complex64)
    norms=jnp.array([.1,1.,3.],dtype=jnp.float32)
    expected=jax.device_get(_cpu_surface(h,strain,norms,np.int32(20),64))
    q=jnp.zeros((3,128),dtype=jnp.complex64);raw=jnp.zeros_like(q)
    actual=jax.device_get(_surface_kernel(64,normalize=False)(h,strain,norms,q,raw,np.int32(20)))
    for a,b in zip(actual,expected): np.testing.assert_allclose(a,b,rtol=2e-6,atol=1e-4)


def test_changed_psd_and_older_gate_update_invalidate_cached_spectrum():
    r=reader();backend=JaxLiveConditioner(r,device(),'whiten')
    _,first=backend.overwhitened(1/8);first=np.asarray(first).copy()
    r.psd=r.psd*2;r.psds={};r.segments={}
    _,second=backend.overwhitened(1/8)
    np.testing.assert_allclose(np.asarray(second),first/2,rtol=2e-5,atol=1e-6)
    r.strain.roll(-512);r.strain.start_time+=1
    r.strain.numpy()[-528:]=0
    # A gate earlier than the newest samples must also refresh device memory.
    r.strain.numpy()[3000:3100]=0
    r.gate_params=[(r.strain.start_time+3050/512,50/512,0.)]
    r.segments={}
    expected=r.overwhitened_data(1/8)
    _,actual=backend.overwhitened(1/8)
    np.testing.assert_allclose(np.asarray(actual),np.asarray(expected),rtol=3e-5,atol=3e-6)
