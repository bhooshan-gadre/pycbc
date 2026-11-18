"""Compiled batched ripple generation versus the existing serial generator."""
import os
import numpy as np
import pytest

jax=pytest.importorskip('jax')
pytest.importorskip('ripplegw')


def test_compiled_ripple_bank_padding_and_fractional_cutoffs():
    from pycbc.pnutils import mass1_mass2_to_mchirp_eta
    from pycbc.waveform.jax_phenom import generate_imrphenomd_batch,_ripple_phenomd
    device=jax.devices(os.environ.get('PYCBC_JAX_TEST_PLATFORM','cpu'))[0]
    bodies=[(35.,25.,.2,-.1),(44.,18.,-.1,.3),(30.,30.,.1,.1)]
    params=[]
    for m1,m2,c1,c2 in bodies:
        mc,eta=mass1_mass2_to_mchirp_eta(m1,m2)
        params.append([mc,eta,c1,c2,100.,0.,0.,0.])
    lower=np.array([20.03,23.07,21.08]);refs=np.array([20.,25.,21.])
    with jax.default_device(device):
        actual=generate_imrphenomd_batch(2049,.5,params,lower,refs,1024.,device,batch_size=2)
        assert actual.devices()=={device}
        for i,(m1,m2,c1,c2) in enumerate(bodies):
            expected=_ripple_phenomd(2049,.5,lower[i],refs[i],100.,m1,m2,c1,c2)
            np.testing.assert_allclose(np.asarray(actual[i]),expected,rtol=3e-5,atol=1e-29)


def test_strict_ripple_rejects_missing_dependency(monkeypatch):
    from pycbc.waveform import jax_phenom as module
    monkeypatch.setattr(module,'_load_ripple',lambda:None)
    # Use a unique grid so an existing executable cannot mask the import error.
    with pytest.raises(RuntimeError,match='no fallback'):
        module._ripple_batch_kernel(333,.123456)


@pytest.mark.parametrize('shift',[-1,1])
def test_sample_agreement_allows_one_time_sample_and_frequency_edge(shift):
    from pycbc.waveform.jax_phenom import waveform_sample_agreement
    rng=np.random.default_rng(15)
    ref=np.zeros(257,dtype='complex64')
    ref[20:200]=(rng.normal(size=180)+1j*rng.normal(size=180)).astype('complex64')
    generated=ref*np.exp(-2j*np.pi*np.arange(257)*shift/512)
    generated[199]=0
    report=waveform_sample_agreement(ref,generated)
    assert report['passed'] and report['time_sample_shift']==shift
    assert report['frequency_high_bin_delta']==-1
    generated[197:199]=0
    assert not waveform_sample_agreement(ref,generated)['passed']


def test_sample_agreement_rejects_larger_time_offset():
    from pycbc.waveform.jax_phenom import waveform_sample_agreement
    ref=np.zeros(257,dtype='complex64');ref[20:200]=1
    generated=ref*np.exp(-2j*np.pi*np.arange(257)*3/512)
    assert not waveform_sample_agreement(ref,generated)['passed']
