"""Aligned-spin IMRPhenomD frequency series for the JAX live filter.

``ripplegw`` is used when it imports. Its PhenomD is checked against LAL
at startup. If ``ripplegw`` is not installed, a small in-tree aligned-spin
model is used instead and the same startup comparison reports the mismatch.
Templates are generated once and returned as complex64 arrays.
"""

import logging
from functools import lru_cache

import numpy

from pycbc.waveform.jax_taylorf2 import (
    generate_spa_template,
    taylorf2_phasing_coefficients,
    zero_lag_match,
)
from pycbc.waveform.spa_tmplt import spa_tmplt_end

logger = logging.getLogger('pycbc.waveform.jax_phenom')

_RIPPLE = None
_RIPPLE_TRIED = False
_RIPPLE_COMPILED = None


def waveform_sample_agreement(reference, generated, mismatch_limit=1e-6):
    """Check waveform shape allowing one time sample and cutoff-frequency bin.

    Only validation is aligned; the generated waveform is returned unchanged
    to filtering. SNR/veto values are still independently checked downstream.
    """
    reference, generated = numpy.asarray(reference), numpy.asarray(generated)
    if reference.ndim != 1 or reference.shape != generated.shape or len(reference) < 2:
        raise ValueError('Waveform validation requires the same nontrivial frequency grid')
    if not numpy.isfinite(reference).all() or not numpy.isfinite(generated).all():
        return dict(passed=False,reason='nonfinite waveform')
    a, b = numpy.flatnonzero(reference), numpy.flatnonzero(generated)
    if not len(a) or not len(b):
        return dict(passed=False, reason='empty waveform support')
    low, high = int(b[0]-a[0]), int(b[-1]-a[-1])
    start, stop = max(a[0], b[0]), min(a[-1], b[-1])+1
    if start >= stop:
        return dict(passed=False, reason='disjoint waveform support')
    ntime = 2*(len(reference)-1)
    freq = numpy.arange(start, stop, dtype=numpy.float64)
    matches = [(zero_lag_match(reference[start:stop], generated[start:stop]*
                numpy.exp(2j*numpy.pi*freq*offset/ntime)), offset)
               for offset in (0, -1, 1)]
    aligned, shift = max(matches, key=lambda item: item[0])
    passed = (abs(low) <= 1 and abs(high) <= 1 and
              numpy.isfinite(aligned) and 1-aligned < mismatch_limit)
    return dict(passed=bool(passed),
        match_template=float(zero_lag_match(reference,generated)),
        aligned_match_template=float(aligned),time_sample_shift=int(shift),
        frequency_low_bin_delta=low,frequency_high_bin_delta=high)


@lru_cache(None)
def _ripple_batch_kernel(length, delta_f):
    """One JIT/vmap executable per grid and parameter-batch shape."""
    import jax
    import jax.numpy as jnp
    gen = _load_ripple()
    if gen is None:
        raise RuntimeError('Compiled ripple IMRPhenomD requires ripplegw; no fallback is permitted')
    @jax.jit
    def run(parameters, lower, reference, final):
        # Ripple derives its cutoff from f[1]-f[0]; keep the grid uniform.
        # The singular unused DC value is masked from the returned bank.
        freq = jnp.arange(length,dtype=jnp.float64)*delta_f
        def one(p,lo,ref,hi):
            hp,_ = gen(freq,p,ref)
            index=jnp.arange(length)
            keep = (index >= (lo/delta_f).astype(jnp.int32))&(index < (hi/delta_f).astype(jnp.int32))&(index>0)&(index<length-1)
            return jnp.where(keep,hp,0).astype(jnp.complex64)
        return jax.vmap(one)(parameters,lower,reference,final)
    return run


def generate_imrphenomd_batch(length, delta_f, parameters, lower, reference,
                             final, device, batch_size=32):
    """Generate a device-resident bank with compiled float64 ripple phasing.

    Parameters are ripple's [Mchirp, eta, chi1z, chi2z, distance, tc, phase,
    inclination]. Pad the final chunk so it reuses the same executable.
    Returns complex64 on the requested device; never substitutes a model.
    """
    import jax
    import jax.numpy as jnp
    parameters=numpy.asarray(parameters,dtype=numpy.float64)
    if parameters.ndim != 2 or parameters.shape[1] != 8 or len(parameters)==0 or batch_size < 1:
        raise ValueError('Expected a nonempty (templates,8) parameter array and positive batch size')
    vectors=[numpy.broadcast_to(numpy.asarray(x,dtype=numpy.float64),(len(parameters),)) for x in (lower,reference,final)]
    # Avoid generating 32 rows for a two-template bank, or exhausting device
    # memory with float64 waveform intermediates on a very long FFT grid.
    batch_size=min(batch_size,1<<(len(parameters)-1).bit_length(),max(1,2**20//int(length)))
    result=[]
    with jax.default_device(device),jax.enable_x64(True):
        kernel=_ripple_batch_kernel(int(length),float(delta_f))
        for start in range(0,len(parameters),batch_size):
            count=min(batch_size,len(parameters)-start)
            arrays=[parameters[start:start+count],*[v[start:start+count] for v in vectors]]
            padded=[numpy.pad(a,[(0,batch_size-count)]+[(0,0)]*(a.ndim-1),mode='edge') for a in arrays]
            value=kernel(*[jax.device_put(a,device) for a in padded])
            result.append(value[:count])
        bank=jnp.concatenate(result,axis=0)
        if not bool(jnp.isfinite(bank).all()):
            raise ValueError('Compiled ripple IMRPhenomD produced nonfinite strain')
        return bank


def _load_ripple():
    global _RIPPLE, _RIPPLE_TRIED
    if _RIPPLE_TRIED:
        return _RIPPLE
    _RIPPLE_TRIED = True
    try:
        from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD import gen_IMRPhenomD_hphc
        _RIPPLE = gen_IMRPhenomD_hphc
    except ImportError:
        _RIPPLE = None
        logger.info('ripplegw is not importable; using the in-tree IMRPhenomD')
    return _RIPPLE


def phenomd_implementation():
    """``'ripple'`` or ``'intree'``."""
    return 'ripple' if _load_ripple() is not None else 'intree'


def _f_ref_hz(f_ref, f_lower):
    """LAL treats a non-positive f_ref as the low-frequency cutoff."""
    f_ref = float(f_ref or 0.0)
    if f_ref <= 0.0:
        return float(f_lower)
    return f_ref


def _ripple_phenomd(length, delta_f, f_lower, f_ref, distance,
                    mass1, mass2, spin1z, spin2z):
    import jax
    import jax.numpy as jnp
    from pycbc.pnutils import mass1_mass2_to_mchirp_eta

    global _RIPPLE_COMPILED
    gen = _load_ripple()
    if gen is None:
        raise RuntimeError('ripplegw IMRPhenomD is unavailable')
    if _RIPPLE_COMPILED is None:
        # Reuse one compiled generator across templates. Passing parameters
        # and f_ref as arrays keeps masses/spins/reference frequency dynamic;
        # executable variants are keyed by grid shape, dtype and device.
        _RIPPLE_COMPILED = jax.jit(gen)
    gen = _RIPPLE_COMPILED
    # Ripple's chirp-mass conversion returns the heavier body first.
    if float(mass1) >= float(mass2):
        chi1, chi2 = float(spin1z), float(spin2z)
    else:
        chi1, chi2 = float(spin2z), float(spin1z)
    mc, eta = mass1_mass2_to_mchirp_eta(float(mass1), float(mass2))
    f_ref = _f_ref_hz(f_ref, f_lower)
    # PhenomD phase coefficients are sensitive to rounded eta/frequency input.
    # Generate in double precision, then filter/store complex64 as before.
    # Float32 generation can pass a global overlap check while moving SNR
    # sidelobes and off-peak veto samples beyond the live parity tolerance.
    with jax.enable_x64(True):
        freqs = jnp.arange(int(length), dtype=jnp.float64) * jnp.float64(delta_f)
        params = jnp.asarray(
            [mc, eta, chi1, chi2, float(distance), 0.0, 0.0, 0.0],
            dtype=jnp.float64)
        hp, _hc = gen(freqs, params, jnp.float64(f_ref))
        h = numpy.asarray(hp).astype(numpy.complex64, copy=True)
    if not numpy.isfinite(h).all():
        # The unused DC bin can be singular; nonfinite in-band strain is an
        # actual generation error, not a waveform to silently zero out.
        kmin = int(float(f_lower) / float(delta_f))
        if not numpy.isfinite(h[max(1, kmin):-1]).all():
            raise ValueError('ripplegw IMRPhenomD produced nonfinite in-band strain')
    kmin = int(float(f_lower) / float(delta_f))
    if kmin > 0:
        h[:kmin] = 0
    h[0] = 0
    # LAL's bank frequency interval excludes its f_final / Nyquist endpoint.
    h[-1] = 0
    return numpy.asarray(h, dtype=numpy.complex64)


def _intree_phenomd(length, delta_f, f_lower, f_ref, distance,
                    mass1, mass2, spin1z, spin2z):
    """Aligned-spin stand-in: SPA inspiral plus a Lorentzian ringdown.

    This is not LAL IMRPhenomD. It exists so the filter can still be built
    without ``ripplegw``, and so startup can report the mismatch explicitly.
    """
    f_isco = float(spa_tmplt_end(mass1=mass1, mass2=mass2))
    coeffs = taylorf2_phasing_coefficients(
        mass1, mass2, spin1z, spin2z, distance=distance)
    h = generate_spa_template(length, delta_f, f_lower, f_isco, coeffs)
    # Attach a simple ringdown above ISCO so the spectrum is not pure SPA.
    k_isco = int(f_isco / float(delta_f))
    if 0 < k_isco < int(length) - 1:
        freqs = numpy.arange(k_isco, int(length), dtype=numpy.float64) * float(delta_f)
        # Ringdown frequency ~ 1.2 f_ISCO, width ~ 0.1 f_ISCO, scaled to the
        # last inspiral sample so the junction is continuous in amplitude.
        amp0 = abs(h[k_isco - 1]) if k_isco else 0.0
        fring = 1.2 * f_isco
        gamma = 0.1 * f_isco
        lorentz = amp0 * gamma**2 / ((freqs - fring)**2 + gamma**2)
        phase = numpy.exp(-2j * numpy.pi * (freqs - f_isco) / max(f_isco, 1.0))
        h[k_isco:] = (lorentz * phase).astype(numpy.complex64)
    return h


def generate_imrphenomd(length, delta_f, f_lower, f_ref, distance,
                        mass1, mass2, spin1z, spin2z):
    """Complex64 IMRPhenomD frequency series on ``[0, length) * delta_f``."""
    if _load_ripple() is not None:
        return _ripple_phenomd(length, delta_f, f_lower, f_ref, distance,
                               mass1, mass2, spin1z, spin2z)
    return _intree_phenomd(length, delta_f, f_lower, f_ref, distance,
                           mass1, mass2, spin1z, spin2z)


def compare_phenomd_to_reference(h_ref, length, delta_f, f_lower, f_ref,
                                 distance, mass1, mass2, spin1z, spin2z):
    """Generate the JAX PhenomD and return ``(match, implementation)``.

    The match is the zero-lag normalized overlap against ``h_ref`` (the LAL
    series the live bank already produced, or any series on the same grid).
    """
    h = generate_imrphenomd(length, delta_f, f_lower, f_ref, distance,
                            mass1, mass2, spin1z, spin2z)
    return zero_lag_match(h_ref, h), phenomd_implementation(), h
