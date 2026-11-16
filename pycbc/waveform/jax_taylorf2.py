"""Batched complex64 stationary-phase TaylorF2 templates.

This is the SPAtmplt fast path (``spa_tmplt_cpu.spa_tmplt_inline``): LAL
aligned-spin phasing coefficients, a float32 polynomial phase, and the
fast sine/cosine approximation. Coefficients are computed once per
template on the CPU. The stride loop only evaluates the JAX frequency
series.
"""

import logging
from math import log

import numpy

import pycbc.pnutils
from pycbc.constants import PI, MTSUN_SI
from pycbc.libutils import import_optional
from pycbc.waveform.spa_tmplt import spa_amplitude_factor, spa_tmplt_precondition

logger = logging.getLogger('pycbc.waveform.jax_taylorf2')

lal = import_optional('lal')
lalsimulation = import_optional('lalsimulation')

# libc M_PI, assigned to float in the Cython engine.
_TWO_PI = numpy.float32(2.0 * PI)
_PI = numpy.float32(PI)
_PI_2 = numpy.float32(PI / 2.0)
_PI_4 = numpy.float32(PI / 4.0)
_LOG4 = numpy.float32(log(4.0))


def taylorf2_phasing_coefficients(mass1, mass2, spin1z, spin2z,
                                  phase_order=-1, spin_order=-1,
                                  distance=1.0):
    """LAL ``SimInspiralTaylorF2AlignedPhasing`` coefficients, once per template.

    Returns
    -------
    coeffs : dict
        Float32 scalars consumed by :func:`spa_on_grid`. ``amp_factor`` already
        includes the luminosity-distance scaling used by ``spa_tmplt``.
    """
    if lal is None or lalsimulation is None:
        raise RuntimeError('LAL is required to build TaylorF2 phasing coefficients')

    phase_order = int(phase_order)
    spin_order = int(spin_order)
    lal_pars = lal.CreateDict()
    if phase_order != -1:
        lalsimulation.SimInspiralWaveformParamsInsertPNPhaseOrder(
            lal_pars, phase_order)
    if spin_order != -1:
        lalsimulation.SimInspiralWaveformParamsInsertPNSpinOrder(
            lal_pars, spin_order)

    phasing = lalsimulation.SimInspiralTaylorF2AlignedPhasing(
        float(mass1), float(mass2), float(spin1z), float(spin2z), lal_pars)

    pfaN = phasing.v[0]
    amp = spa_amplitude_factor(mass1=mass1, mass2=mass2) / float(distance)
    piM = PI * (float(mass1) + float(mass2)) * MTSUN_SI
    piM_f = numpy.float32(piM)
    # Match the Cython engine: cbrt/log are the double libm calls, then stored
    # as float.
    piM13 = numpy.float32(numpy.cbrt(float(piM_f)))
    return {
        'piM': piM_f,
        'piM13': piM13,
        'logpiM13': numpy.float32(numpy.log(float(piM13))),
        'pfaN': numpy.float32(pfaN),
        'pfa2': numpy.float32(phasing.v[2] / pfaN),
        'pfa3': numpy.float32(phasing.v[3] / pfaN),
        'pfa4': numpy.float32(phasing.v[4] / pfaN),
        'pfa5': numpy.float32(phasing.v[5] / pfaN),
        'pfa6': numpy.float32((phasing.v[6] - phasing.vlogv[6] * log(4.0)) / pfaN),
        'pfa7': numpy.float32(phasing.v[7] / pfaN),
        'pfl5': numpy.float32(phasing.vlogv[5] / pfaN),
        'pfl6': numpy.float32(phasing.vlogv[6] / pfaN),
        'amp_factor': numpy.float32(amp),
    }


def _import_jax():
    import jax.numpy as jnp
    return jnp


def _fast_phase(phasing, jnp):
    """Float32 phase wrap and fast sin/cos used by ``spa_tmplt_inline``."""
    two_pi = jnp.float32(_TWO_PI)
    pi = jnp.float32(_PI)
    pi_2 = jnp.float32(_PI_2)
    pi_4 = jnp.float32(_PI_4)
    phasing = phasing - pi_4
    nwrap = jnp.trunc(phasing / two_pi)
    phasing = phasing - nwrap * two_pi
    phasing = jnp.where(phasing < -pi, phasing + two_pi, phasing)
    phasing = jnp.where(phasing > pi, phasing - two_pi, phasing)

    a = jnp.float32(1.273239545)
    b = jnp.float32(0.405284735)
    c = jnp.float32(0.225)
    sinp = a * phasing - b * phasing * jnp.abs(phasing)
    sinp = c * (sinp * jnp.abs(sinp) - sinp) + sinp

    phasing = phasing + pi_2
    phasing = jnp.where(phasing > pi, phasing - two_pi, phasing)
    cosp = a * phasing - b * phasing * jnp.abs(phasing)
    cosp = c * (cosp * jnp.abs(cosp) - cosp) + cosp
    return cosp, sinp


def _phasing_polynomial(v, logv, coeffs, jnp):
    """Horner evaluation of the SPAtmplt phase, before the pi/4 shift."""
    pfa7 = coeffs['pfa7']
    pfa6 = coeffs['pfa6']
    pfl6 = coeffs['pfl6']
    pfa5 = coeffs['pfa5']
    pfl5 = coeffs['pfl5']
    pfa4 = coeffs['pfa4']
    pfa3 = coeffs['pfa3']
    pfa2 = coeffs['pfa2']
    pfaN = coeffs['pfaN']
    log4 = jnp.float32(_LOG4)
    phasing = pfa7 * v
    phasing = (phasing + pfa6 + pfl6 * (logv + log4)) * v
    phasing = (phasing + pfa5 + pfl5 * logv) * v
    phasing = (phasing + pfa4) * v
    phasing = (phasing + pfa3) * v
    phasing = (phasing + pfa2) * v * v + jnp.float32(1.0)
    v5 = v * v * v * v * v
    return phasing * pfaN / v5


def spa_on_grid(kfac, cbrt, logv, coeffs):
    """Evaluate one SPAtmplt slice on a prepared frequency grid.

    Parameters
    ----------
    kfac, cbrt, logv : array
        Float32 amplitude, cube-root and log tables for this slice, in the
        same layout as ``spa_tmplt_engine`` (length ``kmax - kmin``).
    coeffs : dict
        Output of :func:`taylorf2_phasing_coefficients`.

    Returns
    -------
    h : ndarray
        Complex64 strain for the slice.
    """
    jnp = _import_jax()
    kfac = jnp.asarray(kfac, dtype=jnp.float32)
    cbrt = jnp.asarray(cbrt, dtype=jnp.float32)
    logv = jnp.asarray(logv, dtype=jnp.float32)
    piM13 = jnp.float32(coeffs['piM13'])
    logpiM13 = jnp.float32(coeffs['logpiM13'])
    v = piM13 * cbrt
    logv = logv * jnp.float32(1.0 / 3.0) + logpiM13
    amp = jnp.float32(coeffs['amp_factor']) * kfac
    phasing = _phasing_polynomial(v, logv, coeffs, jnp)
    cosp, sinp = _fast_phase(phasing, jnp)
    h = (cosp - sinp * 1j) * amp
    return numpy.asarray(h, dtype=numpy.complex64)


def batched_spa(kfac, cbrt, logv, coeff_list):
    """Stack :func:`spa_on_grid` over templates that share one frequency grid.

    Parameters
    ----------
    coeff_list : sequence of dict
        One coefficient dict per template.

    Returns
    -------
    h : ndarray
        Complex64 array shaped ``(n_templates, n_freq)``.
    """
    if len(coeff_list) == 0:
        return numpy.zeros((0, len(kfac)), dtype=numpy.complex64)
    rows = [spa_on_grid(kfac, cbrt, logv, c) for c in coeff_list]
    return numpy.stack(rows, axis=0)


def grid_tables(kmin, kmax, delta_f):
    """Cube-root, log and amplitude tables for bins ``[kmin, kmax)``."""
    from pycbc.waveform.spa_tmplt_cpu import get_cbrt, get_log

    nslice = int(kmax - kmin)
    if nslice <= 0:
        empty = numpy.zeros(0, dtype=numpy.float32)
        return empty, empty, empty
    kfac = numpy.asarray(
        spa_tmplt_precondition(nslice, delta_f, kmin), dtype=numpy.float32)
    cbrt = numpy.asarray(
        get_cbrt(nslice * delta_f + kmin, delta_f), dtype=numpy.float32)
    logv = numpy.asarray(
        get_log(nslice * delta_f + kmin, delta_f), dtype=numpy.float32)
    return kfac, cbrt[kmin:kmax], logv[kmin:kmax]


def generate_spa_template(length, delta_f, f_lower, f_final, coeffs):
    """Full complex64 frequency series, zero outside ``[f_lower, f_final)``."""
    length = int(length)
    kmin = int(float(f_lower) / float(delta_f))
    kmax = int(float(f_final) / float(delta_f))
    if kmax > length:
        kmax = length
    h = numpy.zeros(length, dtype=numpy.complex64)
    if kmax <= kmin:
        return h
    kfac, cbrt, logv = grid_tables(kmin, kmax, delta_f)
    h[kmin:kmax] = spa_on_grid(kfac, cbrt, logv, coeffs)
    return h


def zero_lag_match(h_ref, h_new):
    """Normalized zero-lag match. Phase and time are not maximized."""
    a = numpy.nan_to_num(numpy.asarray(h_ref, dtype=numpy.complex128))
    b = numpy.nan_to_num(numpy.asarray(h_new, dtype=numpy.complex128))
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    na = numpy.vdot(a, a).real
    nb = numpy.vdot(b, b).real
    if na <= 0 or nb <= 0:
        return float('nan')
    return float(abs(numpy.vdot(a, b)) / numpy.sqrt(na * nb))
