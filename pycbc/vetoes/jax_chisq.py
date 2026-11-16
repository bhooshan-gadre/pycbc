"""JAX power chi-squared and sine-Gaussian chi-squared at trigger peaks.

Power chi-squared follows ``point_chisq_code``: for each frequency bin,
sum ``v[j] * exp(2 * pi * i * shift * j / slen)`` with ``slen = len(corr)``
(the time-series length, not the frequency length), then

    (sum_bins |acc|^2 * n_bins - |snr|^2) * snr_norm**2

Sine-Gaussian chi-squared follows ``SingleDetSGChisq.values``. Tiles come
from ``pycbc.waveform.sinegauss.fd_sine_gaussian``. The time shift is a
multiply by ``exp(-2 * pi * i * dt * k * delta_f)``.
"""

import numpy

from pycbc.events import ranking
from pycbc.filter import sigma
from pycbc.waveform import sinegauss


# Same literal as chisq_cpu.pyx (not a rounded pi).
_POINT_PI = 3.141592653


def _import_jax():
    import jax.numpy as jnp
    return jnp


def power_chisq_at_points(corr, snr, snr_norm, bins, indices):
    """JAX equivalent of ``power_chisq_at_points_from_precomputed``.

    Parameters
    ----------
    corr : array
        Complex correlation, length N = (n_freq - 1) * 2.
    snr : array
        Unnormalized complex SNR at ``indices``.
    snr_norm : float
        SNR normalization ``4 * delta_f / sqrt(sigmasq)``.
    bins : array
        Integer bin edges.
    indices : array
        Peak indices into ``corr``'s inverse FFT (the shift).

    Returns
    -------
    chisq : ndarray
        Float32 unnormalized chi-squared, one value per index.
    """
    jnp = _import_jax()
    corr = numpy.asarray(corr)
    snr = numpy.asarray(snr, dtype=numpy.complex128)
    indices = numpy.asarray(indices, dtype=numpy.float64).reshape(-1)
    bins = numpy.asarray(bins, dtype=numpy.int64)
    if indices.size == 0:
        return numpy.zeros(0, dtype=numpy.float32)

    slen = corr.shape[-1]
    v = jnp.asarray(corr, dtype=jnp.complex64)
    j = jnp.arange(slen, dtype=jnp.float32)
    edges = jnp.asarray(bins)
    num_bins = int(len(bins) - 1)
    snr_norm = float(snr_norm)
    out = numpy.empty(indices.size, dtype=numpy.float64)

    for n, shift in enumerate(indices):
        phase = (2.0 * _POINT_PI) * float(shift) * j / float(slen)
        rot = jnp.exp(1j * phase)
        weighted = v * rot
        csum = jnp.cumsum(weighted)
        acc = 0.0
        for b in range(num_bins):
            start = int(edges[b])
            end = int(edges[b + 1])
            if end <= start:
                continue
            # sum of weighted[start:end]
            end_val = csum[end - 1]
            start_val = csum[start - 1] if start > 0 else 0
            piece = end_val - start_val
            acc = acc + (piece.real ** 2 + piece.imag ** 2)
        sn = snr[n]
        power = float(acc) * num_bins - float((sn.conjugate() * sn).real)
        out[n] = power * (snr_norm ** 2)
    return out.astype(numpy.float32, copy=False)


def reduced_power_chisq(corr, snr, snr_norm, bins, indices):
    """Reduced power chi-squared and the degree-of-freedom vector.

    Matches the live filter, which stores ``c[0] / d[0]`` with
    ``dof = 2 * num_bins - 2``.
    """
    raw = power_chisq_at_points(corr, snr, snr_norm, bins, indices)
    num_bins = len(bins) - 1
    dof = int(num_bins * 2 - 2)
    if dof == 0:
        reduced = numpy.zeros(len(raw), dtype=numpy.float32)
    else:
        reduced = (raw / numpy.float32(dof)).astype(numpy.float32, copy=False)
    dof_vec = numpy.repeat(numpy.uint32(dof), len(raw))
    return reduced, dof_vec, raw


def _time_shift(stilde, dt, delta_f):
    """Multiply by ``exp(-2 * pi * i * dt * k * delta_f)``."""
    jnp = _import_jax()
    data = jnp.asarray(numpy.asarray(stilde), dtype=jnp.complex64)
    k = jnp.arange(data.shape[0], dtype=jnp.float32)
    phase = -2.0 * numpy.pi * float(dt) * k * float(delta_f)
    shifted = data * jnp.exp(1j * phase)
    return numpy.asarray(shifted)


def sg_chisq_values(sg, stilde, template, psd, snrv, snr_norm,
                    bchisq, bchisq_dof, indices):
    """Sine-Gaussian chi-squared, same control flow as ``SingleDetSGChisq``.

    Returns an array of ones when the template hash is not in the location
    map, when sine-Gaussian chi-squared is disabled, or when a tile would
    extend above 0.9 Nyquist. Otherwise the mean tile SNR^2.
    """
    snrv = numpy.asarray(snrv)
    indices = numpy.asarray(indices)
    n = len(snrv)
    if not getattr(sg, 'do', False):
        return None
    if template.params.template_hash not in sg.params:
        return numpy.ones(n, dtype=numpy.float32)

    values = sg.params[template.params.template_hash].split(',')
    bins = sg.cached_chisq_bins(template, psd)
    chisq = numpy.ones(n, dtype=numpy.float64)
    gtem = [None for _ in values]

    for i, snrvi in enumerate(snrv):
        snr = abs(snrvi * snr_norm)
        reduced = float(bchisq[i]) / float(bchisq_dof[i])
        nsnr = ranking.newsnr(snr, reduced)
        if nsnr < sg.snr_threshold:
            continue

        ntime = (len(template) - 1) * 2
        dt_sample = 1.0 / (ntime * template.delta_f)
        kmin_tmpl = int(template.f_lower / psd.delta_f)
        epoch = template.epoch
        epoch = 0.0 if epoch is None else float(epoch)
        time = epoch + dt_sample * float(indices[i])
        stilde_shift = _time_shift(stilde, -time, stilde.delta_f)

        qwindow = 50
        fstep = (bins[-2] - bins[-3])
        fpeak = (bins[-2] + fstep) * template.delta_f
        fstop = len(stilde) * stilde.delta_f * 0.9

        chisq[i] = 0.0
        dof = 0
        for idxx, descr in enumerate(values):
            q, offset = descr.split('-')
            q, offset = float(q), float(offset)
            fcen = fpeak + offset
            flow = max(kmin_tmpl * template.delta_f, fcen - qwindow)
            fhigh = fcen + qwindow
            if fhigh > fstop:
                return numpy.ones(n, dtype=numpy.float32)

            kmin = int(flow / template.delta_f)
            kmax = int(fhigh / template.delta_f)
            if gtem[idxx] is None:
                gtem[idxx] = sinegauss.fd_sine_gaussian(
                    1.0, q, fcen, flow,
                    len(template) * template.delta_f,
                    template.delta_f).astype(numpy.complex64)
            gsigma = sigma(gtem[idxx], psd=psd,
                           low_frequency_cutoff=flow,
                           high_frequency_cutoff=fhigh)
            tile = numpy.asarray(gtem[idxx][kmin:kmax], dtype=numpy.complex128)
            data = numpy.asarray(stilde_shift[kmin:kmax], dtype=numpy.complex128)
            gsnr = numpy.sum(tile * data)
            gsnr *= 4.0 * float(gtem[idxx].delta_f) / float(gsigma)
            chisq[i] += abs(gsnr) ** 2
            dof += 2
        if dof == 0:
            chisq[i] = 1.0
        else:
            chisq[i] /= dof
    return chisq.astype(numpy.float32, copy=False)
