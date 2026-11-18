"""JAX power chi-squared and sine-Gaussian chi-squared at trigger peaks.

Power chi-squared follows ``point_chisq_code``: for each frequency bin,
sum ``v[j] * exp(2 * pi * i * shift * j / slen)`` with ``slen = len(corr)``
(the time-series length, not the frequency length), then

    (sum_bins |acc|^2 * n_bins - |snr|^2) * snr_norm**2

Bin sums preserve the native float32 phase recurrence. GPU point vetoes use
compiled CUDA loops through CuPy/NVRTC and DLPack, avoiding an XLA kernel
launch per frequency sample. Sine-Gaussian
chi-squared follows ``SingleDetSGChisq.values``. Tiles come from
``pycbc.waveform.sinegauss.fd_sine_gaussian``. The time shift is a multiply by
``exp(-2 * pi * i * dt * k * delta_f)``, evaluated for every Q node and every
row in one kernel.
"""

import functools

import numpy

from pycbc.events import ranking
from pycbc.filter import sigma
from pycbc.waveform import sinegauss


# Same literal as chisq_cpu.pyx (not a rounded pi).
_POINT_PI = 3.141592653


def _import_jax():
    import jax
    import jax.numpy as jnp
    return jax, jnp


@functools.lru_cache(maxsize=None)
def _power_kernel(n_edges, ntime):
    """Shift-sum kernel specialized on the (static) number of bin edges."""
    jax, jnp = _import_jax()
    n_edges = int(n_edges)

    @jax.jit
    def kernel(q, shifts, bins, snr, norms, num_bins):
        # q (B, slen), shifts (B,), bins (B, E), snr (B,) complex,
        # norms (B,), num_bins (B,). Empty bins (end <= start) add nothing.
        # num_bins is the unpadded edge count minus one, matching len(bins)-1
        # in power_chisq_at_points_from_precomputed.
        rows, support = q.shape
        start, end = bins[:,:-1], bins[:,1:]
        # Native point_chisq_code rotates in float32 *within each bin*.
        # Exact exp(i*k*phase) and a float64 prefix are mathematically more
        # accurate but differ from that recurrence for long bins. Reproduce
        # the native arithmetic, parallel over both candidates and bins.
        phase = (2.0*_POINT_PI)*shifts.astype(jnp.float64)[:,None]/ntime
        angle = phase*start.astype(jnp.float64)
        pr, pi = jnp.cos(angle).astype(jnp.float32), jnp.sin(angle).astype(jnp.float32)
        vsr, vsi = jnp.cos(phase).astype(jnp.float32), jnp.sin(phase).astype(jnp.float32)
        zero = jnp.zeros_like(pr)
        width = jnp.maximum(end-start,0)
        def body(j, carry):
            pr,pi,real,imag = carry
            index = start+j
            value = q[jnp.arange(rows)[:,None],jnp.clip(index,0,support-1)]
            vr,vi = value.real,value.imag
            vs,va = vr+vi,vi-vr
            k1,k2,k3 = jax.lax.optimization_barrier((vr*(pr+pi),pr*va,pi*vs))
            rr,ii,ri,ir = jax.lax.optimization_barrier((pr*vsr,pi*vsi,pr*vsi,pi*vsr))
            real = real+jnp.where(j<width,k1-k3,0)
            imag = imag+jnp.where(j<width,k1+k2,0)
            return rr-ii,ri+ir,real,imag
        _,_,real,imag = jax.lax.fori_loop(0,jnp.max(width),body,(pr,pi,zero,zero))
        pieces = jax.lax.optimization_barrier(real*real)+jax.lax.optimization_barrier(imag*imag)
        # Match the native sequential accumulation over bins too.
        acc = jax.lax.fori_loop(0,n_edges-1,lambda i,s:s+pieces[:,i],jnp.zeros(rows,jnp.float32))
        power = jax.lax.optimization_barrier(acc*num_bins)-(
            jax.lax.optimization_barrier(snr.real*snr.real)+jax.lax.optimization_barrier(snr.imag*snr.imag))
        return power*(norms*norms)

    return kernel


def power_chisq_batch(q, snr, norms, bins, shifts, num_bins, ntime=None):
    """Batched shift-sum chi-squared for one row per template (or per index).

    Parameters
    ----------
    q : array
        Complex correlation, shape ``(batch, slen)``.
    snr : array
        Unnormalized complex SNR, shape ``(batch,)``.
    norms : array
        SNR normalization ``4 * delta_f / sqrt(sigmasq)``, shape ``(batch,)``.
    bins : array
        Integer bin edges, shape ``(batch, n_edges)``. Padded edges must
        repeat the last real edge so those intervals are empty.
    shifts : array
        Peak index into the inverse FFT, shape ``(batch,)``.
    num_bins : array
        Real bin count ``len(edges) - 1`` before padding, shape ``(batch,)``.
    ntime : int, optional
        Original inverse FFT length. Supply this when ``q`` contains only
        the positive-frequency support; phase shifts still use the original
        time length. All bin edges must lie within the supplied support.

    Returns
    -------
    chisq : jax array
        Float32 unnormalized chi-squared, shape ``(batch,)``. Stays on the
        device of ``q``.
    """
    jax, jnp = _import_jax()
    q = jnp.asarray(q, dtype=jnp.complex64)
    if q.ndim == 1:
        q = q[None, :]
    ntime = int(q.shape[1] if ntime is None else ntime)
    if ntime < q.shape[1]:
        raise ValueError('ntime cannot be shorter than the correlation support')
    snr = jnp.asarray(snr, dtype=jnp.complex64).reshape(-1)
    norms = jnp.asarray(norms, dtype=jnp.float32).reshape(-1)
    bins = jnp.asarray(bins, dtype=jnp.int32)
    if bins.ndim == 1:
        bins = jnp.broadcast_to(bins, (q.shape[0], bins.shape[0]))
    shifts = jnp.asarray(shifts, dtype=jnp.float32).reshape(-1)
    num_bins = jnp.asarray(num_bins, dtype=jnp.float32).reshape(-1)
    if next(iter(q.devices())).platform == 'gpu':
        from .jax_cuda import power_chisq_cuda
        return power_chisq_cuda(q,snr,norms,bins,shifts,num_bins,
                               numpy.arange(q.shape[0],dtype=numpy.int32),ntime)
    # Native initializes float32 rotations from double-precision angles.
    with jax.enable_x64(True):
        return _power_kernel(int(bins.shape[1]), ntime)(
            q, shifts, bins, snr, norms, num_bins)


@functools.lru_cache(maxsize=None)
def _selected_power_kernel(n_edges, support, ntime):
    """Gather candidate rows and truncate zero frequencies inside one JIT."""
    jax, _jnp = _import_jax()

    @jax.jit
    def kernel(q, snr, norms, bins, shifts, num_bins, rows):
        return _power_kernel(n_edges, ntime)(
            q[rows, :support], shifts[rows], bins[rows], snr[rows],
            norms[rows], num_bins[rows])

    return kernel


def power_chisq_selected(q, snr, norms, bins, shifts, num_bins, rows, support):
    """Veto only selected rows, without scanning the unused FFT half.

    ``rows`` may repeat a candidate to pad a batch to a stable bucket size.
    ``support`` is the largest chi-squared bin edge in the duration group.
    The caller discards the padded outputs.
    """
    jax, jnp = _import_jax()
    q = jnp.asarray(q,dtype=jnp.complex64)
    if next(iter(q.devices())).platform == 'gpu':
        from .jax_cuda import power_chisq_cuda
        return power_chisq_cuda(q,snr,norms,bins,jnp.asarray(shifts,dtype=jnp.float32),num_bins,rows,q.shape[1])
    with jax.enable_x64(True):
        return _selected_power_kernel(int(bins.shape[1]), int(support), q.shape[1])(
            q, snr, norms, bins, shifts, num_bins,
            jnp.asarray(rows, dtype=jnp.int32))


@functools.lru_cache(maxsize=None)
def _power_points_kernel(n_edges, support, ntime):
    jax, _jnp = _import_jax()

    @jax.jit
    def kernel(q, norms, bins, num_bins, rows, indices, peaks):
        selected_norms = norms[rows]
        return _power_kernel(n_edges, ntime)(
            q[rows, :support], indices, bins[rows], peaks / selected_norms,
            selected_norms, num_bins[rows])

    return kernel


def power_chisq_candidate_points(q, norms, bins, num_bins, rows, indices,
                                peaks, support):
    """Power veto at arbitrary offline peaks, fusing all candidate gathers.

    A template row can occur more than once, with a different time index
    each time. ``peaks`` contains normalized complex SNR samples.
    """
    jax, jnp = _import_jax()
    q = jnp.asarray(q,dtype=jnp.complex64)
    norms = jnp.asarray(norms,dtype=jnp.float32)
    # Normalize in float32, as the surface is complex64. The accumulation
    # inside _power_kernel uses native-compatible per-bin rotations.
    rows = jnp.asarray(rows, dtype=jnp.int32)
    indices = jnp.asarray(indices, dtype=jnp.int32)
    peaks = jnp.asarray(peaks, dtype=jnp.complex64)
    if next(iter(q.devices())).platform == 'gpu':
        from .jax_cuda import power_chisq_cuda
        return power_chisq_cuda(q,peaks/norms[rows],norms,bins,
                               indices.astype(jnp.float32),num_bins,rows,q.shape[1],points=True)
    with jax.enable_x64(True):
        return _power_points_kernel(int(bins.shape[1]), int(support), q.shape[1])(
            q, norms, bins, num_bins, rows, indices, peaks)


def power_chisq_at_points(corr, snr, snr_norm, bins, indices):
    """JAX equivalent of ``power_chisq_at_points_from_precomputed``.

    Parameters
    ----------
    corr : array
        Complex correlation, shape ``(slen,)`` or ``(batch, slen)``. A single
        correlation is reused for every index.
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
    corr = numpy.asarray(corr)
    snr = numpy.asarray(snr, dtype=numpy.complex64).reshape(-1)
    indices = numpy.asarray(indices, dtype=numpy.float64).reshape(-1)
    bins = numpy.asarray(bins, dtype=numpy.int32).reshape(-1)
    if indices.size == 0:
        return numpy.zeros(0, dtype=numpy.float32)
    if corr.ndim == 1:
        q = numpy.repeat(corr[None, :], indices.size, axis=0)
    elif corr.shape[0] == 1 and indices.size > 1:
        q = numpy.repeat(corr, indices.size, axis=0)
    else:
        q = corr
    num_bins = numpy.full(indices.size, bins.size - 1, dtype=numpy.float32)
    norms = numpy.full(indices.size, float(snr_norm), dtype=numpy.float32)
    bin_rows = numpy.repeat(bins[None, :], indices.size, axis=0)
    raw = power_chisq_batch(q, snr, norms, bin_rows, indices, num_bins)
    return numpy.asarray(raw, dtype=numpy.float32)


def reduced_power_chisq(corr, snr, snr_norm, bins, indices):
    """Reduced power chi-squared and the degree-of-freedom vector.

    Matches the live filter, which stores ``c[0] / d[0]`` with
    ``dof = 2 * num_bins - 2``.
    """
    raw = power_chisq_at_points(corr, snr, snr_norm, bins, indices)
    num_bins = len(numpy.asarray(bins).reshape(-1)) - 1
    dof = int(num_bins * 2 - 2)
    if dof == 0:
        reduced = numpy.zeros(len(raw), dtype=numpy.float32)
    else:
        reduced = (raw / numpy.float32(dof)).astype(numpy.float32, copy=False)
    dof_vec = numpy.repeat(numpy.uint32(dof), len(raw))
    return reduced, dof_vec, raw


def reduced_from_raw(raw, num_bins):
    """Reduced chi-squared ``raw / (2 * num_bins - 2)`` and the dof vector."""
    raw = numpy.asarray(raw, dtype=numpy.float32)
    num_bins = numpy.asarray(num_bins, dtype=numpy.int32).reshape(-1)
    dof = (num_bins * 2 - 2).astype(numpy.int32)
    reduced = numpy.zeros(raw.shape, dtype=numpy.float32)
    ok = dof > 0
    reduced[ok] = raw[ok] / dof[ok].astype(numpy.float32)
    return reduced, dof.astype(numpy.uint32)


@functools.lru_cache(maxsize=None)
def _sg_kernel():
    """Tile inner products for every row and every Q node."""
    jax, jnp = _import_jax()

    @jax.jit
    def kernel(stilde, tiles, k_idx, gsigma, times, delta_f):
        # stilde (F,), tiles (B, Q, K), k_idx (B, Q, K), gsigma (B, Q),
        # times (B,). k_idx < 0 is padding. Phase matches
        # apply_fseries_time_shift(stilde, -time).
        valid = (k_idx >= 0) & (k_idx < stilde.shape[0])
        safe = jnp.clip(k_idx, 0, stilde.shape[0] - 1)
        data = stilde[safe]
        # Match utils_cpu.fstimeshift32's single-precision phase increment
        # and phase reset every 100 bins. A single large FP32 angle loses
        # enough precision to move SG vetoes at late peaks by ~1e-3.
        phi = (2.0 * jnp.pi * times.astype(jnp.float64)
               * delta_f.astype(jnp.float64)).astype(jnp.float32)
        # Tiles have contiguous frequency support. Build each 100-bin
        # rotation sequence once, rather than doing up to 99 rotations and
        # a double-precision cosine separately for every frequency sample.
        first = jnp.maximum(k_idx[:,:,0],0)//100*100
        groups = (k_idx.shape[-1]+198)//100
        anchor = first[:,:,None]+100*jnp.arange(groups)[None,None,:]
        base = (phi[:, None, None] * anchor.astype(jnp.float32)).astype(jnp.float64)
        real,imag = jnp.cos(base).astype(jnp.float32),jnp.sin(base).astype(jnp.float32)
        inc_r = jnp.cos(phi.astype(jnp.float64)).astype(jnp.float32)[:,None,None]
        inc_i = jnp.sin(phi.astype(jnp.float64)).astype(jnp.float32)[:,None,None]
        def rotate(carry,_):
            r,i=carry
            rr,ii,ri,ir=jax.lax.optimization_barrier((r*inc_r,i*inc_i,r*inc_i,i*inc_r))
            return (rr-ii,ri+ir),(r,i)
        _,(real_table,imag_table)=jax.lax.scan(rotate,(real,imag),None,length=100)
        group=jnp.clip((k_idx-first[:,:,None])//100,0,groups-1)
        row=jnp.arange(k_idx.shape[0])[:,None,None]
        node=jnp.arange(k_idx.shape[1])[None,:,None]
        real=real_table[k_idx%100,row,node,group]
        imag=imag_table[k_idx%100,row,node,group]
        rr,ii,ri,ir=jax.lax.optimization_barrier((real*data.real,imag*data.imag,real*data.imag,imag*data.real))
        shifted = jax.lax.complex(rr-ii,ri+ir)
        prod = tiles * shifted
        prod = jnp.where(valid, prod, 0)
        gsnr = jnp.sum(prod, axis=-1)
        node_ok = jnp.any(valid, axis=-1) & (gsigma > 0)
        scale = jnp.where(node_ok, (4.0 * jnp.float32(delta_f)) / gsigma, 0)
        gsnr = gsnr * scale
        power = jnp.where(node_ok, gsnr.real * gsnr.real + gsnr.imag * gsnr.imag, 0)
        dof = jnp.sum(node_ok, axis=-1).astype(jnp.float32) * jnp.float32(2)
        total = jnp.sum(power, axis=-1)
        chisq = jnp.where(dof > 0, total / dof, jnp.float32(1))
        return chisq.astype(jnp.float32)

    return kernel


def sg_chisq_batch(stilde, tiles, k_idx, gsigma, times, delta_f):
    """Batched sine-Gaussian chi-squared.

    Returns the mean tile SNR squared, shape ``(batch,)``, on the device of
    ``stilde``. A row with no valid Q nodes is 1.
    """
    jax, jnp = _import_jax()
    stilde = jnp.asarray(stilde, dtype=jnp.complex64)
    tiles = jnp.asarray(tiles, dtype=jnp.complex64)
    if tiles.ndim == 2:
        tiles = tiles[None, :, :]
    k_idx = jnp.asarray(k_idx, dtype=jnp.int32)
    if k_idx.ndim == 2:
        k_idx = k_idx[None, :, :]
    gsigma = jnp.asarray(gsigma, dtype=jnp.float32)
    if gsigma.ndim == 1:
        gsigma = gsigma[None, :]
    times = jnp.asarray(times, dtype=jnp.float32).reshape(-1)
    delta_f = jnp.asarray(delta_f, dtype=jnp.float32)
    with jax.enable_x64(True):
        return _sg_kernel()(stilde, tiles, k_idx, gsigma, times, delta_f)


@functools.lru_cache(maxsize=None)
def _selected_sg_kernel():
    jax, jnp = _import_jax()

    @jax.jit
    def kernel(stilde, tiles, k_idx, gsigma, epochs, indices, rows, delta_f, dt):
        times = epochs[rows] + dt * indices[rows].astype(jnp.float32)
        return _sg_kernel()(
            stilde, tiles[rows], k_idx[rows], gsigma[rows], times, delta_f)

    return kernel


def sg_chisq_selected(stilde, bundle, epochs, indices, rows, dt):
    """Gather only activated candidates and evaluate their tiles in one JIT."""
    jax, jnp = _import_jax()
    with jax.enable_x64(True):
        return _selected_sg_kernel()(
            stilde, bundle['tiles'], bundle['k_idx'], bundle['gsigma'], epochs,
            indices, jnp.asarray(rows, dtype=jnp.int32),
            numpy.float32(bundle['delta_f']), numpy.float32(dt))


@functools.lru_cache(maxsize=None)
def _sg_points_kernel():
    jax, jnp = _import_jax()

    @jax.jit
    def kernel(stilde, tiles, k_idx, gsigma, epochs, indices, rows, delta_f, dt):
        times = epochs[rows] + dt * indices.astype(jnp.float32)
        return _sg_kernel()(
            stilde, tiles[rows], k_idx[rows], gsigma[rows], times, delta_f)

    return kernel


def sg_chisq_candidate_points(stilde, bundle, epochs, rows, indices, dt):
    """Sine-Gaussian veto at arbitrary offline peaks in a single dispatch."""
    jax, jnp = _import_jax()
    with jax.enable_x64(True):
        return _sg_points_kernel()(
            stilde, bundle['tiles'], bundle['k_idx'], bundle['gsigma'], epochs,
            jnp.asarray(indices, dtype=jnp.int32), jnp.asarray(rows, dtype=jnp.int32),
            numpy.float32(bundle['delta_f']), numpy.float32(dt))


def _epoch(template):
    epoch = template.epoch
    if epoch is None:
        return 0.0
    return float(epoch)


def prepare_sg_bundle(sg, templates, psd, stilde_length, stilde_delta_f, bin_edges):
    """Cache sine-Gaussian tiles for one duration group and one PSD.

    Parameters
    ----------
    bin_edges : sequence
        One unpadded integer edge array per template. Entries may be None
        when the template is not in the sine-Gaussian location map.

    Returns
    -------
    bundle : dict
        Numpy arrays ``tiles``, ``k_idx``, ``gsigma``, and ``compute``.
        ``compute`` is false when the template hash is absent or a tile
        would pass 0.9 Nyquist; those rows stay 1 in the live filter.
    """
    if not getattr(sg, 'do', False):
        return None
    nbatch = len(templates)
    per_template = []
    max_q = 1
    max_k = 1
    for template, edges in zip(templates, bin_edges):
        tiles = _tiles_for_template(
            sg, template, psd, stilde_length, stilde_delta_f, edges)
        per_template.append(tiles)
        if tiles and tiles != 'nyquist':
            max_q = max(max_q, len(tiles))
            max_k = max(max_k, max(len(tile[0]) for tile in tiles))
    tiles = numpy.zeros((nbatch, max_q, max_k), dtype=numpy.complex64)
    k_idx = numpy.full((nbatch, max_q, max_k), -1, dtype=numpy.int32)
    gsigma = numpy.ones((nbatch, max_q), dtype=numpy.float32)
    compute = numpy.zeros(nbatch, dtype=numpy.bool_)
    for i, spec in enumerate(per_template):
        if not spec or spec == 'nyquist':
            continue
        compute[i] = True
        for q_i, (tile, kmin, _kmax, sigma_val, _df) in enumerate(spec):
            width = len(tile)
            tiles[i, q_i, :width] = tile
            k_idx[i, q_i, :width] = numpy.arange(kmin, kmin + width, dtype=numpy.int32)
            gsigma[i, q_i] = numpy.float32(sigma_val)
    return {
        'tiles': tiles,
        'k_idx': k_idx,
        'gsigma': gsigma,
        'compute': compute,
        'delta_f': float(templates[0].delta_f),
    }


def _tiles_for_template(sg, template, psd, stilde_length, stilde_delta_f, edges):
    """Return tile slices, ``'nyquist'``, or None (chi-squared stays 1)."""
    if template.params.template_hash not in sg.params:
        return None
    values = sg.params[template.params.template_hash].split(',')
    if edges is None or len(edges) < 3:
        return None
    qwindow = 50
    fstep = (int(edges[-2]) - int(edges[-3]))
    fpeak = (int(edges[-2]) + fstep) * float(template.delta_f)
    fstop = float(stilde_length) * float(stilde_delta_f) * 0.9
    kmin_tmpl = int(template.f_lower / psd.delta_f)
    tiles = []
    for descr in values:
        quality, offset = descr.split('-')
        quality, offset = float(quality), float(offset)
        fcen = fpeak + offset
        flow = max(kmin_tmpl * float(template.delta_f), fcen - qwindow)
        fhigh = fcen + qwindow
        if fhigh > fstop:
            return 'nyquist'
        kmin = int(flow / template.delta_f)
        kmax = int(fhigh / template.delta_f)
        gtem = sinegauss.fd_sine_gaussian(
            1.0, quality, fcen, flow,
            len(template) * float(template.delta_f),
            template.delta_f).astype(numpy.complex64)
        gsigma = sigma(gtem, psd=psd,
                       low_frequency_cutoff=flow,
                       high_frequency_cutoff=fhigh)
        tile = numpy.asarray(gtem[kmin:kmax], dtype=numpy.complex64)
        tiles.append((tile, kmin, kmax, float(gsigma), float(gtem.delta_f)))
    return tiles


def sg_gate_mask(abs_snr, reduced, snr_threshold):
    """True where newsnr is at least the sine-Gaussian activation threshold."""
    nsnr = ranking.newsnr(numpy.asarray(abs_snr, dtype=numpy.float64),
                          numpy.asarray(reduced, dtype=numpy.float64))
    return numpy.asarray(nsnr) >= float(snr_threshold)


def sg_chisq_values(sg, stilde, template, psd, snrv, snr_norm,
                    bchisq, bchisq_dof, indices, bundle=None, bins=None):
    """Sine-Gaussian chi-squared, same control flow as ``SingleDetSGChisq``.

    Returns an array of ones when the template hash is not in the location
    map, when sine-Gaussian chi-squared is disabled, or when a tile would
    extend above 0.9 Nyquist. Otherwise the mean tile SNR^2. Indices that
    pass the newsnr gate are evaluated together.
    """
    snrv = numpy.asarray(snrv)
    indices = numpy.asarray(indices)
    n = len(snrv)
    if not getattr(sg, 'do', False):
        return None
    if bins is None:
        bins = sg.cached_chisq_bins(template, psd)
    if bundle is None:
        bundle = prepare_sg_bundle(
            sg, [template], psd, len(stilde), stilde.delta_f, [numpy.asarray(bins)])
    if template.params.template_hash not in sg.params or not bundle['compute'][0]:
        return numpy.ones(n, dtype=numpy.float32)

    chisq = numpy.ones(n, dtype=numpy.float32)
    snr = numpy.abs(numpy.asarray(snrv) * float(snr_norm))
    reduced = (numpy.asarray(bchisq, dtype=numpy.float64)
               / numpy.asarray(bchisq_dof, dtype=numpy.float64))
    active = numpy.flatnonzero(sg_gate_mask(snr, reduced, sg.snr_threshold))
    if active.size == 0:
        return chisq

    ntime = (len(template) - 1) * 2
    dt_sample = 1.0 / (ntime * float(template.delta_f))
    times = _epoch(template) + dt_sample * indices[active].astype(numpy.float64)
    tiles = numpy.repeat(bundle['tiles'][:1], active.size, axis=0)
    k_idx = numpy.repeat(bundle['k_idx'][:1], active.size, axis=0)
    gsigma = numpy.repeat(bundle['gsigma'][:1], active.size, axis=0)
    data = numpy.asarray(stilde, dtype=numpy.complex64)
    values = sg_chisq_batch(
        data, tiles, k_idx, gsigma, times.astype(numpy.float32), template.delta_f)
    chisq[active] = numpy.asarray(values, dtype=numpy.float32)
    return chisq
