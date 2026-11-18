"""JAX batched live matched filter.

Same constructor arguments and ``process_data`` return contract as
``LiveBatchMatchedFilter``. Rank 0 and the strain buffer stay on the CPU
scheme; this class does not use the ``@schemed`` pointer ``BatchCorrelator``
or the CUDA/cuPy processing scheme.

Templates are generated once (TaylorF2 / SPAtmplt and IMRPhenomD) and kept
as complex64 device arrays. If a generated waveform is not close enough to
the frequency series the CPU bank produced for a 1e-4 SNR comparison, the
filter uses that CPU series instead. The comparison is always logged.
"""

import functools
import logging
import time as _time

import numpy

from pycbc.events import ranking
from pycbc.waveform.jax_taylorf2 import (
    batched_spa,
    grid_tables,
    taylorf2_phasing_coefficients,
    zero_lag_match,
)
from pycbc.waveform.jax_phenom import (
    compare_phenomd_to_reference,
    phenomd_implementation,
)
from pycbc.vetoes.jax_chisq import (
    power_chisq_selected,
    prepare_sg_bundle,
    reduced_from_raw,
    sg_chisq_batch,
    sg_chisq_selected,
    sg_gate_mask,
)

logger = logging.getLogger('pycbc.filter.jax_matchedfilter')

# Only replace the bank series when the generated waveform is close enough
# that a weak trigger cannot move by more than the 1e-4 relative SNR gate.
_WAVEFORM_MISMATCH_LIMIT = 1.0e-6


def _import_jax():
    import jax
    import jax.numpy as jnp
    return jax, jnp


def configure_jax(platform='cpu', device_id=0):
    """Select a visible device without modifying scheduler visibility.

    GPU requests fail explicitly; CPU and out-of-range GPU fallbacks can
    silently put multiple filtering ranks on one device and are prohibited.
    """
    platform = (platform or 'cpu').lower()
    if platform not in ('cpu', 'gpu'):
        raise ValueError("jax platform must be 'cpu' or 'gpu', got %s" % platform)

    jax, _jnp = _import_jax()
    try:
        jax.config.update('jax_enable_x64', False)
    except AttributeError:
        pass

    if platform == 'gpu':
        gpu_devices = jax.devices('gpu')
        index = int(device_id or 0)
        if not 0 <= index < len(gpu_devices):
            raise ValueError(f'GPU index {index} outside {len(gpu_devices)} visible devices')
        return gpu_devices[index], True
    return jax.devices('cpu')[0], True


def _approximant_name(template):
    name = getattr(template, 'approximant', None)
    if name is None and hasattr(template, 'params'):
        if 'approximant' in template.params.dtype.names:
            name = template.params['approximant']
    if name is None:
        return ''
    if hasattr(name, 'decode'):
        name = name.decode()
    return str(name)


def _param(template, name, default=None):
    params = template.params
    if name in params.dtype.names:
        value = params[name]
        if hasattr(value, 'decode'):
            value = value.decode()
        return value
    return default


def _duration_chunks(templates, maxelements):
    """Group equal FFT grids with integer batch sizes and no empty chunks."""
    if not templates or maxelements < 1:
        raise ValueError('nonempty bank and positive maxelements required')
    templates = sorted(templates, key=lambda t: (1 / t.delta_f, len(t)))
    sizes = []
    offset = 0
    while offset < len(templates):
        end = offset + 1
        key = (templates[offset].delta_f, len(templates[offset]))
        while end < len(templates) and (templates[end].delta_f, len(templates[end])) == key:
            end += 1
        batch = max(1, int(maxelements) // ((len(templates[offset]) - 1) * 2))
        sizes.extend(min(batch, end - i) for i in range(offset, end, batch))
        offset = end
    return templates, numpy.asarray(sizes, dtype=numpy.uint32)


def resolve_filter_backend(args, rank, size):
    """Choose ``cpu`` or ``jax`` for this MPI rank.

    Rank 0 is always ``cpu``. ``--rank-backend`` is a comma-separated list
    cycled across filtering ranks. Without it, every filtering rank uses
    ``--filter-backend``.
    """
    del size
    if rank <= 0:
        return 'cpu'
    listed = getattr(args, 'rank_backend', None)
    if listed:
        parts = [part.strip() for part in listed.split(',') if part.strip()]
        bad = [part for part in parts if part not in ('cpu', 'jax')]
        if not parts or bad:
            raise ValueError(
                "--rank-backend must be a comma-separated list of 'cpu' and 'jax'"
            )
        return parts[(rank - 1) % len(parts)]
    backend = getattr(args, 'filter_backend', 'cpu') or 'cpu'
    if backend not in ('cpu', 'jax'):
        raise ValueError("--filter-backend must be 'cpu' or 'jax'")
    return backend


def _q_and_unnormalized(h, s, q):
    """Batched ``conj(h) * s`` and unnormalized ``ifft * N``.

    ``q`` is zero except ``q[:, :nfreq] = conj(h) * s``. ``N`` is the time
    length. Called from jitted surface code so the FFT stays on device.
    """
    _jax, jnp = _import_jax()
    flen = h.shape[1]
    ntime = q.shape[1]
    q = q.at[:, :].set(jnp.complex64(0))
    q = q.at[:, :flen].set(jnp.conj(h) * s[None, :])
    unnorm = jnp.fft.ifft(q, axis=-1) * jnp.asarray(ntime, dtype=jnp.float32)
    return q, unnorm.astype(jnp.complex64)


@functools.lru_cache(maxsize=None)
def _correlate_kernel():
    """Unnormalized correlate used by the diagnostic helper."""
    jax, _jnp = _import_jax()

    @jax.jit(donate_argnums=(2,))
    def kernel(h, s, q):
        return _q_and_unnormalized(h, s, q)

    return kernel


@functools.lru_cache(maxsize=None)
def _surface_kernel(valid_len, normalize=True):
    """One compiled surface for a fixed valid-window length.

    Input shapes select the duration group. ``valid_start`` is a traced
    scalar, so a new 8 s block does not compile again. ``q`` and ``snr``
    are donated and reused as the frequency product and the normalized
    surface. With normalize=False the returned buffer is the raw IFFT;
    live peaks are normalized only after reduction, avoiding a full-size
    normalization/write and following native peak selection before scaling.
    """
    jax, jnp = _import_jax()
    valid_len = int(valid_len)

    @jax.jit(donate_argnums=(3, 4))
    def kernel(h, s, norms, q, snr, valid_start):
        q, unnorm = _q_and_unnormalized(h, s, q)
        # ρ(t, φ, θ). The per-template norm is real and positive, so the
        # peak index matches the unnormalized series. Chi-squared still
        # uses the unnormalized sample, as the CPU veto does.
        surface = snr.at[:, :].set(
            (unnorm * norms[:, None]).astype(jnp.complex64) if normalize else unnorm)
        start = (jnp.int32(0), valid_start)
        window = (h.shape[0], valid_len)
        segment = jax.lax.dynamic_slice(surface, start, window)
        unnorm_seg = jax.lax.dynamic_slice(unnorm, start, window)
        power = segment.real * segment.real + segment.imag * segment.imag
        local = jnp.argmax(power, axis=-1)
        rows = jnp.arange(h.shape[0])
        peak_unnorm = unnorm_seg[rows, local]
        peak = segment[rows, local] if normalize else peak_unnorm * norms

        return q, surface, peak, peak_unnorm, local + valid_start, jnp.abs(peak)

    return kernel


@functools.lru_cache(maxsize=None)
def _cpu_product_kernel(ntime):
    jax, jnp = _import_jax()

    @jax.jit
    def kernel(h, s):
        return jnp.pad(jnp.conj(h) * s[None, :],
                       ((0, 0), (0, ntime - h.shape[1])))

    return kernel


@functools.lru_cache(maxsize=None)
def _cpu_ifft_kernel():
    jax, jnp = _import_jax()

    @jax.jit
    def kernel(q):
        return (jnp.fft.ifft(q, axis=-1)
                * jnp.float32(q.shape[1])).astype(jnp.complex64)

    return kernel


@functools.lru_cache(maxsize=None)
def _peak_kernel(valid_len):
    jax, jnp = _import_jax()

    @jax.jit
    def kernel(unnorm, norms, valid_start):
        part = jax.lax.dynamic_slice(
            unnorm, (jnp.int32(0), valid_start), (unnorm.shape[0], valid_len))
        power = part.real * part.real + part.imag * part.imag
        # Positive normalization does not change the maximizing time. This
        # follows the CPU filter, which finds its maximum before normalizing.
        local = jnp.argmax(power, axis=-1)
        raw = part[jnp.arange(unnorm.shape[0]), local]
        peak = raw * norms
        return peak, raw, local + valid_start, jnp.abs(peak)

    return kernel


@functools.lru_cache(maxsize=None)
def _normalize_surface_kernel():
    jax, _jnp = _import_jax()

    @jax.jit
    def kernel(unnorm, norms):
        return unnorm * norms[:, None]

    return kernel


@functools.lru_cache(maxsize=None)
def _normalized_window_kernel(valid_len, already_normalized):
    jax, jnp = _import_jax()

    @jax.jit
    def kernel(series, norms, valid_start):
        window = jax.lax.dynamic_slice(
            series, (jnp.int32(0), valid_start), (series.shape[0], valid_len))
        return window if already_normalized else window * norms[:, None]

    return kernel


def _cpu_surface(h, s, norms, valid_start, valid_len):
    """Separate JIT stages for the CPU runtime's FFT scheduling.

    On the tested XLA CPU runtime, a combined product/FFT returning both
    arrays costs substantially more than the same stages dispatched
    separately. Keep this CPU-specific; GPU execution uses the fused kernel.
    Only normalize the retained peaks here. Diagnostics normalize the full
    surface lazily when ``state.snr`` is requested.
    """
    ntime = (h.shape[1] - 1) * 2
    q = _cpu_product_kernel(ntime)(h, s)
    unnorm = _cpu_ifft_kernel()(q)
    peaks = _peak_kernel(int(valid_len))(unnorm, norms, valid_start)
    return (q, unnorm, *peaks)


def batched_correlate_ifft(series, stilde, device):
    """Unnormalized complex SNR from the live JAX correlate and IFFT.

    ``q`` is zero except ``q[:, :nfreq] = conj(h) * s``, and the time series
    is ``ifft(q) * N`` (FFTW backward convention). The live filter multiplies
    by the per-template norm inside ``_surface_kernel`` and does not use
    this helper.

    Parameters
    ----------
    series : ndarray
        Complex64 templates, shape ``(nbatch, nfreq)``.
    stilde : array
        Overwhitened frequency series. The first ``nfreq`` bins are used.
    device : jax device
        Device ``series`` and ``stilde`` are placed on.

    Returns
    -------
    snr_ts : jax array
        Complex64, shape ``(nbatch, ntime)``.
    q : jax array
        Complex64 frequency-domain product, shape ``(nbatch, ntime)``.
    """
    jax, jnp = _import_jax()
    h = jnp.asarray(series, dtype=jnp.complex64)
    strain = numpy.asarray(stilde, dtype=numpy.complex64)
    if strain.shape[0] < h.shape[1]:
        raise ValueError(
            "overwhitened data is shorter than the template "
            "(%s < %s)" % (strain.shape[0], h.shape[1]))
    s = jnp.asarray(strain[:h.shape[1]], dtype=jnp.complex64)
    h = jax.device_put(h, device)
    s = jax.device_put(s, device)
    nbatch, flen = h.shape
    ntime = (flen - 1) * 2
    q = jax.device_put(jnp.zeros((nbatch, ntime), dtype=jnp.complex64), device)
    return _correlate_kernel()(h, s, q)


class _DurationGroup(object):
    """Device templates, reused surface buffers, and PSD-keyed weights."""

    def __init__(self, series, templates, device):
        jax, jnp = _import_jax()
        self.templates = templates
        self.device = device
        self.nbatch, self.flen = series.shape
        self.ntime = (self.flen - 1) * 2
        self.delta_f = float(templates[0].delta_f)
        self.h = jax.device_put(jnp.asarray(series, dtype=jnp.complex64), device)
        self.q = jax.device_put(
            jnp.zeros((self.nbatch, self.ntime), dtype=jnp.complex64), device)
        self.snr = jax.device_put(
            jnp.zeros((self.nbatch, self.ntime), dtype=jnp.complex64), device)
        self.unnorm = None
        self.psd_key = None
        self.sigmasq = None
        self.norms = None
        self.bins = None
        self.num_bins = None
        self.sg = None
        self.strain = None
        epochs = numpy.array([
            0.0 if template.epoch is None else float(template.epoch)
            for template in templates], dtype=numpy.float32)
        self.epochs = jax.device_put(jnp.asarray(epochs), device)

    @property
    def snr(self):
        """Normalized surface, materialized only for diagnostics/offline use."""
        if self._snr is None and self.unnorm is not None:
            self._snr = _normalize_surface_kernel()(self.unnorm, self.norms)
            # Once the full surface is requested, retain that representation
            # rather than both full-size time arrays. Peak vetoes retain their
            # raw scalar samples separately.
            self.unnorm = None
        return self._snr

    @snr.setter
    def snr(self, value):
        self._snr = value

    def snr_window(self, start, length):
        """Normalize/copy only the valid window needed by offline clustering."""
        start, length = int(start), int(length)
        if start < 0 or length <= 0 or start + length > self.ntime:
            raise ValueError('SNR window is outside the filter buffer')
        normalized = self.unnorm is None
        series = self._snr if normalized else self.unnorm
        return _normalized_window_kernel(length, normalized)(
            series, self.norms, numpy.int32(start))

    def ensure_psd(self, psd, stilde, power_chisq, sg_chisq):
        """Recompute sigmasq, chi-squared edges, and tiles when the PSD changes."""
        key = (id(psd), int(len(stilde)))
        if key == self.psd_key and self.norms is not None:
            return
        jax, jnp = _import_jax()
        sigmasq = numpy.empty(self.nbatch, dtype=numpy.float64)
        edges = []
        sg_edges = []
        width = 1
        for i, template in enumerate(self.templates):
            sigmasq[i] = float(template.sigmasq(psd))
            if power_chisq.do:
                row = numpy.asarray(
                    power_chisq.cached_chisq_bins(template, psd),
                    dtype=numpy.int32)
            else:
                row = numpy.zeros(1, dtype=numpy.int32)
            edges.append(row)
            width = max(width, int(row.size))
            if getattr(sg_chisq, 'do', False):
                sg_edges.append(numpy.asarray(
                    sg_chisq.cached_chisq_bins(template, psd),
                    dtype=numpy.int32))
            else:
                sg_edges.append(None)
        bins = numpy.zeros((self.nbatch, width), dtype=numpy.int32)
        num_bins = numpy.zeros(self.nbatch, dtype=numpy.int32)
        for i, row in enumerate(edges):
            bins[i, :row.size] = row
            if row.size < width:
                bins[i, row.size:] = row[-1]
            num_bins[i] = max(int(row.size) - 1, 0)
        norms = (4.0 * self.delta_f / numpy.sqrt(sigmasq)).astype(numpy.float32)
        self.sigmasq = sigmasq
        self.num_bins = num_bins
        self.power_stop = max(1, min(self.flen, int(bins.max())))
        self.bins = jax.device_put(jnp.asarray(bins), self.device)
        self.norms = jax.device_put(jnp.asarray(norms), self.device)
        self.num_bins_d = jax.device_put(jnp.asarray(num_bins, dtype=jnp.float32),
                                         self.device)
        bundle = None
        if getattr(sg_chisq, 'do', False):
            bundle = prepare_sg_bundle(
                sg_chisq, self.templates, psd, len(stilde), stilde.delta_f,
                sg_edges)
        if bundle is None:
            self.sg = None
        else:
            self.sg = {
                'tiles': jax.device_put(jnp.asarray(bundle['tiles']), self.device),
                'k_idx': jax.device_put(jnp.asarray(bundle['k_idx']), self.device),
                'gsigma': jax.device_put(jnp.asarray(bundle['gsigma']), self.device),
                'compute': bundle['compute'],
                'delta_f': bundle['delta_f'],
            }
        self.psd_key = key

    def set_strain(self, stilde):
        """Transfer this block once, without an additional scatter copy."""
        jax, _jnp = _import_jax()
        strain = numpy.asarray(stilde, dtype=numpy.complex64)
        self.strain = jax.device_put(strain, self.device)
        return self.strain


class JaxLiveBatchMatchedFilter(object):
    """Batched matched filter and signal-consistency tests on JAX."""

    def __init__(self, templates, snr_threshold, chisq_bins, sg_chisq,
                 maxelements=2**27,
                 snr_abort_threshold=None,
                 newsnr_threshold=None,
                 max_triggers_in_batch=None,
                 jax_platform='cpu',
                 processing_device_id=0,
                 device_id=None,
                 autotune=False, tuning_candidates=(1, 2, 4, 8, 16, 32, 64),
                 tuning_repeats=5, memory_budget_mb=None,
                 max_batch_latency_ms=None, waveform_source="auto", profile_stages=False):
        if device_id is not None:
            processing_device_id = device_id
        if waveform_source not in ('auto', 'bank'):
            raise ValueError('waveform_source must be auto or bank')
        self.profile_stages = profile_stages
        self.last_profile = {}
        self.waveform_source = waveform_source
        self.snr_threshold = snr_threshold
        self.snr_abort_threshold = snr_abort_threshold
        self.newsnr_threshold = newsnr_threshold
        self.max_triggers_in_batch = max_triggers_in_batch
        self.jax_platform = jax_platform
        self.processing_device_id = processing_device_id

        from pycbc import vetoes
        self.power_chisq = vetoes.SingleDetPowerChisq(chisq_bins, None)
        self.sg_chisq = sg_chisq

        self.device, self.gpu_available = configure_jax(
            jax_platform, processing_device_id)
        self.startup_comparisons = []
        self.autotune = autotune
        self.tuning_candidates = tuple(sorted(set(int(x) for x in tuning_candidates)))
        if not self.tuning_candidates or min(self.tuning_candidates) < 1 or tuning_repeats < 1:
            raise ValueError('positive tuning candidates and repeats required')
        if memory_budget_mb is not None and (not numpy.isfinite(memory_budget_mb) or memory_budget_mb <= 0):
            raise ValueError('memory budget must be positive')
        if max_batch_latency_ms is not None and (not numpy.isfinite(max_batch_latency_ms) or max_batch_latency_ms <= 0):
            raise ValueError('latency limit must be positive')
        self.tuning_repeats = tuning_repeats
        self.memory_budget_mb = memory_budget_mb
        self.max_batch_latency_ms = max_batch_latency_ms
        self.tuning_report = None
        self._tuning = False

        templates, self.chunks = _duration_chunks(templates, maxelements)
        self.tgroups = []
        self._series = []
        self._lengths = []
        self._states = []
        offset = 0
        for count in self.chunks:
            count = int(count)
            group = templates[offset:offset + count]
            offset += count
            self.tgroups.append(group)
            jax, _ = _import_jax()
            with jax.default_device(self.device):
                series = self._prepare_group(group)
            self._series.append(series)
            self._lengths.append((len(group[0]) - 1) * 2)
            if not autotune:
                with jax.default_device(self.device):
                    self._states.append(_DurationGroup(series, group, self.device))

        self._bank_groups = list(zip(self.tgroups, self._series))
        self._log_startup_summary()

    def _prepare_group(self, group):
        """Build the complex64 series actually filtered for one duration group."""
        chosen = []
        for template in group:
            generated, report = self._generate_and_compare(template)
            self.startup_comparisons.append(report)
            if report['use_generated']:
                chosen.append(numpy.asarray(generated, dtype=numpy.complex64))
            else:
                chosen.append(numpy.array(template, dtype=numpy.complex64, copy=True))
        return numpy.stack(chosen, axis=0)

    def _generate_and_compare(self, template):
        """Compare the JAX generator to the bank series and to reference models."""
        name = _approximant_name(template)
        h_ref = numpy.array(template, dtype=numpy.complex64, copy=True)
        if self.waveform_source == 'bank':
            report = dict(approximant=name, template_id=int(template.id),
                          implementation='native-bank', use_generated=False,
                          match_template=1.0, waveform_source='exact native bank')
            logger.info('JAX waveform source: exact native bank, template %s %s',
                        template.id, name)
            return None, report
        distance = 1.0 / __import__('pycbc').DYN_RANGE_FAC
        mass1 = float(_param(template, 'mass1'))
        mass2 = float(_param(template, 'mass2'))
        spin1z = float(_param(template, 'spin1z', 0.0))
        spin2z = float(_param(template, 'spin2z', 0.0))
        f_lower = float(template.f_lower)
        f_final = float(getattr(template, 'end_frequency', 0.0) or 0.0)
        if f_final <= 0.0:
            f_final = (len(template) - 1) * float(template.delta_f)
        phase_order = int(_param(template, 'phase_order', -1))
        spin_order = int(_param(template, 'spin_order', -1))
        f_ref = _param(template, 'f_ref', 0.0)
        report = {
            'approximant': name,
            'template_id': int(getattr(template, 'id', -1)),
            'match_template': float('nan'),
            'match_spa': float('nan'),
            'match_lal_taylorf2': float('nan'),
            'match_lal_phenomd': float('nan'),
            'implementation': '',
            'use_generated': False,
        }

        generated = None
        if name in ('TaylorF2', 'SPAtmplt'):
            coeffs = taylorf2_phasing_coefficients(
                mass1, mass2, spin1z, spin2z,
                phase_order=phase_order, spin_order=spin_order,
                distance=distance)
            kmin = int(f_lower / float(template.delta_f))
            kmax = min(len(template), int(f_final / float(template.delta_f)))
            # Batched evaluation on the shared grid (one row here; groups of
            # templates that share delta_f use the same function).
            kfac, cbrt, logv = grid_tables(kmin, kmax, template.delta_f)
            batched = batched_spa(kfac, cbrt, logv, [coeffs])
            generated = numpy.zeros(len(template), dtype=numpy.complex64)
            if kmax > kmin:
                generated[kmin:kmax] = batched[0]
            # Always compare to spa_tmplt and to LAL TaylorF2, even when the
            # bank series is only one of those two.
            h_spa = self._cpu_spa(
                len(template), template.delta_f, f_lower, f_final,
                mass1, mass2, spin1z, spin2z, distance, phase_order, spin_order)
            report['match_spa'] = zero_lag_match(h_spa, generated)
            h_lal = self._lal_taylorf2(
                len(template), template.delta_f, f_lower, f_final,
                mass1, mass2, spin1z, spin2z, distance, phase_order, spin_order)
            report['match_lal_taylorf2'] = zero_lag_match(h_lal, generated)
            report['match_template'] = zero_lag_match(h_ref, generated)
            report['implementation'] = 'spa'
        elif name == 'IMRPhenomD':
            match, impl, generated = compare_phenomd_to_reference(
                h_ref, len(template), template.delta_f, f_lower, f_ref,
                distance, mass1, mass2, spin1z, spin2z)
            report['match_template'] = match
            report['match_lal_phenomd'] = match
            report['implementation'] = impl
        else:
            report['implementation'] = 'cpu-series'
            report['use_generated'] = False
            logger.info(
                'JAX waveform check template %s approximant %s: no JAX '
                'generator, filtering the CPU bank series',
                report['template_id'], name or 'unknown')
            return None, report

        mismatch = 1.0 - report['match_template']
        report['use_generated'] = bool(
            numpy.isfinite(report['match_template'])
            and mismatch < _WAVEFORM_MISMATCH_LIMIT)
        logger.info(
            'JAX waveform check template %s %s impl=%s '
            'match_template=%.8g match_spa=%.8g match_lal_taylorf2=%.8g '
            'match_lal_phenomd=%.8g using=%s',
            report['template_id'], name, report['implementation'],
            report['match_template'], report['match_spa'],
            report['match_lal_taylorf2'], report['match_lal_phenomd'],
            'jax' if report['use_generated'] else 'cpu-series')
        return generated, report

    @staticmethod
    def _cpu_spa(length, delta_f, f_lower, f_final, mass1, mass2,
                 spin1z, spin2z, distance, phase_order, spin_order):
        from pycbc.types import zeros
        from pycbc.waveform.spa_tmplt import spa_tmplt

        out = zeros(int(length), dtype=numpy.complex64)
        htilde = spa_tmplt(
            out=out, mass1=mass1, mass2=mass2, spin1z=spin1z, spin2z=spin2z,
            phase_order=phase_order, spin_order=spin_order, distance=distance,
            f_lower=float(f_lower), f_final=float(f_final),
            delta_f=float(delta_f))
        return numpy.array(htilde, dtype=numpy.complex64, copy=True)

    @staticmethod
    def _lal_taylorf2(length, delta_f, f_lower, f_final, mass1, mass2,
                      spin1z, spin2z, distance, phase_order, spin_order):
        from pycbc.types import zeros
        from pycbc.waveform import get_waveform_filter

        class _Row(object):
            pass

        row = _Row()
        row.mass1 = mass1
        row.mass2 = mass2
        row.spin1z = spin1z
        row.spin2z = spin2z
        row.spin1x = 0.0
        row.spin1y = 0.0
        row.spin2x = 0.0
        row.spin2y = 0.0
        row.phase_order = phase_order
        row.spin_order = spin_order
        out = zeros(int(length), dtype=numpy.complex64)
        htilde = get_waveform_filter(
            out, row, approximant='TaylorF2', f_lower=float(f_lower),
            f_final=float(f_final), delta_f=float(delta_f), distance=distance)
        return numpy.array(htilde, dtype=numpy.complex64, copy=True)

    def _log_startup_summary(self):
        if not self.startup_comparisons:
            logger.info('JAX waveform check: no templates')
            return
        used = sum(1 for r in self.startup_comparisons if r['use_generated'])
        logger.info(
            'JAX waveform check summary: %d/%d templates use the JAX '
            'generator (phenomd=%s, platform=%s, device=%s)',
            used, len(self.startup_comparisons), phenomd_implementation(),
            self.jax_platform, self.device)

    def set_data(self, data):
        self.data = data
        self.block_id = 0
        # Several chunks can share the same FFT grid. Upload that strain once
        # per block, including serial (one-template-per-chunk) execution.
        self._strain_cache = {}

    def combine_results(self, results):
        result = {}
        for key in results[0]:
            result[key] = numpy.concatenate([r[key] for r in results])
        return result

    def process_data(self, data_reader):
        jax, _ = _import_jax()
        with jax.default_device(self.device):
            if self.autotune and self.tuning_report is None and not self._tuning:
                from .jax_tuning import tune_live_filter
                tune_live_filter(self, data_reader)
            self.last_profile = {}
            self.set_data(data_reader)
            began = _time.perf_counter()
            result = self.process_all()
            if self.profile_stages:
                self.last_profile['filter_total_s'] = _time.perf_counter()-began
                self.last_profile['host_selection_s'] = max(0.,self.last_profile['filter_total_s']-sum(
                    self.last_profile.get(k,0.) for k in ['whitening_transfer_s','psd_weights_s','surface_s','peak_transfer_s','vetoes_s']))
            return result

    def process_all(self):
        results = []
        links = []
        if self.device.platform == 'gpu':
            batches = self._collect_device_batches()
        else:
            batches = iter(self._process_batch, (None, None))
        for result, batch_links in batches:
            if result is False:
                return False
            results.append(result)
            links.extend(batch_links)

        result = self.combine_results(results)
        if self.max_triggers_in_batch:
            sort = result['snr'].argsort()[::-1][:self.max_triggers_in_batch]
            for key in result:
                result[key] = result[key][sort]
            links = [links[i] for i in sort]
        began = _time.perf_counter()
        result = self._process_vetoes(result, links)
        if self.profile_stages:
            self.last_profile['vetoes_s'] = _time.perf_counter()-began
        return result

    def _collect_device_batches(self):
        """Enqueue the bank before downloading its small peak vectors.

        Each duration group owns its surface buffers. No subsequent group
        overwrites a pending peak or correlation used by the vetoes.
        """
        pending = []
        while self.block_id < len(self.tgroups):
            pending.append(self._process_batch(defer_download=True))
        jax, _ = _import_jax()
        began = _time.perf_counter()
        peaks = jax.device_get([(s.peak, s.index, s.abs_snr)
                                for _, s, _ in pending])
        if self.profile_stages:
            self.last_profile['peak_transfer_s'] = _time.perf_counter()-began
        for (group, state, valid_start), host in zip(pending, peaks):
            yield self._select_batch(group, state, valid_start, host)

    def _evaluate_surface(self, state, valid_start, valid_len):
        """Normalized ρ(t, φ, θ) and on-device peaks for one duration group."""
        jax, jnp = _import_jax()
        strain = state.strain
        if strain.shape[0] < state.flen:
            raise ValueError(
                "overwhitened data is shorter than the template "
                "(%s < %s)" % (strain.shape[0], state.flen))
        s = strain[:state.flen]
        start = numpy.int32(valid_start)
        if self.device.platform == 'cpu':
            state.q, state.unnorm, peak, peak_unnorm, index, abs_snr = _cpu_surface(
                state.h, s, state.norms, start, int(valid_len))
            state.snr = None
        else:
            kernel = _surface_kernel(int(valid_len), normalize=False)
            q = state.q
            snr = state.unnorm if state.unnorm is not None else state._snr
            state.q = state._snr = state.unnorm = None
            q, snr, peak, peak_unnorm, index, abs_snr = kernel(
                state.h, s, state.norms, q, snr, start)
            state.q = q
            state.unnorm = snr
        state.peak = peak
        state.peak_unnorm = peak_unnorm
        state.index = index
        state.abs_snr = abs_snr
        return abs_snr

    def _process_batch(self, defer_download=False):
        if self.block_id == len(self.tgroups):
            return None, None

        group = self.tgroups[self.block_id]
        state = self._states[self.block_id]
        psize = self._lengths[self.block_id]
        delta_f = group[0].delta_f
        began = _time.perf_counter()
        if delta_f not in self._strain_cache:
            conditioner = getattr(self.data, 'jax_conditioner', None)
            if conditioner is not None:
                stilde, device_strain = conditioner.overwhitened(delta_f)
            else:
                stilde = self.data.overwhitened_data(delta_f)
                device_strain = state.set_strain(stilde)
            self._strain_cache[delta_f] = (stilde, device_strain)
        stilde, state.strain = self._strain_cache[delta_f]
        if self.profile_stages:
            state.strain.block_until_ready()
            self.last_profile['whitening_transfer_s'] = self.last_profile.get('whitening_transfer_s', 0)+_time.perf_counter()-began
        began = _time.perf_counter()
        state.ensure_psd(stilde.psd, stilde, self.power_chisq, self.sg_chisq)
        if self.profile_stages:
            state.norms.block_until_ready()
            self.last_profile['psd_weights_s'] = self.last_profile.get('psd_weights_s', 0)+_time.perf_counter()-began
        state.stilde = stilde

        valid_end = int(psize - self.data.trim_padding)
        valid_start = int(valid_end - self.data.blocksize * self.data.sample_rate)
        valid_len = int(valid_end - valid_start)
        began = _time.perf_counter()
        abs_snr = self._evaluate_surface(state, valid_start, valid_len)
        if self.profile_stages:
            abs_snr.block_until_ready()
            self.last_profile['surface_s'] = self.last_profile.get('surface_s', 0)+_time.perf_counter()-began
        self.block_id += 1

        if defer_download:
            return group, state, valid_start

        jax, _jnp = _import_jax()
        # A group has few peaks but a large surface. Transfer just the peak
        # vectors together; avoid separate device gathers and synchronizations.
        host_peak, host_index, host_abs = jax.device_get(
            (state.peak, state.index, abs_snr))
        return self._select_batch(group, state, valid_start,
                                  (host_peak, host_index, host_abs))

    def _select_batch(self, group, state, valid_start, host):
        host_peak, host_index, host_abs = host
        state.host_abs_snr = host_abs
        above = host_abs >= numpy.float32(self.snr_threshold)
        rows = numpy.flatnonzero(above)
        peak = numpy.asarray(host_peak[rows], dtype=numpy.complex64)
        index = numpy.asarray(host_index[rows], dtype=numpy.int64)
        if (self.snr_abort_threshold is not None and rows.size
                and float(numpy.abs(peak).max()) > self.snr_abort_threshold):
            logger.info("We are seeing some *really* high SNRs, let's "
                        "assume they aren't signals and just give up")
            return False, []

        nkeep = int(rows.size)
        time = numpy.full(nkeep, self.data.start_time, dtype=numpy.float64)
        time += (index - valid_start) / float(self.data.sample_rate)
        result = {}
        tkeys = group[0].params.dtype.names
        for key in tkeys:
            result[key] = []
        if any(hasattr(template, 'time_offset') for template in group):
            result['time_offset'] = []
        links = []
        for row in rows:
            htilde = group[int(row)]
            links.append((state, int(row)))
            if not hasattr(htilde, 'dict_params'):
                htilde.dict_params = {}
                for key in tkeys:
                    htilde.dict_params[key] = htilde.params[key]
            for key in tkeys:
                result[key].append(htilde.dict_params[key])
            if hasattr(htilde, 'time_offset'):
                result['time_offset'].append(htilde.time_offset)

        result['snr'] = numpy.abs(peak)
        result['coa_phase'] = numpy.angle(peak)
        result['end_time'] = time
        result['template_id'] = numpy.array(
            [group[int(row)].id for row in rows], dtype=numpy.uint64)
        result['sigmasq'] = numpy.asarray(
            state.sigmasq[rows], dtype=numpy.float32)
        for key in tkeys:
            result[key] = numpy.array(result[key])
        if 'time_offset' in result:
            result['time_offset'] = numpy.array(result['time_offset'])
        return result, links

    def _process_vetoes(self, results, links):
        n = len(links)
        chisq = numpy.zeros(n, dtype=numpy.float32)
        dof = numpy.zeros(n, dtype=numpy.uint32)
        sg_chisq = numpy.zeros(n, dtype=numpy.float32)
        results['chisq'] = chisq
        results['chisq_dof'] = dof
        results['sg_chisq'] = sg_chisq
        if n == 0:
            return results

        grouped = {}
        for out_i, (state, row) in enumerate(links):
            grouped.setdefault(id(state), {'state': state, 'out': [], 'row': []})
            grouped[id(state)]['out'].append(out_i)
            grouped[id(state)]['row'].append(row)

        for group in grouped.values():
            state = group['state']
            out = numpy.asarray(group['out'], dtype=numpy.int64)
            row = numpy.asarray(group['row'], dtype=numpy.int32)
            if self.power_chisq.do:
                selected = _bucket_rows(row, state.nbatch)
                raw = power_chisq_selected(
                    state.q, state.peak_unnorm, state.norms, state.bins,
                    state.index, state.num_bins_d, selected, state.power_stop)
                raw = numpy.asarray(raw, dtype=numpy.float32)[:row.size]
                reduced, dof_vec = reduced_from_raw(raw, state.num_bins[row])
            else:
                raw = numpy.zeros(row.size, dtype=numpy.float32)
                reduced = numpy.zeros(row.size, dtype=numpy.float32)
                dof_vec = numpy.ones(row.size, dtype=numpy.uint32)
            chisq[out] = reduced
            dof[out] = dof_vec
            if getattr(self.sg_chisq, 'do', False) and state.sg is not None:
                abs_snr = state.host_abs_snr[row]
                gate = sg_gate_mask(abs_snr, reduced, self.sg_chisq.snr_threshold)
                use = state.sg['compute'][row] & gate
                picked = numpy.ones(row.size, dtype=numpy.float32)
                active = numpy.flatnonzero(use)
                if active.size:
                    selected = _bucket_rows(row[active], state.nbatch)
                    sg_values = self._sg_for_group(state, selected)
                    picked[active] = numpy.asarray(sg_values)[:active.size]
                sg_chisq[out] = picked

        if self.newsnr_threshold:
            newsnr = ranking.newsnr(results['snr'], chisq)
            keep = numpy.flatnonzero(newsnr >= self.newsnr_threshold).astype(numpy.uint32)
            for key in results:
                results[key] = results[key][keep]
        return results

    def _sg_for_group(self, state, rows=None):
        """Sine-Gaussian chi-squared for a group or selected candidate rows."""
        if rows is not None:
            return sg_chisq_selected(
                state.strain, state.sg, state.epochs, state.index, rows,
                1.0 / (state.ntime * state.delta_f))
        _jax, jnp = _import_jax()
        dt = jnp.asarray(
            1.0 / (state.ntime * state.delta_f), dtype=jnp.float32)
        times = state.epochs + dt * state.index.astype(jnp.float32)
        return sg_chisq_batch(
            state.strain, state.sg['tiles'], state.sg['k_idx'],
            state.sg['gsigma'], times, state.sg['delta_f'])


def _bucket_rows(rows, maximum):
    """Pad candidate rows to powers of two to limit JIT shape variants.

    Duplicate the last real row rather than introducing a dummy template;
    callers discard padding. Never pad beyond the duration group's size.
    """
    size = min(maximum, 1 << (int(len(rows)) - 1).bit_length())
    return numpy.pad(rows, (0, size - len(rows)), mode='edge')
