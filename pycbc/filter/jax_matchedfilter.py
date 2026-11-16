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

import logging

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
from pycbc.vetoes.jax_chisq import reduced_power_chisq, sg_chisq_values

logger = logging.getLogger('pycbc.filter.jax_matchedfilter')

# Only replace the bank series when the generated waveform is close enough
# that a weak trigger cannot move by more than the 1e-4 relative SNR gate.
_WAVEFORM_MISMATCH_LIMIT = 1.0e-6


def _import_jax():
    import jax
    import jax.numpy as jnp
    return jax, jnp


def configure_jax(platform='cpu', device_id=0):
    """Select the JAX platform before the first use in this process.

    ``device_id`` is the CUDA device index from ``--processing-device-id``.
    On GPU it is applied with ``CUDA_VISIBLE_DEVICES`` when JAX has not been
    imported yet, and with ``jax.device_put`` either way.

    Returns
    -------
    device : jax device
    gpu_available : bool
        False when a GPU was requested but JAX has no CUDA device.
    """
    import os
    import sys

    platform = (platform or 'cpu').lower()
    if platform not in ('cpu', 'gpu'):
        raise ValueError("jax platform must be 'cpu' or 'gpu', got %s" % platform)

    if 'jax' not in sys.modules:
        if platform == 'cpu':
            os.environ['JAX_PLATFORMS'] = 'cpu'
        elif device_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(int(device_id))

    jax, _jnp = _import_jax()
    try:
        jax.config.update('jax_enable_x64', False)
    except AttributeError:
        pass

    gpu_devices = [d for d in jax.devices() if getattr(d, 'platform', '') == 'gpu']
    if platform == 'gpu':
        if not gpu_devices:
            logger.warning('JAX GPU requested but jax.devices() has no CUDA device')
            return jax.devices('cpu')[0], False
        index = int(device_id or 0)
        if index >= len(gpu_devices):
            logger.warning('processing-device-id %s is outside the visible GPU '
                           'list (length %s); using GPU 0', index, len(gpu_devices))
            index = 0
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
    """Same duration grouping as ``LiveBatchMatchedFilter``."""
    durations = numpy.array([1.0 / t.delta_f for t in templates])
    lsort = durations.argsort()
    durations = durations[lsort]
    templates = [templates[li] for li in lsort]

    _uniq, counts = numpy.unique(durations, return_counts=True)
    tsamples = [(len(t) - 1) * 2 for t in templates]
    grabs = maxelements / numpy.unique(tsamples)

    chunks = numpy.array([])
    num = 0
    for count, grab in zip(counts, grabs):
        chunks = numpy.append(chunks, numpy.arange(num, count + num, grab))
        chunks = numpy.append(chunks, [count + num])
        num += count
    chunks = numpy.unique(chunks).astype(numpy.uint32)
    sizes = chunks[1:] - chunks[0:-1]
    return templates, sizes


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


def batched_correlate_ifft(series, stilde, device):
    """Unnormalized complex SNR from the live JAX correlate and IFFT.

    Same operation as ``JaxLiveBatchMatchedFilter._correlate_and_peak``:
    ``q`` is zero except ``q[:, :nfreq] = conj(h) * s``, and the time series
    is ``ifft(q) * N`` (FFTW backward convention).

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
    q = jnp.zeros((nbatch, ntime), dtype=jnp.complex64)
    q = q.at[:, :flen].set(jnp.conj(h) * s[None, :])
    snr_ts = jnp.fft.ifft(q, axis=-1) * jnp.asarray(ntime, dtype=jnp.float32)
    snr_ts = snr_ts.astype(jnp.complex64)
    return snr_ts, q


class JaxLiveBatchMatchedFilter(object):
    """Batched matched filter and signal-consistency tests on JAX."""

    def __init__(self, templates, snr_threshold, chisq_bins, sg_chisq,
                 maxelements=2**27,
                 snr_abort_threshold=None,
                 newsnr_threshold=None,
                 max_triggers_in_batch=None,
                 jax_platform='cpu',
                 processing_device_id=0,
                 device_id=None):
        if device_id is not None:
            processing_device_id = device_id
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

        templates, self.chunks = _duration_chunks(templates, maxelements)
        self.tgroups = []
        self._series = []
        self._lengths = []
        offset = 0
        for count in self.chunks:
            count = int(count)
            group = templates[offset:offset + count]
            offset += count
            self.tgroups.append(group)
            series = self._prepare_group(group)
            self._series.append(series)
            self._lengths.append((len(group[0]) - 1) * 2)

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

    def combine_results(self, results):
        result = {}
        for key in results[0]:
            result[key] = numpy.concatenate([r[key] for r in results])
        return result

    def process_data(self, data_reader):
        self.set_data(data_reader)
        return self.process_all()

    def process_all(self):
        results = []
        veto_info = []
        while 1:
            result, veto = self._process_batch()
            if result is False:
                return False
            if result is None:
                break
            results.append(result)
            veto_info += veto

        result = self.combine_results(results)
        if self.max_triggers_in_batch:
            sort = result['snr'].argsort()[::-1][:self.max_triggers_in_batch]
            for key in result:
                result[key] = result[key][sort]
            tmp = veto_info
            veto_info = [tmp[i] for i in sort]
        return self._process_vetoes(result, veto_info)

    def _correlate_and_peak(self, series, stilde, valid_start, valid_end):
        """Batched ``conj(h) * s`` and unnormalized complex64 IFFT.

        Only the first ``len(h)`` bins of the length-N buffer are written.
        The transform is ``ifft(q) * N``, the FFTW backward convention.
        """
        jax, jnp = _import_jax()
        snr_ts, q = batched_correlate_ifft(series, stilde, self.device)
        segment = snr_ts[:, valid_start:valid_end]
        power = segment.real * segment.real + segment.imag * segment.imag
        # First index with a strictly larger modulus squared.
        local = jnp.argmax(power, axis=-1)
        rows = jnp.arange(snr_ts.shape[0])
        snrv = segment[rows, local]
        q_host = numpy.asarray(q, dtype=numpy.complex64)
        local_host = numpy.asarray(local, dtype=numpy.int64)
        snrv_host = numpy.asarray(snrv, dtype=numpy.complex64)
        return q_host, local_host, snrv_host

    def _process_batch(self):
        if self.block_id == len(self.tgroups):
            return None, None

        group = self.tgroups[self.block_id]
        series = self._series[self.block_id]
        psize = self._lengths[self.block_id]
        stilde = self.data.overwhitened_data(group[0].delta_f)
        psd = stilde.psd

        valid_end = int(psize - self.data.trim_padding)
        valid_start = int(valid_end - self.data.blocksize * self.data.sample_rate)
        q, local, snrv = self._correlate_and_peak(
            series, stilde, valid_start, valid_end)
        self.block_id += 1

        ntmpl = len(group)
        snr = numpy.zeros(ntmpl, dtype=numpy.complex64)
        time = numpy.zeros(ntmpl, dtype=numpy.float64)
        templates = numpy.zeros(ntmpl, dtype=numpy.uint64)
        sigmasq = numpy.zeros(ntmpl, dtype=numpy.float32)
        time[:] = self.data.start_time

        result = {}
        tkeys = group[0].params.dtype.names
        for key in tkeys:
            result[key] = []
        veto_info = []

        i = 0
        for j, htilde in enumerate(group):
            if hasattr(htilde, 'time_offset') and 'time_offset' not in result:
                result['time_offset'] = []

            l = int(local[j]) + valid_start
            sgm = htilde.sigmasq(psd)
            norm = 4.0 * htilde.delta_f / (sgm ** 0.5)
            snr_i = snrv[j]
            s = abs(snr_i) * norm
            if s < self.snr_threshold:
                continue

            time[i] += float(l - valid_start) / self.data.sample_rate
            if self.snr_abort_threshold is not None and s > self.snr_abort_threshold:
                logger.info("We are seeing some *really* high SNRs, let's "
                            "assume they aren't signals and just give up")
                return False, []

            veto_info.append((numpy.array([snr_i], dtype=numpy.complex64),
                              float(norm), l, htilde, stilde, q[j], sgm))
            snr[i] = snr_i * norm
            sigmasq[i] = sgm
            templates[i] = htilde.id
            if not hasattr(htilde, 'dict_params'):
                htilde.dict_params = {}
                for key in tkeys:
                    htilde.dict_params[key] = htilde.params[key]
            for key in tkeys:
                result[key].append(htilde.dict_params[key])
            if hasattr(htilde, 'time_offset'):
                result['time_offset'].append(htilde.time_offset)
            i += 1

        result['snr'] = abs(snr[0:i])
        result['coa_phase'] = numpy.angle(snr[0:i])
        result['end_time'] = time[0:i]
        result['template_id'] = templates[0:i]
        result['sigmasq'] = sigmasq[0:i]
        for key in tkeys:
            result[key] = numpy.array(result[key])
        if 'time_offset' in result:
            result['time_offset'] = numpy.array(result['time_offset'])
        return result, veto_info

    def _process_vetoes(self, results, veto_info):
        chisq = numpy.array(numpy.zeros(len(veto_info)), numpy.float32, ndmin=1)
        dof = numpy.array(numpy.zeros(len(veto_info)), numpy.uint32, ndmin=1)
        sg_chisq = numpy.array(numpy.zeros(len(veto_info)), numpy.float32, ndmin=1)
        results['chisq'] = chisq
        results['chisq_dof'] = dof
        results['sg_chisq'] = sg_chisq

        keep = []
        for i, (snrv, norm, l, htilde, stilde, corr, _sgm) in enumerate(veto_info):
            if self.power_chisq.do:
                bins = self.power_chisq.cached_chisq_bins(htilde, stilde.psd)
                reduced, dof_vec, raw = reduced_power_chisq(
                    corr, snrv, norm, bins, [l])
                chisq[i] = reduced[0]
                dof[i] = dof_vec[0]
                c = raw
                d = dof_vec
            else:
                c = numpy.zeros(1, dtype=numpy.float32)
                d = numpy.ones(1, dtype=numpy.uint32)

            sgv = sg_chisq_values(self.sg_chisq, stilde, htilde, stilde.psd,
                                  snrv, norm, c, d, [l])
            if sgv is not None:
                sg_chisq[i] = sgv[0]

            if self.newsnr_threshold:
                newsnr = ranking.newsnr(results['snr'][i], chisq[i])
                if newsnr >= self.newsnr_threshold:
                    keep.append(i)

        if self.newsnr_threshold:
            keep = numpy.array(keep, dtype=numpy.uint32)
            for key in results:
                results[key] = results[key][keep]
        return results
