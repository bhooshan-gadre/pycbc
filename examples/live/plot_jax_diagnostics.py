#!/usr/bin/env python
"""Diagnostic plots of the CPU and JAX PyCBC Live matched filters.

Three 8-second valid blocks, one figure each:

1. Gaussian noise colored by the aLIGOZeroDetHighPower PSD.
2. That same noise plus one IMRPhenomD injection.
3. GWOSC H1 strain around GW150914 (downloaded if it is not cached).

The plotted SNR is the live normalization ``4 * df / sqrt(sigmasq)`` times
the unnormalized IFFT (``ifft * N``). Power and sine-Gaussian chi-squared are
the live vetoes. The live path stores those only at the peak, so the curves
use the same veto code on a decimated set of samples.
"""

import os
import sys
import urllib.request

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

_SOURCE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy

import pycbc
import pycbc.psd
from pycbc.filter import LiveBatchMatchedFilter, highpass, sigma
from pycbc.filter.jax_matchedfilter import (
    JaxLiveBatchMatchedFilter,
    batched_correlate_ifft,
)
from pycbc.noise import noise_from_string
from pycbc.psd import interpolate, inverse_spectrum_truncation, welch
from pycbc.scheme import CPUScheme
from pycbc.types import Array, FrequencySeries, TimeSeries
from pycbc.vetoes.chisq import power_chisq_at_points_from_precomputed
from pycbc.vetoes.jax_chisq import reduced_power_chisq, sg_chisq_values
from pycbc.vetoes.sgchisq import SingleDetSGChisq
from pycbc.waveform import get_td_waveform
from pycbc.waveform.bank import LiveFilterBank

SAMPLE_RATE = 2048
F_LOW = 20.0
BLOCK_SECONDS = 8.0
BUFFER_SECONDS = 16.0
TRIM_SECONDS = 0.25
SNR_THRESHOLD = 4.0
CHISQ_BINS = '8'
PSD_NAME = 'aLIGOZeroDetHighPower'
NOISE_GPS = 1272790000.0
NOISE_SECONDS = 40.0
INJECT_INTO_VALID = 5.0
SG_LOCATIONS = ['mtotal>0:10-0']

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'diagnostics')
CACHE = os.path.join(OUT_DIR, 'cache')
GWOSC_NAME = 'H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5'
GWOSC_URL = (
    'https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/' + GWOSC_NAME)
GWOSC_GPS = 1126259447
GWOSC_EVENT = 1126259462.4
# Valid-block start chosen so the merger sits inside the 8 s window and the
# 16 s filter buffer stays inside the 32 s GWOSC file.
REAL_VALID_GPS = 1126259456.0

CPU_COLOR = '#222222'
JAX_COLOR = '#d95f02'
RES_COLOR = '#7570b3'


class BlockReader(object):
    """The slice of StrainBuffer both live filters read."""

    def __init__(self, stilde, start_time, sample_rate, blocksize, trim_padding):
        self._stilde = stilde
        self.start_time = float(start_time)
        self.sample_rate = sample_rate
        self.blocksize = float(blocksize)
        self.trim_padding = int(trim_padding)

    def overwhitened_data(self, delta_f):
        if abs(float(delta_f) - float(self._stilde.delta_f)) > 1e-8:
            raise ValueError(
                'overwhitened delta_f %s does not match the template %s'
                % (self._stilde.delta_f, delta_f))
        return self._stilde


def valid_bounds(template, reader):
    psize = (len(template) - 1) * 2
    valid_end = int(psize - reader.trim_padding)
    valid_start = int(valid_end - reader.blocksize * reader.sample_rate)
    if valid_start < 0 or valid_end > psize:
        raise ValueError(
            'valid slice [%s, %s) is outside the %s-sample filter buffer'
            % (valid_start, valid_end, psize))
    return valid_start, valid_end


def write_bank(path, rows, f_lower):
    with h5py.File(path, 'w') as handle:
        handle.attrs['parameters'] = numpy.array(
            ['mass1', 'mass2', 'spin1z', 'spin2z', 'f_lower'], dtype='S')
        for name in ('mass1', 'mass2', 'spin1z', 'spin2z'):
            handle[name] = numpy.array([row[name] for row in rows], dtype=numpy.float64)
        handle['f_lower'] = numpy.full(len(rows), f_lower, dtype=numpy.float64)


def build_templates():
    os.makedirs(CACHE, exist_ok=True)
    rows = [
        dict(mass1=39.0, mass2=33.0, spin1z=0.0, spin2z=0.0),
        dict(mass1=36.0, mass2=29.0, spin1z=0.1, spin2z=0.0),
        dict(mass1=30.0, mass2=25.0, spin1z=0.0, spin2z=0.1),
        dict(mass1=22.0, mass2=15.0, spin1z=0.0, spin2z=0.0),
    ]
    path = os.path.join(CACHE, 'diagnostic_bank.hdf')
    write_bank(path, rows, F_LOW)
    bank = LiveFilterBank(
        path, SAMPLE_RATE, minimum_buffer=BLOCK_SECONDS,
        approximant='IMRPhenomD', increment=8, low_frequency_cutoff=F_LOW)
    delta_f = min(bank.freq_resolution_for_template(i) for i in range(len(bank)))
    templates = [bank.get_template(i, delta_f=delta_f) for i in range(len(bank))]
    duration = 1.0 / float(templates[0].delta_f)
    if abs(duration - BUFFER_SECONDS) > 1e-6:
        raise RuntimeError(
            'template buffer is %.3f s, expected %.3f s' % (duration, BUFFER_SECONDS))
    for template in templates[1:]:
        if abs(float(template.delta_f) - float(templates[0].delta_f)) > 1e-10:
            raise RuntimeError('templates do not share one frequency resolution')
    return bank, templates


def truncate_psd(psd):
    truncated = inverse_spectrum_truncation(
        psd, int(3.5 * SAMPLE_RATE), low_frequency_cutoff=F_LOW)
    data = numpy.real(numpy.asarray(truncated, dtype=numpy.complex128))
    data = numpy.array(data, dtype=numpy.float64, copy=True)
    data[~numpy.isfinite(data)] = 0.0
    return FrequencySeries(data, delta_f=psd.delta_f, epoch=psd.epoch)


def analytic_psd(flen, delta_f):
    psd = pycbc.psd.from_string(PSD_NAME, int(flen), float(delta_f), F_LOW)
    return truncate_psd(psd)


def overwhiten(strain, psd, epoch):
    """Divide the dynamic-range-scaled FFT by the same scaled PSD."""
    dyn = float(pycbc.DYN_RANGE_FAC)
    scaled = numpy.asarray(strain, dtype=numpy.float64) * dyn
    series = TimeSeries(scaled.astype(numpy.float32), delta_t=1.0 / SAMPLE_RATE,
                        epoch=epoch)
    spectrum = series.to_frequencyseries()
    if len(spectrum) != len(psd):
        raise ValueError(
            'FFT length %s does not match the PSD length %s'
            % (len(spectrum), len(psd)))
    weight = numpy.asarray(psd, dtype=numpy.float64) * (dyn ** 2)
    spec = numpy.asarray(spectrum, dtype=numpy.complex128)
    data = numpy.zeros(len(spec), dtype=numpy.complex64)
    good = numpy.isfinite(weight) & (weight > 0) & numpy.isfinite(spec.real) & numpy.isfinite(spec.imag)
    data[good] = (spec[good] / weight[good]).astype(numpy.complex64)
    stilde = FrequencySeries(data, delta_f=float(psd.delta_f), epoch=epoch)
    stilde.psd = FrequencySeries(
        weight.astype(numpy.float32), delta_f=float(psd.delta_f), epoch=epoch)
    return stilde


def make_reader(strain, psd, buffer_epoch, valid_epoch):
    ntime = int(round(BUFFER_SECONDS * SAMPLE_RATE))
    if len(strain) != ntime:
        raise ValueError('strain block has %s samples, expected %s' % (len(strain), ntime))
    stilde = overwhiten(strain, psd, buffer_epoch)
    return BlockReader(
        stilde, valid_epoch, SAMPLE_RATE, BLOCK_SECONDS,
        int(round(TRIM_SECONDS * SAMPLE_RATE)))


def slice_time(series, gps_start, n_samples):
    i0 = int(round((gps_start - float(series.start_time)) * series.sample_rate))
    i1 = i0 + int(n_samples)
    if i0 < 0 or i1 > len(series):
        raise ValueError(
            'slice %s:%s is outside the series of length %s (epoch %s)'
            % (i0, i1, len(series), series.start_time))
    return numpy.array(series[i0:i1], dtype=numpy.float64)


def gaussian_strain(delta_f):
    length = int(NOISE_SECONDS * SAMPLE_RATE)
    noise = noise_from_string(
        PSD_NAME, length, 1.0 / SAMPLE_RATE, seed=2048, low_frequency_cutoff=F_LOW)
    noise.start_time = NOISE_GPS
    flen = int(round(SAMPLE_RATE / (2.0 * delta_f) + 1))
    psd = analytic_psd(flen, delta_f)
    return noise, psd


def inject(noise, template, psd):
    """Add one physical IMRPhenomD waveform on top of ``noise``."""
    from pycbc.waveform import get_fd_waveform
    hp_fd, _hc = get_fd_waveform(
        approximant='IMRPhenomD',
        mass1=float(template.params.mass1),
        mass2=float(template.params.mass2),
        spin1z=float(template.params.spin1z),
        spin2z=float(template.params.spin2z),
        delta_f=float(psd.delta_f),
        f_lower=F_LOW,
        distance=1.0)
    kmax = min(len(hp_fd), len(psd))
    hp_fd = hp_fd[:kmax]
    snr_at_1_mpc = float(sigma(hp_fd, psd[:kmax], low_frequency_cutoff=F_LOW))
    distance = snr_at_1_mpc / 25.0
    hp, _hc = get_td_waveform(
        approximant='IMRPhenomD',
        mass1=float(template.params.mass1),
        mass2=float(template.params.mass2),
        spin1z=float(template.params.spin1z),
        spin2z=float(template.params.spin2z),
        delta_t=1.0 / SAMPLE_RATE,
        f_lower=F_LOW,
        distance=distance)
    wave = numpy.asarray(hp, dtype=numpy.float64)
    peak = int(numpy.argmax(numpy.abs(wave)))
    # Buffer occupies [16 s, 32 s) of the 32 s realization. The valid 8 s
    # starts 7.75 s into that buffer.
    valid_offset = BUFFER_SECONDS - TRIM_SECONDS - BLOCK_SECONDS
    target = int(round((16.0 + valid_offset + INJECT_INTO_VALID) * SAMPLE_RATE))
    start = target - peak
    stop = start + len(wave)
    if start < 0 or stop > len(noise):
        raise RuntimeError('injection does not fit in the noise realization')
    data = numpy.array(numpy.asarray(noise, dtype=numpy.float64), dtype=numpy.float64, copy=True)
    data[start:stop] += wave
    return data, distance, snr_at_1_mpc / distance


def ensure_gwosc(path):
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    request = urllib.request.Request(GWOSC_URL, headers={'User-Agent': 'pycbc-live-diagnostics'})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    if len(payload) < 1000:
        raise RuntimeError('GWOSC download was unexpectedly small')
    with open(path, 'wb') as handle:
        handle.write(payload)


def real_strain(delta_f):
    path = os.path.join(CACHE, GWOSC_NAME)
    ensure_gwosc(path)
    with h5py.File(path, 'r') as handle:
        samples = numpy.array(handle['strain/Strain'], dtype=numpy.float64)
        epoch = float(handle['strain/Strain'].attrs['Xstart'])
        delta_t = float(handle['strain/Strain'].attrs['Xspacing'])
    raw = TimeSeries(samples, delta_t=delta_t, epoch=epoch)
    raw = highpass(raw, 15.0)
    from pycbc.filter import resample_to_delta_t
    decimated = resample_to_delta_t(raw, 1.0 / SAMPLE_RATE)
    ntime = int(round(BUFFER_SECONDS * SAMPLE_RATE))
    valid_offset = BUFFER_SECONDS - TRIM_SECONDS - BLOCK_SECONDS
    buffer_epoch = REAL_VALID_GPS - valid_offset
    block = slice_time(decimated, buffer_epoch, ntime)
    # PSD from the whole resampled file, which is longer than the 8 s block.
    seg = 4 * SAMPLE_RATE
    estimated = welch(decimated, seg_len=seg, seg_stride=seg // 2, avg_method='median')
    estimated = interpolate(estimated, delta_f)
    flen = int(round(SAMPLE_RATE / (2.0 * delta_f) + 1))
    if len(estimated) != flen:
        data = numpy.zeros(flen, dtype=numpy.float64)
        n = min(flen, len(estimated))
        data[:n] = numpy.asarray(estimated[:n], dtype=numpy.float64)
        estimated = FrequencySeries(data, delta_f=delta_f)
    psd = truncate_psd(estimated)
    return block, psd, (
        'GWOSC GW150914 H1 %s, GPS %s, resampled from %.0f Hz to %d Hz'
        % (GWOSC_NAME, int(epoch), 1.0 / delta_t, SAMPLE_RATE))


def trigger_table(result):
    found = {}
    if result is False or result is None or len(result.get('snr', [])) == 0:
        return found
    for i, template_id in enumerate(result['template_id']):
        found[int(template_id)] = {
            key: result[key][i] for key in result
            if len(numpy.atleast_1d(result[key])) == len(result['snr'])
        }
    return found


def collect(cpu_filter, jax_filter, reader):
    cpu_result = cpu_filter.process_data(reader)
    stilde = reader.overwhitened_data(cpu_filter.tgroups[0][0].delta_f)
    cpu = {}
    for group in cpu_filter.tgroups:
        valid_start, valid_end = valid_bounds(group[0], reader)
        for template in group:
            sigmasq = float(template.sigmasq(stilde.psd))
            norm = 4.0 * float(template.delta_f) / numpy.sqrt(sigmasq)
            full = numpy.array(template.out, dtype=numpy.complex64)
            cpu[int(template.id)] = dict(
                valid=full[valid_start:valid_end] * numpy.complex64(norm),
                full=full,
                norm=norm,
                sigmasq=sigmasq,
                template=template,
                corr=numpy.array(template.cout, dtype=numpy.complex64),
                valid_start=valid_start,
                valid_end=valid_end,
            )
    jax_result = jax_filter.process_data(reader)
    jax_map = {}
    for series, group in zip(jax_filter._series, jax_filter.tgroups):
        valid_start, valid_end = valid_bounds(group[0], reader)
        snr_ts, q = batched_correlate_ifft(series, stilde, jax_filter.device)
        snr_ts = numpy.asarray(snr_ts, dtype=numpy.complex64)
        q = numpy.asarray(q, dtype=numpy.complex64)
        for row, template in enumerate(group):
            sigmasq = float(template.sigmasq(stilde.psd))
            norm = 4.0 * float(template.delta_f) / numpy.sqrt(sigmasq)
            jax_map[int(template.id)] = dict(
                valid=snr_ts[row, valid_start:valid_end] * numpy.complex64(norm),
                full=snr_ts[row],
                q=q[row],
                norm=norm,
                sigmasq=sigmasq,
                template=template,
                valid_start=valid_start,
                valid_end=valid_end,
            )
    return cpu, jax_map, trigger_table(cpu_result), trigger_table(jax_result), stilde


def peak_stats(valid, start_time):
    mag = numpy.abs(numpy.asarray(valid, dtype=numpy.complex128))
    index = int(numpy.argmax(mag))
    return float(mag[index]), float(start_time) + index / float(SAMPLE_RATE), index


def veto_indices(valid_start, valid_end, center):
    coarse = numpy.arange(valid_start, valid_end, 64, dtype=numpy.int64)
    half = int(0.1 * SAMPLE_RATE)
    fine_lo = max(valid_start, int(center) - half)
    fine_hi = min(valid_end, int(center) + half + 1)
    fine = numpy.arange(fine_lo, fine_hi, 4, dtype=numpy.int64)
    return numpy.unique(numpy.concatenate([
        coarse, fine, numpy.array([int(center)], dtype=numpy.int64)])).astype(numpy.int64)


def reduced_from_cpu(corr, full, norm, bins, indices, psd, template):
    snr = numpy.asarray(full[indices], dtype=numpy.complex64)
    raw = numpy.asarray(power_chisq_at_points_from_precomputed(
        Array(numpy.asarray(corr, dtype=numpy.complex64), dtype=numpy.complex64),
        snr, float(norm), bins, numpy.asarray(indices, dtype=numpy.float64)),
        dtype=numpy.float64)
    dof = (len(bins) - 1) * 2 - 2
    reduced = raw / float(dof)
    return reduced, raw, numpy.repeat(numpy.float64(dof), len(indices))


def reduced_from_jax(q, full, norm, bins, indices):
    snr = numpy.asarray(full[indices], dtype=numpy.complex64)
    reduced, dof_vec, raw = reduced_power_chisq(
        q, snr, float(norm), bins, numpy.asarray(indices, dtype=numpy.float64))
    return (numpy.asarray(reduced, dtype=numpy.float64),
            numpy.asarray(raw, dtype=numpy.float64),
            numpy.asarray(dof_vec, dtype=numpy.float64))


def sg_reason(sg, template):
    if not getattr(sg, 'do', False):
        return 'Sine-Gaussian chi-squared is disabled.'
    if template.params.template_hash not in sg.params:
        return ('Sine-Gaussian chi-squared is not defined for this template '
                '(it is outside the location mask).')
    return None


def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out')


def plot_case(path, title, cpu, jax, cpu_trig, jax_trig, template_id, reader,
              stilde, sg, note):
    cpu_row = cpu[template_id]
    jax_row = jax[template_id]
    template = cpu_row['template']
    nvalid = cpu_row['valid_end'] - cpu_row['valid_start']
    if nvalid != int(BLOCK_SECONDS * SAMPLE_RATE):
        raise RuntimeError('valid block is %s samples, not 8 seconds' % nvalid)
    time = numpy.arange(nvalid) / float(SAMPLE_RATE)
    cpu_mag = numpy.abs(numpy.asarray(cpu_row['valid'], dtype=numpy.complex128))
    jax_mag = numpy.abs(numpy.asarray(jax_row['valid'], dtype=numpy.complex128))
    residual = jax_mag - cpu_mag
    max_abs = float(numpy.max(numpy.abs(residual)))
    cpu_peak, cpu_gps, cpu_i = peak_stats(cpu_row['valid'], reader.start_time)
    jax_peak, jax_gps, jax_i = peak_stats(jax_row['valid'], reader.start_time)
    center = cpu_row['valid_start'] + cpu_i
    jax_center = jax_row['valid_start'] + jax_i
    indices = veto_indices(cpu_row['valid_start'], cpu_row['valid_end'], center)
    indices = numpy.unique(numpy.append(indices, numpy.int64(jax_center)))
    bins = numpy.asarray(
        sg.cached_chisq_bins(template, stilde.psd), dtype=numpy.int64)
    cpu_red, cpu_raw, cpu_dof = reduced_from_cpu(
        cpu_row['corr'], cpu_row['full'], cpu_row['norm'], bins, indices,
        stilde.psd, template)
    jax_red, jax_raw, jax_dof = reduced_from_jax(
        jax_row['q'], jax_row['full'], jax_row['norm'], bins, indices)
    cpu_at = int(numpy.where(indices == center)[0][0])
    jax_at = int(numpy.where(indices == jax_center)[0][0])
    skip = sg_reason(sg, template)
    if skip is None:
        cpu_sg = numpy.asarray(sg.values(
            stilde, template, stilde.psd,
            numpy.asarray(cpu_row['full'][indices], dtype=numpy.complex64),
            cpu_row['norm'], cpu_raw.astype(numpy.float32),
            cpu_dof.astype(numpy.uint32), indices), dtype=numpy.float64)
        jax_sg = numpy.asarray(sg_chisq_values(
            sg, stilde, template, stilde.psd,
            numpy.asarray(jax_row['full'][indices], dtype=numpy.complex64),
            jax_row['norm'], jax_raw.astype(numpy.float32),
            jax_dof.astype(numpy.uint32), indices), dtype=numpy.float64)
    else:
        cpu_sg = jax_sg = None

    def at_peak(trig, fallback_snr, fallback_gps, series_mag, index, reduced, sg_series, ind):
        if trig is not None and template_id in trig:
            row = trig[template_id]
            return (float(row['snr']), float(row['end_time']),
                    float(row['chisq']), float(row['sg_chisq']), 'live trigger')
        where = int(numpy.where(ind == index)[0][0])
        sg_value = float('nan') if sg_series is None else float(sg_series[where])
        return (fallback_snr, fallback_gps, float(reduced[where]), sg_value,
                'series peak, below the record threshold')

    cpu_summary = at_peak(
        cpu_trig, cpu_peak, cpu_gps, cpu_mag, center, cpu_red, cpu_sg, indices)
    jax_summary = at_peak(
        jax_trig, jax_peak, jax_gps, jax_mag, jax_center, jax_red, jax_sg, indices)

    fig = plt.figure(figsize=(11.0, 12.2))
    grid = fig.add_gridspec(4, 2, height_ratios=[1.15, 1.05, 1.05, 0.85],
                            hspace=0.38, wspace=0.28,
                            left=0.08, right=0.98, top=0.93, bottom=0.05)
    ax_snr = fig.add_subplot(grid[0, 0])
    ax_res = fig.add_subplot(grid[0, 1])
    ax_chi = fig.add_subplot(grid[1, 0])
    ax_chi_res = fig.add_subplot(grid[1, 1])
    ax_sg = fig.add_subplot(grid[2, 0])
    ax_zoom = fig.add_subplot(grid[2, 1])
    ax_sum = fig.add_subplot(grid[3, :])

    ax_snr.plot(time, cpu_mag, color=CPU_COLOR, lw=1.0, label='CPU')
    ax_snr.plot(time, jax_mag, color=JAX_COLOR, lw=1.0, ls='--', label='JAX')
    ax_snr.set_ylabel('SNR')
    ax_snr.set_xlabel('Time in valid block (s)')
    ax_snr.set_xlim(0.0, BLOCK_SECONDS)
    ax_snr.legend(loc='upper right')
    ax_snr.set_title('Complex-SNR magnitude')

    ax_res.plot(time, residual, color=RES_COLOR, lw=0.8)
    ax_res.axhline(0.0, color='0.5', lw=0.6)
    ax_res.set_ylabel('JAX − CPU SNR')
    ax_res.set_xlabel('Time in valid block (s)')
    ax_res.set_xlim(0.0, BLOCK_SECONDS)
    ax_res.set_title('max |dSNR| = %.3e' % max_abs)

    veto_t = (indices - cpu_row['valid_start']) / float(SAMPLE_RATE)
    ax_chi.plot(veto_t, cpu_red, color=CPU_COLOR, lw=1.0, label='CPU')
    ax_chi.plot(veto_t, jax_red, color=JAX_COLOR, lw=1.0, ls='--', label='JAX')
    ax_chi.plot(veto_t[cpu_at], cpu_red[cpu_at], 'o', color=CPU_COLOR, ms=4.5)
    ax_chi.plot(veto_t[jax_at], jax_red[jax_at], 'o', color=JAX_COLOR, ms=3.5)
    ax_chi.set_ylabel('Reduced power χ²')
    ax_chi.set_xlabel('Time in valid block (s)')
    ax_chi.set_xlim(0.0, BLOCK_SECONDS)
    ax_chi.legend(loc='upper right')
    ax_chi.set_title('Power χ², decimated; dot is the SNR-peak sample')

    ax_chi_res.plot(veto_t, jax_red - cpu_red, color=RES_COLOR, lw=0.8)
    ax_chi_res.axhline(0.0, color='0.5', lw=0.6)
    ax_chi_res.set_ylabel('JAX − CPU reduced χ²')
    ax_chi_res.set_xlabel('Time in valid block (s)')
    ax_chi_res.set_xlim(0.0, BLOCK_SECONDS)
    ax_chi_res.set_title('Power χ² residual')

    if skip is None:
        ax_sg.plot(veto_t, cpu_sg, color=CPU_COLOR, lw=1.0, label='CPU')
        ax_sg.plot(veto_t, jax_sg, color=JAX_COLOR, lw=1.0, ls='--', label='JAX')
        ax_sg.plot(veto_t[cpu_at], cpu_sg[cpu_at], 'o', color=CPU_COLOR, ms=4.5)
        ax_sg.plot(veto_t[jax_at], jax_sg[jax_at], 'o', color=JAX_COLOR, ms=3.5)
        ax_sg.set_ylabel('Sine-Gaussian χ²')
        ax_sg.set_xlabel('Time in valid block (s)')
        ax_sg.set_xlim(0.0, BLOCK_SECONDS)
        ax_sg.legend(loc='upper right')
        ax_sg.set_title('Sine-Gaussian χ² (1 where new SNR is below %g)' % sg.snr_threshold)
    else:
        ax_sg.axis('off')
        ax_sg.text(0.0, 0.5, 'Sine-Gaussian panel skipped.\n' + skip,
                   transform=ax_sg.transAxes, va='center', ha='left', fontsize=10)

    half = int(0.1 * SAMPLE_RATE)
    z0 = max(0, cpu_i - half)
    z1 = min(nvalid, cpu_i + half)
    ax_zoom.plot(time[z0:z1], cpu_mag[z0:z1], color=CPU_COLOR, lw=1.2, label='CPU')
    ax_zoom.plot(time[z0:z1], jax_mag[z0:z1], color=JAX_COLOR, lw=1.2, ls='--', label='JAX')
    ax_zoom.set_ylabel('SNR')
    ax_zoom.set_xlabel('Time in valid block (s)')
    ax_zoom.legend(loc='upper right')
    ax_zoom.set_title('Zoom, 0.2 s around the CPU peak')

    def fmt(summary):
        snr, gps, chisq, sg_value, origin = summary
        sg_text = 'n/a' if not numpy.isfinite(sg_value) else '%.6g' % sg_value
        return snr, gps, chisq, sg_text, origin

    c_snr, c_gps, c_chi, c_sg, c_origin = fmt(cpu_summary)
    j_snr, j_gps, j_chi, j_sg, j_origin = fmt(jax_summary)
    lines = [
        'template %d   IMRPhenomD   m1=%.1f  m2=%.1f  s1z=%.2f  s2z=%.2f'
        % (template_id, template.params.mass1, template.params.mass2,
           template.params.spin1z, template.params.spin2z),
        note,
        'valid block GPS %.3f to %.3f    SNR threshold %g'
        % (reader.start_time, reader.start_time + BLOCK_SECONDS, SNR_THRESHOLD),
        '',
        '%-22s %16s %16s' % ('', 'CPU', 'JAX'),
        '%-22s %16.6g %16.6g' % ('peak SNR', c_snr, j_snr),
        '%-22s %16.6f %16.6f' % ('peak GPS time (s)', c_gps, j_gps),
        '%-22s %16.6g %16.6g' % ('reduced power χ²', c_chi, j_chi),
        '%-22s %16s %16s' % ('sine-Gaussian χ²', c_sg, j_sg),
        '%-22s %16s %16s' % ('value source', c_origin, j_origin),
        '',
        'max |SNR_JAX − SNR_CPU| over the 8 s series: %.6e' % max_abs,
    ]
    ax_sum.axis('off')
    ax_sum.text(0.0, 1.0, '\n'.join(lines), transform=ax_sum.transAxes,
                va='top', ha='left', family='monospace', fontsize=9)

    for axis in (ax_snr, ax_res, ax_chi, ax_chi_res, ax_zoom):
        style_axes(axis)
    if skip is None:
        style_axes(ax_sg)

    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return dict(
        template_id=template_id,
        cpu_snr=c_snr,
        jax_snr=j_snr,
        cpu_time=c_gps,
        jax_time=j_gps,
        cpu_chisq=c_chi,
        jax_chisq=j_chi,
        cpu_sg=c_sg,
        jax_sg=j_sg,
        max_abs_dsn=max_abs,
        series_cpu_snr=cpu_peak,
        series_jax_snr=jax_peak,
    )


def choose_template(kind, injected_id, cpu):
    if kind == 'injection':
        return injected_id
    return max(cpu, key=lambda template_id: peak_stats(cpu[template_id]['valid'], 0.0)[0])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with CPUScheme():
        bank, templates = build_templates()
        sg = SingleDetSGChisq(
            bank, num_bins=CHISQ_BINS, snr_threshold=SNR_THRESHOLD,
            chisq_locations=SG_LOCATIONS)
        common = dict(
            snr_threshold=SNR_THRESHOLD,
            chisq_bins=CHISQ_BINS,
            sg_chisq=sg,
            snr_abort_threshold=None,
            newsnr_threshold=None,
        )
        cpu_filter = LiveBatchMatchedFilter(templates, **common)
        jax_filter = JaxLiveBatchMatchedFilter(
            list(templates), jax_platform='cpu', device_id=0, **common)
        used = sum(1 for report in jax_filter.startup_comparisons if report['use_generated'])
        print('JAX waveform generator used for %d/%d templates'
              % (used, len(jax_filter.startup_comparisons)))
        for report in jax_filter.startup_comparisons:
            print('  id=%s match_template=%s using=%s' % (
                report['template_id'], report['match_template'],
                'jax' if report['use_generated'] else 'cpu-series'))

        delta_f = float(templates[0].delta_f)
        noise, psd = gaussian_strain(delta_f)
        valid_offset = BUFFER_SECONDS - TRIM_SECONDS - BLOCK_SECONDS
        buffer_epoch = NOISE_GPS + 16.0
        valid_epoch = buffer_epoch + valid_offset
        noise_block = slice_time(noise, buffer_epoch, int(BUFFER_SECONDS * SAMPLE_RATE))
        injected, distance, expected = inject(noise, templates[0], psd)
        injected_block = slice_time(
            TimeSeries(injected, delta_t=1.0 / SAMPLE_RATE, epoch=NOISE_GPS),
            buffer_epoch, int(BUFFER_SECONDS * SAMPLE_RATE))
        real_block, real_psd, real_origin = real_strain(delta_f)
        real_buffer_epoch = REAL_VALID_GPS - valid_offset

        cases = [
            ('gaussian_noise.png', 'Gaussian noise, aLIGOZeroDetHighPower',
             'gaussian', make_reader(noise_block, psd, buffer_epoch, valid_epoch),
             'Colored Gaussian noise, no injection.'),
            ('injection.png', 'Gaussian noise plus IMRPhenomD injection',
             'injection', make_reader(injected_block, psd, buffer_epoch, valid_epoch),
             'Injection of template %d at %.1f Mpc, expected optimal SNR %.1f.'
             % (int(templates[0].id), distance, expected)),
            ('real_h1_gw150914.png', 'GWOSC H1 around GW150914',
             'real', make_reader(real_block, real_psd, real_buffer_epoch, REAL_VALID_GPS),
             real_origin),
        ]

        summaries = []
        for filename, title, kind, reader, note in cases:
            cpu, jax, cpu_trig, jax_trig, stilde = collect(cpu_filter, jax_filter, reader)
            template_id = choose_template(kind, int(templates[0].id), cpu)
            summary = plot_case(
                os.path.join(OUT_DIR, filename), title, cpu, jax, cpu_trig,
                jax_trig, template_id, reader, stilde, sg, note)
            summary['file'] = filename
            summary['kind'] = kind
            summaries.append(summary)
            print(
                '%s template %d  CPU SNR %.6g at %.6f   JAX SNR %.6g at %.6f   '
                'max |dSNR| %.6e' % (
                    kind, summary['template_id'], summary['cpu_snr'], summary['cpu_time'],
                    summary['jax_snr'], summary['jax_time'], summary['max_abs_dsn']))
            print('  series peaks CPU %.6g JAX %.6g  reduced χ² CPU %.6g JAX %.6g  '
                  'sg %s %s' % (
                      summary['series_cpu_snr'], summary['series_jax_snr'],
                      summary['cpu_chisq'], summary['jax_chisq'],
                      summary['cpu_sg'], summary['jax_sg']))
        return summaries


if __name__ == '__main__':
    main()
