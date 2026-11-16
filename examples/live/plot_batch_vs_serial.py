#!/usr/bin/env python
"""Compare one-template matched filtering with the JAX live SNR surface.

Serial processing calls ``pycbc.filter.matched_filter_core`` once per
template on a shared overwhitened frequency series. Batch processing is
``JaxLiveBatchMatchedFilter`` (the on-device ρ(t, φ, θ) surface). Both
correlate the complex64 series the batch filter stores (its generator when
that series matches the bank to 1e-6, otherwise the bank series), on the
same overwhitened block, with the bank normalization
``4 * df / sqrt(sigmasq)``. The overwhitened series is not divided by the
PSD again: that division already happened when the block was whitened.

The CPU ``LiveBatchMatchedFilter`` duration-group engine is drawn on the
zoom plots only. It is not the serial baseline.

Regenerate from the repository root with::

    /home/bug/work/pycbc_live/.venv/bin/python examples/live/plot_batch_vs_serial.py
"""

import os
import sys
import time

if __name__ == '__main__':
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')

_SOURCE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy

import pycbc
import pycbc.psd
from pycbc.filter import LiveBatchMatchedFilter, matched_filter_core, sigma
from pycbc.filter.jax_matchedfilter import JaxLiveBatchMatchedFilter
from pycbc.noise import noise_from_string
from pycbc.psd import inverse_spectrum_truncation
from pycbc.scheme import CPUScheme
from pycbc.types import Array, FrequencySeries, TimeSeries
from pycbc.vetoes.jax_chisq import (
    power_chisq_batch,
    reduced_from_raw,
    sg_gate_mask,
)
from pycbc.vetoes.sgchisq import SingleDetSGChisq
from pycbc.waveform import get_fd_waveform, get_td_waveform
from pycbc.waveform.bank import LiveFilterBank

SEED = 20261116
SAMPLE_RATE = 2048
F_LOW = 20.0
BLOCK_SECONDS = 8.0
TRIM_SECONDS = 0.25
TARGET_SNR = 20.0
MASS_RANGE = (5.0, 40.0)
SPIN_RANGE = (-0.5, 0.5)
PSD_NAME = 'aLIGOZeroDetHighPower'
CHISQ_BINS = '8'
SG_LOCATIONS = ['mtotal>0:10-0']
SG_SNR_THRESHOLD = 4.0
BANK_SIZES = (10, 20)
APPROXIMANT = 'IMRPhenomD'

# Parity gate used by the JAX live filter.
SNR_RTOL = 1.0e-4
SNR_ATOL_QUIET = 1.0e-3
CHISQ_RTOL = 1.0e-4
CHISQ_ATOL = 1.0e-3

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'diagnostics', 'batch_vs_serial')

SERIAL_COLOR = '#222222'
BATCH_COLOR = '#d95f02'
LIVE_COLOR = '#1b9e77'
RES_COLOR = '#7570b3'


class BlockReader(object):
    """The slice of strain both filters read."""

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


def write_bank(path, rows):
    with h5py.File(path, 'w') as handle:
        handle.attrs['parameters'] = numpy.array(
            ['mass1', 'mass2', 'spin1z', 'spin2z', 'f_lower'], dtype='S')
        for name in ('mass1', 'mass2', 'spin1z', 'spin2z'):
            handle[name] = numpy.array([row[name] for row in rows], dtype=numpy.float64)
        handle['f_lower'] = numpy.full(len(rows), F_LOW, dtype=numpy.float64)


def draw_rows(n, rng):
    rows = []
    seen = set()
    while len(rows) < n:
        mass1 = float(rng.uniform(*MASS_RANGE))
        mass2 = float(rng.uniform(MASS_RANGE[0], mass1))
        row = dict(
            mass1=mass1,
            mass2=mass2,
            spin1z=float(rng.uniform(*SPIN_RANGE)),
            spin2z=float(rng.uniform(*SPIN_RANGE)),
        )
        key = tuple(numpy.round([row[k] for k in ('mass1', 'mass2', 'spin1z', 'spin2z')], 12))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def build_templates(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, 'random_bank.hdf')
    write_bank(path, rows)
    bank = LiveFilterBank(
        path, SAMPLE_RATE, minimum_buffer=BLOCK_SECONDS,
        approximant=APPROXIMANT, increment=8, low_frequency_cutoff=F_LOW)
    delta_f = min(bank.freq_resolution_for_template(i) for i in range(len(bank)))
    templates = [bank.get_template(i, delta_f=delta_f) for i in range(len(bank))]
    for template in templates[1:]:
        if abs(float(template.delta_f) - float(templates[0].delta_f)) > 1e-12:
            raise RuntimeError('templates do not share one frequency resolution')
    return bank, templates, float(delta_f)


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


def overwhiten(strain, psd):
    """Divide the dynamic-range-scaled FFT by the same scaled PSD."""
    dyn = float(pycbc.DYN_RANGE_FAC)
    scaled = numpy.asarray(strain, dtype=numpy.float64) * dyn
    series = TimeSeries(
        scaled.astype(numpy.float32), delta_t=1.0 / SAMPLE_RATE, epoch=0.0)
    spectrum = series.to_frequencyseries()
    if len(spectrum) != len(psd):
        raise ValueError(
            'FFT length %s does not match the PSD length %s'
            % (len(spectrum), len(psd)))
    weight = numpy.asarray(psd, dtype=numpy.float64) * (dyn ** 2)
    spec = numpy.asarray(spectrum, dtype=numpy.complex128)
    data = numpy.zeros(len(spec), dtype=numpy.complex64)
    good = (numpy.isfinite(weight) & (weight > 0)
            & numpy.isfinite(spec.real) & numpy.isfinite(spec.imag))
    data[good] = (spec[good] / weight[good]).astype(numpy.complex64)
    stilde = FrequencySeries(data, delta_f=float(psd.delta_f), epoch=0.0)
    stilde.psd = FrequencySeries(
        weight.astype(numpy.float32), delta_f=float(psd.delta_f), epoch=0.0)
    return stilde


def make_reader(stilde):
    return BlockReader(
        stilde, 0.0, SAMPLE_RATE, BLOCK_SECONDS,
        int(round(TRIM_SECONDS * SAMPLE_RATE)))


def place_waveform(wave, ntime, valid_start):
    """Put the amplitude peak 5 s into the valid block, cropping if needed."""
    wave = numpy.asarray(wave, dtype=numpy.float64)
    peak = int(numpy.argmax(numpy.abs(wave)))
    target = int(valid_start + round(5.0 * SAMPLE_RATE))
    start = target - peak
    if start < 0:
        wave = wave[-start:]
        start = 0
    stop = start + len(wave)
    if stop > ntime:
        wave = wave[:ntime - start]
        stop = ntime
    placed = numpy.zeros(ntime, dtype=numpy.float64)
    placed[start:stop] = wave
    return placed


def injection_waveform(template, psd, ntime, valid_start):
    """Physical IMRPhenomD waveform scaled to an optimal SNR near 20."""
    hp_fd, _hc = get_fd_waveform(
        approximant=APPROXIMANT,
        mass1=float(template.params.mass1),
        mass2=float(template.params.mass2),
        spin1z=float(template.params.spin1z),
        spin2z=float(template.params.spin2z),
        delta_f=float(psd.delta_f),
        f_lower=F_LOW,
        distance=1.0)
    kmax = min(len(hp_fd), len(psd))
    snr_at_1_mpc = float(sigma(
        hp_fd[:kmax], psd[:kmax], low_frequency_cutoff=F_LOW))
    distance = snr_at_1_mpc / TARGET_SNR
    hp, _hc = get_td_waveform(
        approximant=APPROXIMANT,
        mass1=float(template.params.mass1),
        mass2=float(template.params.mass2),
        spin1z=float(template.params.spin1z),
        spin2z=float(template.params.spin2z),
        delta_t=1.0 / SAMPLE_RATE,
        f_lower=F_LOW,
        distance=distance)
    placed = place_waveform(hp, ntime, valid_start)
    return placed, distance, snr_at_1_mpc / distance


def _as_complex64(series):
    return numpy.array(series, dtype=numpy.complex64, copy=True)


def stored_series(jax_filter):
    """Complex64 rows the batch filter actually correlates, keyed by id."""
    found = {}
    for state in jax_filter._states:
        data = numpy.asarray(state.h)
        for row, template in enumerate(state.templates):
            found[int(template.id)] = numpy.array(data[row], dtype=numpy.complex64, copy=True)
    return found


def _filter_input(template, samples_by_id):
    if samples_by_id is None:
        return template
    epoch = 0.0 if template.epoch is None else float(template.epoch)
    return FrequencySeries(
        numpy.array(samples_by_id[int(template.id)], dtype=numpy.complex64, copy=True),
        delta_f=float(template.delta_f), epoch=epoch)


def serial_correlations(templates, stilde, samples_by_id=None):
    """One matched_filter_core call per template. PSD is not applied again."""
    packs = []
    for template in templates:
        h_norm = float(template.sigmasq(stilde.psd))
        unnorm, corr, norm = matched_filter_core(
            _filter_input(template, samples_by_id), stilde, psd=None,
            low_frequency_cutoff=None, high_frequency_cutoff=None, h_norm=h_norm)
        packs.append((
            _as_complex64(unnorm),
            _as_complex64(corr),
            float(norm),
            h_norm,
        ))
    return packs


def reduced_and_sg(power, sg, template, stilde, corr, unnorm, norm, index):
    snrv = numpy.asarray(unnorm[index:index + 1], dtype=numpy.complex64)
    indices = numpy.array([index], dtype=numpy.float64)
    raw, dof = power.values(
        Array(numpy.asarray(corr, dtype=numpy.complex64)),
        snrv, float(norm), stilde.psd, indices, template)
    raw = numpy.asarray(raw, dtype=numpy.float64)
    dof = numpy.asarray(dof, dtype=numpy.float64)
    reduced = float(raw[0] / dof[0])
    sg_values = sg.values(
        stilde, template, stilde.psd, snrv, float(norm),
        raw.astype(numpy.float32), dof.astype(numpy.uint32), indices)
    return reduced, float(numpy.asarray(sg_values, dtype=numpy.float64)[0])


def summarize_pack(template, unnorm, corr, norm, stilde, reader, power, sg):
    valid_start, valid_end = valid_bounds(template, reader)
    valid = numpy.asarray(unnorm[valid_start:valid_end] * numpy.complex64(norm))
    relative = int(numpy.argmax(numpy.abs(valid) ** 2))
    index = valid_start + relative
    reduced, sg_value = reduced_and_sg(
        power, sg, template, stilde, corr, unnorm, norm, index)
    sample = valid[relative]
    return valid.astype(numpy.complex64, copy=False), dict(
        template_id=int(template.id),
        index=index,
        relative=relative,
        time=relative / float(SAMPLE_RATE),
        snr=float(numpy.abs(sample)),
        phase=float(numpy.angle(sample)),
        chisq=reduced,
        sg=sg_value,
        norm=float(norm),
        complex=complex(sample),
    )


def run_serial(templates, stilde, reader, power, sg, samples_by_id=None):
    """Warm up FFT plans, then time a second one-template loop."""
    serial_correlations(templates, stilde, samples_by_id)
    started = time.perf_counter()
    packs = serial_correlations(templates, stilde, samples_by_id)
    elapsed = time.perf_counter() - started
    valids = []
    rows = []
    for template, pack in zip(templates, packs):
        unnorm, corr, norm, _h_norm = pack
        valid, row = summarize_pack(
            template, unnorm, corr, norm, stilde, reader, power, sg)
        valids.append(valid)
        rows.append(row)
    return numpy.stack(valids, axis=0), rows, elapsed


def _block_until_ready(values):
    import jax
    for value in values:
        if value is not None:
            jax.block_until_ready(value)


def run_batch(jax_filter, stilde, reader):
    """Evaluate the normalized surface. Time the second call."""
    prepared = []
    for group, state in zip(jax_filter.tgroups, jax_filter._states):
        state.ensure_psd(stilde.psd, stilde, jax_filter.power_chisq, jax_filter.sg_chisq)
        state.set_strain(stilde)
        valid_start, valid_end = valid_bounds(group[0], reader)
        prepared.append((group, state, valid_start, valid_end))

    for _group, state, valid_start, valid_end in prepared:
        jax_filter._evaluate_surface(state, valid_start, valid_end - valid_start)
    _block_until_ready([state.abs_snr for _g, state, _a, _b in prepared])

    started = time.perf_counter()
    for _group, state, valid_start, valid_end in prepared:
        jax_filter._evaluate_surface(state, valid_start, valid_end - valid_start)
    _block_until_ready([state.abs_snr for _g, state, _a, _b in prepared])
    elapsed = time.perf_counter() - started

    by_id = {}
    order = []
    for group, state, valid_start, valid_end in prepared:
        surface = numpy.asarray(state.snr, dtype=numpy.complex64)
        raw = numpy.asarray(power_chisq_batch(
            state.q, state.peak_unnorm, state.norms, state.bins,
            state.index, state.num_bins_d), dtype=numpy.float32)
        reduced, dof = reduced_from_raw(raw, state.num_bins)
        sg_values = numpy.ones(len(group), dtype=numpy.float32)
        if getattr(jax_filter.sg_chisq, 'do', False) and state.sg is not None:
            computed = numpy.asarray(
                jax_filter._sg_for_group(state), dtype=numpy.float32)
            gate = sg_gate_mask(
                numpy.asarray(state.abs_snr), reduced, jax_filter.sg_chisq.snr_threshold)
            use = numpy.asarray(state.sg['compute'], dtype=bool) & numpy.asarray(gate, dtype=bool)
            sg_values = numpy.where(use, computed, numpy.float32(1.0))
        indices = numpy.asarray(state.index, dtype=numpy.int64)
        peaks = numpy.asarray(state.peak, dtype=numpy.complex64)
        for row, template in enumerate(group):
            valid = numpy.array(surface[row, valid_start:valid_end], dtype=numpy.complex64, copy=True)
            relative = int(numpy.argmax(numpy.abs(valid) ** 2))
            kernel_relative = int(indices[row] - valid_start)
            if kernel_relative != relative:
                raise RuntimeError(
                    'batch kernel peak %s does not match the copied surface peak %s'
                    % (kernel_relative, relative))
            sample = valid[relative]
            item = dict(
                template_id=int(template.id),
                index=valid_start + relative,
                relative=relative,
                time=relative / float(SAMPLE_RATE),
                snr=float(numpy.abs(sample)),
                phase=float(numpy.angle(sample)),
                chisq=float(reduced[row]),
                sg=float(sg_values[row]),
                norm=float(numpy.asarray(state.norms)[row]),
                complex=complex(sample),
                kernel_snr=float(numpy.abs(peaks[row])),
                kernel_phase=float(numpy.angle(peaks[row])),
                dof=int(dof[row]),
            )
            by_id[int(template.id)] = (valid, item)
            order.append(int(template.id))
    valids = numpy.stack([by_id[i][0] for i in order], axis=0)
    rows = [by_id[i][1] for i in order]
    return valids, rows, elapsed, order


def live_valid_series(live_filter, templates, stilde, reader):
    """Optional CPU duration-group filter, copied off the workspace buffers."""
    try:
        live_filter.process_data(reader)
    except Exception as exc:
        print('CPU live-batch filter skipped: %s' % exc, flush=True)
        return None
    series = {}
    for template in templates:
        if not hasattr(template, 'out'):
            return None
        valid_start, valid_end = valid_bounds(template, reader)
        norm = 4.0 * float(template.delta_f) / numpy.sqrt(float(template.sigmasq(stilde.psd)))
        full = numpy.array(template.out, dtype=numpy.complex64, copy=True)
        series[int(template.id)] = full[valid_start:valid_end] * numpy.complex64(norm)
    return series


def align_rows(templates, serial_rows, batch_rows):
    serial = {row['template_id']: row for row in serial_rows}
    batch = {row['template_id']: row for row in batch_rows}
    ordered_s = []
    ordered_b = []
    for template in templates:
        ordered_s.append(serial[int(template.id)])
        ordered_b.append(batch[int(template.id)])
    return ordered_s, ordered_b


def reorder_valid(templates, valids, rows):
    by_id = {row['template_id']: (valids[i], row) for i, row in enumerate(rows)}
    stacked = []
    ordered = []
    for template in templates:
        valid, row = by_id[int(template.id)]
        stacked.append(valid)
        ordered.append(row)
    return numpy.stack(stacked, axis=0), ordered


def close_enough(ref, other, rtol, atol):
    ref = float(numpy.real(ref))
    other = float(numpy.real(other))
    scale = max(abs(ref), abs(other), 1.0)
    return abs(ref - other) <= atol + rtol * scale


def snr_atol(snr):
    return SNR_ATOL_QUIET if float(snr) < 5.0 else 0.0


def surface_failures(serial_valid, batch_valid, template_ids):
    """Pointwise magnitude gate. Returns human-readable failures."""
    failures = []
    worst = 0.0
    for i, template_id in enumerate(template_ids):
        ref = numpy.abs(numpy.asarray(serial_valid[i], dtype=numpy.complex128))
        other = numpy.abs(numpy.asarray(batch_valid[i], dtype=numpy.complex128))
        diff = numpy.abs(other - ref)
        scale = numpy.maximum(numpy.maximum(ref, other), 1.0)
        atol = numpy.where(ref < 5.0, SNR_ATOL_QUIET, 0.0)
        allowed = atol + SNR_RTOL * scale
        bad = diff > allowed
        worst = max(worst, float(diff.max()))
        if numpy.any(bad):
            where = int(numpy.argmax(diff - allowed))
            failures.append(
                'template %s surface |dSNR|=%.3e at sample %s (serial %.6g, batch %.6g, allowed %.3e)'
                % (template_id, float(diff[where]), where, float(ref[where]),
                   float(other[where]), float(allowed[where])))
    return failures, worst


def peak_failures(serial_rows, batch_rows):
    failures = []
    for serial, batch in zip(serial_rows, batch_rows):
        label = serial['template_id']
        atol = snr_atol(serial['snr'])
        if not close_enough(serial['snr'], batch['snr'], SNR_RTOL, atol):
            failures.append(
                'template %s peak SNR serial %.8g batch %.8g'
                % (label, serial['snr'], batch['snr']))
        if abs(serial['index'] - batch['index']) > 1:
            failures.append(
                'template %s peak sample serial %s batch %s'
                % (label, serial['index'], batch['index']))
        if not close_enough(serial['chisq'], batch['chisq'], CHISQ_RTOL, CHISQ_ATOL):
            failures.append(
                'template %s reduced chisq serial %.8g batch %.8g'
                % (label, serial['chisq'], batch['chisq']))
        if not close_enough(serial['sg'], batch['sg'], CHISQ_RTOL, CHISQ_ATOL):
            failures.append(
                'template %s sg chisq serial %.8g batch %.8g'
                % (label, serial['sg'], batch['sg']))
    return failures


def complex_residual(serial_valid, batch_valid):
    return numpy.asarray(batch_valid, dtype=numpy.complex128) - numpy.asarray(
        serial_valid, dtype=numpy.complex128)


def max_abs_per_template(residual):
    return numpy.max(numpy.abs(residual), axis=1)


def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out')


def heatmap(ax, image, title, vmin, vmax, cmap, cbar_label, norm=None):
    extent = [0.0, BLOCK_SECONDS, -0.5, image.shape[0] - 0.5]
    shown = ax.imshow(
        image, origin='lower', aspect='auto', extent=extent,
        interpolation='nearest', cmap=cmap, vmin=None if norm else vmin,
        vmax=None if norm else vmax, norm=norm)
    ax.set_xlabel('Time in valid block (s)')
    ax.set_ylabel('Template index')
    ax.set_title(title)
    cbar = ax.figure.colorbar(shown, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    return shown


def pool_time(image, n_bins, reduce):
    """Collapse time so a one-sample peak survives the image downsample."""
    ntime = image.shape[1]
    edges = numpy.unique(numpy.linspace(0, ntime, int(n_bins) + 1, dtype=int))
    columns = [reduce(image[:, a:b]) for a, b in zip(edges[:-1], edges[1:])]
    return numpy.stack(columns, axis=1)


def _max_over_bin(chunk):
    return numpy.max(chunk, axis=1)


def _signed_extreme(chunk):
    pick = numpy.argmax(numpy.abs(chunk), axis=1)
    return chunk[numpy.arange(chunk.shape[0]), pick]


def plot_surfaces(path, title, serial_valid, batch_valid):
    serial_mag = numpy.abs(numpy.asarray(serial_valid, dtype=numpy.complex128))
    batch_mag = numpy.abs(numpy.asarray(batch_valid, dtype=numpy.complex128))
    difference = batch_mag - serial_mag
    residual = complex_residual(serial_valid, batch_valid)
    n_bins = 400
    serial_show = pool_time(serial_mag, n_bins, _max_over_bin)
    batch_show = pool_time(batch_mag, n_bins, _max_over_bin)
    difference_show = pool_time(difference, n_bins, _signed_extreme)
    vmax = float(max(serial_show.max(), batch_show.max(), 1.0))
    dmax = float(max(numpy.max(numpy.abs(difference_show)), 1.0e-12))
    cmax = float(numpy.max(numpy.abs(residual)))
    bin_ms = 1000.0 * BLOCK_SECONDS / serial_show.shape[1]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), constrained_layout=True)
    heatmap(axes[0], serial_show, 'Serial |ρ(t, θ)|', 0.0, vmax, 'viridis', 'SNR')
    heatmap(axes[1], batch_show, 'Batch |ρ(t, θ)|', 0.0, vmax, 'viridis', 'SNR')
    heatmap(
        axes[2], difference_show, 'Δ|ρ| = |ρ_batch| − |ρ_serial|',
        -dmax, dmax, 'coolwarm', 'SNR',
        norm=TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax))
    fig.suptitle(
        '%s\nmax |ρ_batch − ρ_serial| = %.3e; each column is %.0f ms'
        % (title, cmax, bin_ms))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return cmax, dmax


def plot_phase(path, title, serial_valid, batch_valid, serial_rows, batch_rows):
    serial_mag = numpy.abs(numpy.asarray(serial_valid, dtype=numpy.complex128))
    phase = numpy.angle(numpy.asarray(batch_valid, dtype=numpy.complex128)
                        * numpy.conj(numpy.asarray(serial_valid, dtype=numpy.complex128)))
    n_bins = 400
    # Keep the phase sample in each bin where the serial SNR is largest.
    edges = numpy.unique(numpy.linspace(0, phase.shape[1], n_bins + 1, dtype=int))
    phase_cols = []
    mag_cols = []
    for start, stop in zip(edges[:-1], edges[1:]):
        pick = numpy.argmax(serial_mag[:, start:stop], axis=1)
        rows = numpy.arange(phase.shape[0])
        phase_cols.append(phase[rows, start + pick])
        mag_cols.append(serial_mag[rows, start + pick])
    phase_show = numpy.stack(phase_cols, axis=1)
    mag_show = numpy.stack(mag_cols, axis=1)
    masked = numpy.ma.array(phase_show, mask=mag_show < 3.0)
    peak_s = numpy.array([row['phase'] for row in serial_rows])
    peak_b = numpy.array([row['phase'] for row in batch_rows])
    peak_d = numpy.angle(numpy.exp(1j * (peak_b - peak_s)))
    index = numpy.arange(len(serial_rows))

    fig, axes = plt.subplots(
        2, 1, figsize=(10.5, 6.4), constrained_layout=True,
        gridspec_kw={'height_ratios': [1.4, 1.0]})
    extent = [0.0, BLOCK_SECONDS, -0.5, phase.shape[0] - 0.5]
    shown = axes[0].imshow(
        masked, origin='lower', aspect='auto', extent=extent,
        interpolation='nearest', cmap='coolwarm', vmin=-numpy.pi, vmax=numpy.pi)
    axes[0].set_xlabel('Time in valid block (s)')
    axes[0].set_ylabel('Template index')
    axes[0].set_title('Phase difference (rad), masked where serial |ρ| < 3; column is the loudest sample in %.0f ms'
                      % (1000.0 * BLOCK_SECONDS / masked.shape[1]))
    cbar = fig.colorbar(shown, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label('arg(ρ_batch) − arg(ρ_serial) (rad)')

    axes[1].plot(index, peak_s, 'o', color=SERIAL_COLOR, ms=4.5, label='Serial peak phase')
    axes[1].plot(index, peak_b, 'x', color=BATCH_COLOR, ms=5.5, label='Batch peak phase')
    axes[1].plot(index, peak_d, color=RES_COLOR, lw=1.0, label='Wrapped difference')
    axes[1].set_xlabel('Template index')
    axes[1].set_ylabel('Phase (rad)')
    axes[1].set_xlim(-0.5, len(index) - 0.5)
    axes[1].legend(loc='best', frameon=False)
    style_axes(axes[1])
    fig.suptitle(title)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_scatter(path, cases):
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), constrained_layout=True)
    specs = (
        ('snr', 'Peak SNR', 'Peak SNR, serial', 'Peak SNR, batch'),
        ('chisq', 'Reduced power χ²', 'Reduced χ², serial', 'Reduced χ², batch'),
        ('sg', 'Sine-Gaussian χ²', 'Sine-Gaussian χ², serial', 'Sine-Gaussian χ², batch'),
    )
    markers = {10: 'o', 20: 's'}
    colors = {'noise': '#222222', 'injection': '#d95f02'}
    for ax, (key, title, xlabel, ylabel) in zip(axes, specs):
        lo = numpy.inf
        hi = -numpy.inf
        for case in cases:
            x = numpy.array([row[key] for row in case['serial_rows']], dtype=numpy.float64)
            y = numpy.array([row[key] for row in case['batch_rows']], dtype=numpy.float64)
            lo = min(lo, float(x.min()), float(y.min()))
            hi = max(hi, float(x.max()), float(y.max()))
            ax.scatter(
                x, y, s=28, marker=markers[case['n']],
                facecolors='none' if case['n'] == 20 else colors[case['kind']],
                edgecolors=colors[case['kind']],
                linewidths=1.1,
                label='%d templates, %s' % (case['n'], case['kind']))
        pad = 0.05 * (hi - lo + 1.0e-6)
        lo -= pad
        hi += pad
        ax.plot([lo, hi], [lo, hi], color='0.45', lw=0.8, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        style_axes(ax)
    axes[0].legend(loc='upper left', frameon=False, fontsize=8)
    fig.suptitle('Peak quantities, one point per template')
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_max_residual(path, cases):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True, sharey=True)
    for ax, kind in zip(axes, ('noise', 'injection')):
        for case in cases:
            if case['kind'] != kind:
                continue
            residual = complex_residual(case['serial_valid'], case['batch_valid'])
            values = max_abs_per_template(residual)
            ax.plot(
                numpy.arange(len(values)), values, marker='o', ms=3.5,
                lw=1.0, label='%d templates' % case['n'])
        ax.set_xlabel('Template index')
        ax.set_yscale('log')
        ax.set_title('Colored noise' if kind == 'noise' else 'Noise plus injection')
        ax.legend(frameon=False)
        style_axes(ax)
    axes[0].set_ylabel(r'max$_t$ $|\rho_{\mathrm{batch}} - \rho_{\mathrm{serial}}|$')
    fig.suptitle('Largest complex SNR residual in the 8 s block')
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_zoom(path, panels):
    fig, axes = plt.subplots(
        len(panels), 2, figsize=(11.0, 2.35 * len(panels)),
        constrained_layout=True, sharex='row')
    if len(panels) == 1:
        axes = numpy.array([axes])
    half = int(0.1 * SAMPLE_RATE)
    for row, panel in enumerate(panels):
        serial = numpy.asarray(panel['serial'], dtype=numpy.complex128)
        batch = numpy.asarray(panel['batch'], dtype=numpy.complex128)
        center = int(panel['center'])
        lo = max(0, center - half)
        hi = min(len(serial), center + half)
        time = numpy.arange(lo, hi) / float(SAMPLE_RATE)
        ax = axes[row, 0]
        ax.plot(time, numpy.abs(serial[lo:hi]), color=SERIAL_COLOR, lw=1.3, label='Serial')
        ax.plot(time, numpy.abs(batch[lo:hi]), color=BATCH_COLOR, lw=1.3, ls='--', label='Batch')
        if panel['live'] is not None:
            live = numpy.asarray(panel['live'], dtype=numpy.complex128)
            ax.plot(time, numpy.abs(live[lo:hi]), color=LIVE_COLOR, lw=1.0, ls=':',
                    label='CPU live batch')
        ax.set_ylabel('SNR')
        ax.set_title(panel['title'])
        ax.legend(loc='best', frameon=False, fontsize=8)
        style_axes(ax)
        residual = batch[lo:hi] - serial[lo:hi]
        axr = axes[row, 1]
        axr.plot(time, numpy.abs(residual), color=RES_COLOR, lw=1.0)
        axr.set_ylabel(r'$|\rho_{\mathrm{batch}} - \rho_{\mathrm{serial}}|$')
        axr.set_title('Residual, max %.3e' % float(numpy.max(numpy.abs(residual))))
        style_axes(axr)
        axr.set_xlabel('Time in valid block (s)')
        ax.set_xlabel('Time in valid block (s)')
    fig.suptitle('SNR within ±0.1 s of the serial peak')
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_summary_table(path, lines):
    fig, ax = plt.subplots(figsize=(12.4, 0.36 * len(lines) + 0.6))
    ax.axis('off')
    ax.text(
        0.01, 0.99, '\n'.join(lines), transform=ax.transAxes, va='top', ha='left',
        family='monospace', fontsize=8)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def loudest_index(rows):
    return int(numpy.argmax([row['snr'] for row in rows]))


def case_title(n, kind):
    dataset = 'colored Gaussian noise' if kind == 'noise' else 'noise plus injection'
    return '%d templates, %s' % (n, dataset)


def write_summary(path, seed, rows_drawn, distance, expected_snr, measured_snr,
                  device, cases, failures, series_note):
    lines = []
    lines.append('seed %d' % seed)
    lines.append('approximant %s' % APPROXIMANT)
    lines.append('sample_rate_Hz %d' % SAMPLE_RATE)
    lines.append('f_lower_Hz %.1f' % F_LOW)
    lines.append('valid_block_s %.1f' % BLOCK_SECONDS)
    lines.append('psd %s' % PSD_NAME)
    lines.append('jax_device %s' % device)
    lines.append('injection_template_index 0')
    lines.append('injection_distance_Mpc %.6g' % distance)
    lines.append('injection_optimal_snr %.6g' % expected_snr)
    lines.append('injection_measured_peak_snr_serial %.6g' % measured_snr)
    lines.append('serial matched_filter_core on the complex64 series the batch filter correlates')
    lines.append('batch JaxLiveBatchMatchedFilter surface, platform cpu')
    lines.append('normalization 4 * delta_f / sqrt(sigmasq); PSD not applied a second time')
    lines.append('chisq reduced power chi-squared at the |rho| peak; sg chi-squared uses the live newsnr gate')
    lines.append('timings are the second call; FFTW plans and the JAX compile are excluded')
    lines.append('series_note %s' % series_note)
    lines.append('')
    lines.append('template parameters (index in the 20-template draw)')
    for i, row in enumerate(rows_drawn):
        lines.append(
            '  %2d  mass1 %.8f  mass2 %.8f  spin1z %.8f  spin2z %.8f'
            % (i, row['mass1'], row['mass2'], row['spin1z'], row['spin2z']))
    lines.append('')
    header = (
        '%4s %10s %3s %12s %12s %12s %10s %10s %10s %10s %10s %10s '
        '%12s %12s %12s %12s %12s %12s %12s'
        % ('n', 'dataset', 'idx', 'snr_s', 'snr_b', 'dsnr',
           't_s', 't_b', 'dt_samp', 'phi_s', 'phi_b', 'dphi',
           'chisq_s', 'chisq_b', 'dchisq', 'sg_s', 'sg_b', 'dsg', 'max_|drho|'))
    lines.append(header)
    aggregate = []
    for case in cases:
        residual = complex_residual(case['serial_valid'], case['batch_valid'])
        per = max_abs_per_template(residual)
        mag = numpy.abs(numpy.abs(case['batch_valid']) - numpy.abs(case['serial_valid']))
        peak_dsnr = []
        dchisq = []
        dsg = []
        for i, (serial, batch) in enumerate(zip(case['serial_rows'], case['batch_rows'])):
            dphi = float(numpy.angle(numpy.exp(1j * (batch['phase'] - serial['phase']))))
            lines.append(
                '%4d %10s %3d %12.6g %12.6g %12.3e %10.5f %10.5f %10d %10.5f %10.5f %10.3e '
                '%12.6g %12.6g %12.3e %12.6g %12.6g %12.3e %12.3e'
                % (case['n'], case['kind'], i,
                   serial['snr'], batch['snr'], batch['snr'] - serial['snr'],
                   serial['time'], batch['time'], batch['index'] - serial['index'],
                   serial['phase'], batch['phase'], dphi,
                   serial['chisq'], batch['chisq'], batch['chisq'] - serial['chisq'],
                   serial['sg'], batch['sg'], batch['sg'] - serial['sg'],
                   float(per[i])))
            peak_dsnr.append(abs(batch['snr'] - serial['snr']))
            dchisq.append(abs(batch['chisq'] - serial['chisq']))
            dsg.append(abs(batch['sg'] - serial['sg']))
        ratio = case['serial_time'] / case['batch_time'] if case['batch_time'] > 0 else float('inf')
        aggregate.append(
            '%d templates, %s: max |complex dSNR| %.6e   max ||ρ| residual| %.6e   '
            'max |d peak SNR| %.6e   max |d chisq| %.6e   max |d sg chisq| %.6e   '
            'serial %.4f s   batch %.4f s   serial/batch %.3f'
            % (case['n'], case['kind'], float(per.max()), float(mag.max()),
               max(peak_dsnr), max(dchisq), max(dsg),
               case['serial_time'], case['batch_time'], ratio))
    lines.append('')
    lines.append('aggregate')
    lines.extend(aggregate)
    if failures:
        lines.append('')
        lines.append('parity gate failures')
        lines.extend(failures)
    else:
        lines.append('')
        lines.append('parity gate: passed')
    text = '\n'.join(lines) + '\n'
    with open(path, 'w') as handle:
        handle.write(text)
    return aggregate, lines


def filter_kwargs(sg):
    return dict(
        snr_threshold=SG_SNR_THRESHOLD,
        chisq_bins=CHISQ_BINS,
        sg_chisq=sg,
        snr_abort_threshold=None,
        newsnr_threshold=None,
    )


def make_power():
    from pycbc.vetoes.chisq import SingleDetPowerChisq
    return SingleDetPowerChisq(CHISQ_BINS, None)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = numpy.random.default_rng(SEED)
    rows = draw_rows(max(BANK_SIZES), rng)
    print('seed %d' % SEED, flush=True)
    print('drew %d aligned-spin %s templates, masses in [%.0f, %.0f], spins in [%.1f, %.1f]'
          % (len(rows), APPROXIMANT, MASS_RANGE[0], MASS_RANGE[1], SPIN_RANGE[0], SPIN_RANGE[1]),
          flush=True)

    with CPUScheme():
        bank, templates, delta_f = build_templates(rows)
        ntime = (len(templates[0]) - 1) * 2
        print('common buffer %.3f s, delta_f %.6g Hz, %d samples'
              % (1.0 / delta_f, delta_f, ntime), flush=True)
        psd = analytic_psd(len(templates[0]), delta_f)
        noise = noise_from_string(
            PSD_NAME, ntime, 1.0 / SAMPLE_RATE, seed=SEED,
            low_frequency_cutoff=F_LOW)
        noise = numpy.array(noise, dtype=numpy.float64, copy=True)
        reader_probe = make_reader(overwhiten(noise, psd))
        valid_start, _valid_end = valid_bounds(templates[0], reader_probe)
        placed, distance, expected = injection_waveform(
            templates[0], psd, ntime, valid_start)
        injected = noise + placed

        sg = SingleDetSGChisq(
            bank, num_bins=CHISQ_BINS, snr_threshold=SG_SNR_THRESHOLD,
            chisq_locations=SG_LOCATIONS)
        power = make_power()

        # One serial pass on the injected template sets the plotted amplitude
        # inside the requested optimal-SNR window if the segment truncates it.
        stilde_try = overwhiten(injected, psd)
        reader = make_reader(stilde_try)
        pack = serial_correlations([templates[0]], stilde_try)[0]
        _valid, peak_row = summarize_pack(
            templates[0], pack[0], pack[1], pack[2], stilde_try, reader, power, sg)
        measured = peak_row['snr']
        print('injection distance %.4g Mpc, sigma SNR %.3f, serial peak SNR %.3f'
              % (distance, expected, measured), flush=True)
        if not (15.0 <= measured <= 25.0) and measured > 0.0:
            scale = TARGET_SNR / measured
            injected = noise + placed * scale
            distance = distance / scale
            expected = expected * scale
            stilde_try = overwhiten(injected, psd)
            pack = serial_correlations([templates[0]], stilde_try)[0]
            _valid, peak_row = summarize_pack(
                templates[0], pack[0], pack[1], pack[2], stilde_try, reader, power, sg)
            measured = peak_row['snr']
            print('rescaled injection by %.4g; serial peak SNR %.3f' % (scale, measured),
                  flush=True)

        datasets = {
            'noise': overwhiten(noise, psd),
            'injection': overwhiten(injected, psd),
        }
        import jax
        device = str(jax.devices()[0])
        print('jax devices %s' % jax.devices(), flush=True)

        cases = []
        failures = []
        series_notes = []
        for n in BANK_SIZES:
            subset = templates[:n]
            print('building filters for %d templates' % n, flush=True)
            jax_filter = JaxLiveBatchMatchedFilter(
                list(subset), jax_platform='cpu', device_id=0, **filter_kwargs(sg))
            used = sum(1 for report in jax_filter.startup_comparisons if report['use_generated'])
            note = 'jax generator used for %d/%d templates' % (
                used, len(jax_filter.startup_comparisons))
            series_notes.append('%d: %s' % (n, note))
            print('  ' + note, flush=True)
            for report in jax_filter.startup_comparisons:
                print('  id %s match_template %.8g using %s' % (
                    report['template_id'], report['match_template'],
                    'jax' if report['use_generated'] else 'cpu-series'), flush=True)
            live_filter = LiveBatchMatchedFilter(list(subset), **filter_kwargs(sg))
            samples_by_id = stored_series(jax_filter)
            for kind in ('noise', 'injection'):
                stilde = datasets[kind]
                reader = make_reader(stilde)
                print('  serial %s' % kind, flush=True)
                serial_valid, serial_rows, serial_time = run_serial(
                    subset, stilde, reader, power, sg, samples_by_id)
                print('  batch %s' % kind, flush=True)
                batch_valid, batch_rows, batch_time, _order = run_batch(
                    jax_filter, stilde, reader)
                serial_valid, serial_rows = reorder_valid(subset, serial_valid, serial_rows)
                batch_valid, batch_rows = reorder_valid(subset, batch_valid, batch_rows)
                live = live_valid_series(live_filter, subset, stilde, reader)
                if live is not None:
                    live = numpy.stack([live[int(t.id)] for t in subset], axis=0)
                ids = [int(t.id) for t in subset]
                surface_bad, _worst = surface_failures(serial_valid, batch_valid, ids)
                peak_bad = peak_failures(serial_rows, batch_rows)
                for item in surface_bad + peak_bad:
                    failures.append('%d %s: %s' % (n, kind, item))
                title = case_title(n, kind)
                plot_surfaces(
                    os.path.join(OUT_DIR, 'n%d_%s_snr.png' % (n, kind)),
                    title, serial_valid, batch_valid)
                plot_phase(
                    os.path.join(OUT_DIR, 'n%d_%s_phase.png' % (n, kind)),
                    title, serial_valid, batch_valid, serial_rows, batch_rows)
                print('  %s serial %.3f s  batch %.3f s  ratio %.3f' % (
                    kind, serial_time, batch_time,
                    serial_time / batch_time if batch_time else float('inf')), flush=True)
                cases.append(dict(
                    n=n, kind=kind, title=title,
                    serial_valid=serial_valid, batch_valid=batch_valid,
                    serial_rows=serial_rows, batch_rows=batch_rows,
                    serial_time=serial_time, batch_time=batch_time,
                    live=live, subset=subset,
                ))

        plot_scatter(os.path.join(OUT_DIR, 'peak_scatter.png'), cases)
        plot_max_residual(os.path.join(OUT_DIR, 'max_abs_drho.png'), cases)

        panels = []
        for n in BANK_SIZES:
            noise_case = next(case for case in cases if case['n'] == n and case['kind'] == 'noise')
            inj_case = next(case for case in cases if case['n'] == n and case['kind'] == 'injection')
            loud = loudest_index(noise_case['serial_rows'])
            for case, index, label in (
                    (inj_case, 0, 'injected template 0'),
                    (noise_case, loud, 'loudest noise template %d' % loud)):
                live_row = None if case['live'] is None else case['live'][index]
                panels.append(dict(
                    title='%s, %s' % (case_title(n, case['kind']), label),
                    serial=case['serial_valid'][index],
                    batch=case['batch_valid'][index],
                    live=live_row,
                    center=case['serial_rows'][index]['relative'],
                ))
        plot_zoom(os.path.join(OUT_DIR, 'zoom_snr.png'), panels)

        summary_path = os.path.join(OUT_DIR, 'comparison_summary.txt')
        aggregate, _summary_lines = write_summary(
            summary_path, SEED, rows, distance, expected, measured, device,
            cases, failures, '; '.join(series_notes))
        headline = [
            'seed %d    device %s' % (SEED, device),
            'injection template 0 at %.4g Mpc, optimal SNR %.3f, serial peak %.3f'
            % (distance, expected, measured),
            'series: ' + '; '.join(series_notes),
            '',
        ] + aggregate
        if failures:
            headline += ['', 'GATE FAILURES (%d)' % len(failures)] + failures[:12]
        else:
            headline += ['', 'parity gate passed']
        plot_summary_table(os.path.join(OUT_DIR, 'summary_metrics.png'), headline)

        print('', flush=True)
        for line in aggregate:
            print(line, flush=True)
        if failures:
            print('parity gate failures: %d' % len(failures), flush=True)
            for item in failures[:30]:
                print(' ', item, flush=True)
        else:
            print('parity gate passed', flush=True)
        pngs = sorted(
            name for name in os.listdir(OUT_DIR) if name.endswith('.png'))
        for name in pngs:
            size = os.path.getsize(os.path.join(OUT_DIR, name))
            print('png %s %d bytes' % (name, size), flush=True)
            if size < 1000:
                raise RuntimeError('%s is unexpectedly small' % name)
        return failures


if __name__ == '__main__':
    problems = main()
    if problems:
        sys.exit(1)
