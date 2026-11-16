#!/usr/bin/env python
"""Compare CPU and JAX PyCBC Live matched filters on TaylorF2 and IMRPhenomD.

Builds a few-template bank, filters one overwhitened block with
``LiveBatchMatchedFilter`` and ``JaxLiveBatchMatchedFilter``, and checks the
SNR, peak time, power chi-squared, and sine-Gaussian chi-squared tolerances
from the JAX live-filter plan. A mixed run filters half the bank on each
backend. Single-detector products are written where ``check_results.py``
reads them, and that checker is run on the single-detector file.

Install the optional stack with::

    pip install 'jax' 'ripplegw'

GPU: ``python compare_jax_live.py --jax-platform gpu`` when JAX sees a GPU.
"""

import argparse
import os
import sys

# Prefer this source tree over a non-editable copy in site-packages.
_SOURCE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

import h5py
import numpy

import pycbc
from pycbc.scheme import CPUScheme
from pycbc.filter import LiveBatchMatchedFilter
from pycbc.filter.jax_matchedfilter import JaxLiveBatchMatchedFilter
from pycbc.types import FrequencySeries
from pycbc.vetoes.sgchisq import SingleDetSGChisq
from pycbc.waveform.bank import LiveFilterBank

# Tolerances from the parity gate.
SNR_RTOL = 1.0e-4
SNR_ATOL_QUIET = 1.0e-3
CHISQ_RTOL = 1.0e-4
CHISQ_ATOL = 1.0e-3

GPS_START = 1272790000.0
GPS_END = 1272790064.0
F_MIN = 20.0


class BlockReader(object):
    """Stand-in for the slice of StrainBuffer the live filter reads."""

    def __init__(self, stilde, psd, start_time, sample_rate, blocksize,
                 trim_padding):
        self._stilde = stilde
        self._psd = psd
        self.start_time = start_time
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.trim_padding = trim_padding

    def overwhitened_data(self, delta_f):
        if abs(delta_f - self._stilde.delta_f) > 1e-6:
            raise ValueError(
                f"reader delta_f {self._stilde.delta_f} != template {delta_f}")
        self._stilde.psd = self._psd
        return self._stilde


def write_bank(path, masses, spins, f_lower):
    """Write an HDF template bank LiveFilterBank can open."""
    mass1 = numpy.array([m[0] for m in masses], dtype=numpy.float64)
    mass2 = numpy.array([m[1] for m in masses], dtype=numpy.float64)
    spin1z = numpy.array([s[0] for s in spins], dtype=numpy.float64)
    spin2z = numpy.array([s[1] for s in spins], dtype=numpy.float64)
    flow = numpy.full(len(mass1), f_lower, dtype=numpy.float64)
    with h5py.File(path, 'w') as handle:
        handle.attrs['parameters'] = numpy.array(
            ['mass1', 'mass2', 'spin1z', 'spin2z', 'f_lower'], dtype='S')
        handle['mass1'] = mass1
        handle['mass2'] = mass2
        handle['spin1z'] = spin1z
        handle['spin2z'] = spin2z
        handle['f_lower'] = flow


def make_reader(bank, template_index, sample_rate):
    """Overwhiten one bank template so the matching template is loud.

    The PSD is scaled so the correlation stays moderate in float32. A
    dynamic-range PSD with values near 1e-5 makes the Allen chi-squared a
    cancellation of two 1e18 terms, which float32 cannot repeat.
    """
    htilde = bank[template_index]
    ntime = (len(htilde) - 1) * 2
    flen = len(htilde)
    ones = FrequencySeries(
        numpy.ones(flen, dtype=numpy.float32),
        delta_f=htilde.delta_f, epoch=GPS_START)
    from pycbc.scheme import CPUScheme
    with CPUScheme():
        sigmasq_ones = float(htilde.sigmasq(ones))
    target_sigmasq = 0.0625
    psd_level = numpy.float32(sigmasq_ones / target_sigmasq)
    psd_data = numpy.full(flen, psd_level, dtype=numpy.float32)
    psd = FrequencySeries(psd_data, delta_f=htilde.delta_f, epoch=GPS_START)
    snr_target = 12.0
    alpha = snr_target / numpy.sqrt(target_sigmasq)
    data = numpy.asarray(htilde.data, dtype=numpy.complex128) / float(psd_level) * alpha
    stilde = FrequencySeries(
        data.astype(numpy.complex64), delta_f=htilde.delta_f, epoch=GPS_START)
    stilde.psd = psd
    return BlockReader(
        stilde, psd, GPS_START, sample_rate,
        blocksize=ntime / float(sample_rate), trim_padding=0), htilde.id


def _by_id(result):
    found = {}
    if result is False or result is None or len(result.get('snr', [])) == 0:
        return found
    for i, template_id in enumerate(result['template_id']):
        found[int(template_id)] = {
            key: result[key][i] for key in result if len(result[key]) == len(result['snr'])
        }
    return found


def _close(ref, other, rtol, atol):
    ref = float(numpy.real(ref))
    other = float(numpy.real(other))
    scale = max(abs(ref), abs(other), 1.0)
    return abs(ref - other) <= atol + rtol * scale


def assert_parity(cpu, jax, label):
    """Raise AssertionError when a trigger column misses the parity gate."""
    cpu_map = _by_id(cpu)
    jax_map = _by_id(jax)
    if not cpu_map:
        raise AssertionError(f"{label}: CPU filter produced no triggers")
    missing = sorted(set(cpu_map) - set(jax_map))
    extra = sorted(set(jax_map) - set(cpu_map))
    if missing or extra:
        raise AssertionError(
            f"{label}: template ids differ, missing {missing}, extra {extra}")
    deltas = {'snr': 0.0, 'time': 0.0, 'chisq': 0.0, 'sg_chisq': 0.0}
    for template_id, cpu_row in cpu_map.items():
        jax_row = jax_map[template_id]
        snr = float(cpu_row['snr'])
        atol = SNR_ATOL_QUIET if snr < 5.0 else 0.0
        deltas['snr'] = max(deltas['snr'], abs(float(cpu_row['snr']) - float(jax_row['snr'])))
        if not _close(cpu_row['snr'], jax_row['snr'], SNR_RTOL, atol):
            raise AssertionError(
                f"{label}: SNR template {template_id}: "
                f"cpu {cpu_row['snr']} jax {jax_row['snr']}")
        dt = abs(float(cpu_row['end_time']) - float(jax_row['end_time']))
        deltas['time'] = max(deltas['time'], dt)
        if dt > 1.0 / 2048.0 + 1e-9:
            raise AssertionError(
                f"{label}: peak time template {template_id} differs by {dt}")
        for key in ('chisq', 'sg_chisq'):
            deltas[key] = max(
                deltas[key], abs(float(cpu_row[key]) - float(jax_row[key])))
            if not _close(cpu_row[key], jax_row[key], CHISQ_RTOL, CHISQ_ATOL):
                raise AssertionError(
                    f"{label}: {key} template {template_id}: "
                    f"cpu {cpu_row[key]} jax {jax_row[key]}")
        if int(cpu_row['chisq_dof']) != int(jax_row['chisq_dof']):
            raise AssertionError(
                f"{label}: chisq_dof template {template_id}: "
                f"cpu {cpu_row['chisq_dof']} jax {jax_row['chisq_dof']}")
    return deltas


def filter_pair(bank, reader, chisq_bins, sg_chisq, jax_platform):
    templates = list(bank)
    common = dict(
        snr_threshold=4.0,
        chisq_bins=chisq_bins,
        sg_chisq=sg_chisq,
        snr_abort_threshold=None,
        newsnr_threshold=None,
        max_triggers_in_batch=None,
    )
    cpu = LiveBatchMatchedFilter(templates, **common)
    jax_filter = JaxLiveBatchMatchedFilter(
        list(bank), jax_platform=jax_platform, device_id=0, **common)
    return cpu.process_data(reader), jax_filter.process_data(reader), jax_filter


def mixed_versus_cpu(bank, reader, chisq_bins, sg_chisq, jax_platform):
    """One CPU rank and one JAX rank, compared with an all-CPU filter."""
    templates = list(bank)
    common = dict(
        snr_threshold=4.0,
        chisq_bins=chisq_bins,
        sg_chisq=sg_chisq,
    )
    cpu_all = LiveBatchMatchedFilter(templates, **common).process_data(reader)
    cpu_half = LiveBatchMatchedFilter(templates[0::2], **common).process_data(reader)
    jax_half = JaxLiveBatchMatchedFilter(
        templates[1::2], jax_platform=jax_platform, **common).process_data(reader)
    mixed = _by_id(cpu_half)
    mixed.update(_by_id(jax_half))
    cpu_map = _by_id(cpu_all)
    if set(mixed) != set(cpu_map):
        raise AssertionError(
            "mixed ranks recovered template ids "
            f"{sorted(mixed)} vs CPU {sorted(cpu_map)}")
    for template_id, cpu_row in cpu_map.items():
        other = mixed[template_id]
        atol = SNR_ATOL_QUIET if float(cpu_row['snr']) < 5.0 else 0.0
        if not _close(cpu_row['snr'], other['snr'], SNR_RTOL, atol):
            raise AssertionError(
                f"mixed SNR template {template_id}: "
                f"cpu {cpu_row['snr']} mixed {other['snr']}")
        if abs(float(cpu_row['end_time']) - float(other['end_time'])) > 1.0 / 2048.0 + 1e-9:
            raise AssertionError(f"mixed peak time template {template_id}")
    return cpu_all


def write_single_hdf(path, bank, result, psd, detector='H1'):
    """Write one live-style single-detector file for check_results.py."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    asd = 1.0e-23 * pycbc.DYN_RANGE_FAC
    stored = numpy.full(len(psd), asd ** 2, dtype=numpy.float64)
    with h5py.File(path, 'w') as handle:
        group = handle.create_group(detector)
        for key in ('snr', 'chisq', 'chisq_dof', 'end_time', 'template_id',
                    'mass1', 'mass2', 'spin1z', 'spin2z'):
            group[key] = numpy.asarray(result[key])
        psd_group = group.create_dataset('psd', data=stored)
        psd_group.attrs['delta_f'] = float(psd.delta_f)
        psd_group.attrs['epoch'] = float(psd.epoch)


def write_injections(path, bank, template_index, end_time):
    row = bank.table[template_index]
    with h5py.File(path, 'w') as handle:
        handle['mass1'] = numpy.array([row.mass1])
        handle['mass2'] = numpy.array([row.mass2])
        handle['spin1z'] = numpy.array([row.spin1z])
        handle['spin2z'] = numpy.array([row.spin2z])
        handle['tc'] = numpy.array([end_time])


def run_check_results(bank_path, injection_path):
    """Run the single-detector checks from examples/live/check_results.py."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import check_results
    args = argparse.Namespace(
        gps_start=GPS_START,
        gps_end=GPS_END,
        f_min=F_MIN,
        bank=os.path.basename(bank_path),
        injections=os.path.basename(injection_path),
        detectors=['H1'],
        verbose=False,
    )
    previous = os.getcwd()
    try:
        os.chdir(os.path.dirname(os.path.abspath(bank_path)))
        failed = check_results.check_single_results(args)
    finally:
        os.chdir(previous)
    if failed:
        raise AssertionError("check_results.py single-detector checks failed")


def run_approximant(approximant, jax_platform, out_dir):
    sample_rate = 2048
    masses = [(30.0, 30.0), (35.0, 29.0), (28.0, 22.0), (40.0, 15.0)]
    spins = [(0.1, 0.0), (0.2, -0.1), (0.0, 0.15), (-0.2, 0.05)]
    bank_path = os.path.join(out_dir, f'{approximant}_bank.hdf')
    os.makedirs(out_dir, exist_ok=True)
    write_bank(bank_path, masses, spins, F_MIN)
    bank = LiveFilterBank(
        bank_path, sample_rate, minimum_buffer=8.0,
        approximant=approximant, increment=8,
        low_frequency_cutoff=F_MIN)
    sg = SingleDetSGChisq(
        bank, num_bins='4', snr_threshold=4.0,
        chisq_locations=['mtotal>0:10-0'])
    reader, injected_id = make_reader(bank, 0, sample_rate)
    with CPUScheme():
        cpu, jax_result, jax_filter = filter_pair(
            bank, reader, '4', sg, jax_platform)
        deltas = assert_parity(cpu, jax_result, approximant)
        mixed_versus_cpu(bank, reader, '4', sg, jax_platform)
    reports = jax_filter.startup_comparisons
    if not reports:
        raise AssertionError(f"{approximant}: startup waveform comparison did not run")
    for report in reports:
        print(
            f"  waveform id={report['template_id']} impl={report['implementation']} "
            f"match_template={report['match_template']:.8g} "
            f"match_spa={report['match_spa']:.8g} "
            f"match_lal_taylorf2={report['match_lal_taylorf2']:.8g} "
            f"match_lal_phenomd={report['match_lal_phenomd']:.8g} "
            f"using={'jax' if report['use_generated'] else 'cpu-series'}")
    if approximant == 'SPAtmplt':
        worst = min(r['match_spa'] for r in reports)
        if not (worst == worst) or worst < 1.0 - 1.0e-4:
            raise AssertionError(f"JAX SPA vs spa_tmplt match {worst}")
    by_id = _by_id(cpu)
    if injected_id not in by_id:
        raise AssertionError(
            f"{approximant}: injected template {injected_id} was not found")
    day = '2020_05_05'
    hdf_path = os.path.join(out_dir, 'output', day, f'{approximant}.hdf')
    # check_results globs output/????_??_??/*.hdf from the process cwd.
    write_single_hdf(hdf_path, bank, cpu, reader._psd)
    inj_path = os.path.join(out_dir, f'{approximant}_injections.hdf')
    write_injections(inj_path, bank, 0, float(by_id[injected_id]['end_time']))
    run_check_results(bank_path, inj_path)
    return {
        'approximant': approximant,
        'injected_id': int(injected_id),
        'snr': float(by_id[injected_id]['snr']),
        'ntriggers': len(by_id),
        'deltas': deltas,
        'reports': reports,
    }


def check_power_chisq_shift_sum():
    """JAX shift-sum versus power_chisq_at_points_from_precomputed."""
    from pycbc.types import Array
    from pycbc.vetoes.chisq import power_chisq_at_points_from_precomputed
    from pycbc.vetoes.jax_chisq import power_chisq_at_points

    rng = numpy.random.default_rng(1)
    ntime = 4096
    corr = (rng.normal(size=ntime) + 1j * rng.normal(size=ntime)).astype(numpy.complex64)
    corr[ntime // 2 + 1:] = 0
    bins = numpy.array([8, 40, 100, 400, 800, 1500], dtype=numpy.uint32)
    indices = numpy.array([10.0, 100.0, 333.0, 2000.0])
    snr = (rng.normal(size=4) + 1j * rng.normal(size=4)).astype(numpy.complex64)
    norm = 0.013
    with CPUScheme():
        reference = numpy.asarray(power_chisq_at_points_from_precomputed(
            Array(corr, dtype=numpy.complex64), snr, norm, bins, indices),
            dtype=numpy.float64)
    got = numpy.asarray(
        power_chisq_at_points(corr, snr, norm, bins, indices), dtype=numpy.float64)
    rel = numpy.max(numpy.abs(reference - got) / numpy.maximum(numpy.abs(reference), 1e-12))
    print(f"power-chisq shift-sum max relative delta {rel:.3e}")
    if rel > 1.0e-4:
        raise AssertionError(f"power chisq shift-sum relative delta {rel}")
    return float(rel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jax-platform', default='cpu', choices=['cpu', 'gpu'])
    parser.add_argument('--out-dir', default=None,
                        help='Directory for banks and check_results products')
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'jax_parity')
    os.makedirs(out_dir, exist_ok=True)
    if args.jax_platform == 'gpu':
        import jax
        devices = [dev for dev in jax.devices() if dev.platform == 'gpu']
        if not devices:
            print('No JAX GPU device; skipping the GPU parity run')
            return 0
    chisq_rel = check_power_chisq_shift_sum()
    summaries = []
    for approximant in ('TaylorF2', 'SPAtmplt', 'IMRPhenomD'):
        # Separate output trees so check_results sees one bank at a time.
        summary = run_approximant(
            approximant, args.jax_platform,
            os.path.join(out_dir, approximant))
        summaries.append(summary)
        deltas = summary['deltas']
        print(
            f"{summary['approximant']}: injected template "
            f"{summary['injected_id']} SNR {summary['snr']:.6g}, "
            f"{summary['ntriggers']} triggers, "
            f"max |dSNR|={deltas['snr']:.3e} |dt|={deltas['time']:.3e}s "
            f"|dchisq|={deltas['chisq']:.3e} |dsg|={deltas['sg_chisq']:.3e}")
    print(f"power-chisq unit relative delta {chisq_rel:.3e}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
