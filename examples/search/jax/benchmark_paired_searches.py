#!/usr/bin/env python
"""Interleave old fused, optimized split, and PyCBC CPU search timings.

Uses identical generated templates, data, PSDs, thresholds and vetoes. The
reference changes only SNR surface evaluation, making the CPU kernel change
reviewable without checking out an old branch or timing different processes.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/pycbc-matplotlib')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'examples/live'))

import numpy as np
from pycbc.filter import LiveBatchMatchedFilter
from pycbc.filter.jax_matchedfilter import JaxLiveBatchMatchedFilter, _surface_kernel
from pycbc.scheme import CPUScheme
from pycbc.vetoes.sgchisq import SingleDetSGChisq
from compare_searches import offline_cpu, offline_jax, compare
import plot_batch_vs_serial as fixture


class FusedCPUReference(JaxLiveBatchMatchedFilter):
    """Previous surface implementation; all remaining code stays identical."""
    def _evaluate_surface(self, state, valid_start, valid_len):
        kernel = _surface_kernel(int(valid_len))
        q, rho = state.q, state.snr
        state.q = state.snr = None
        (state.q, state.snr, state.peak, state.peak_unnorm,
         state.index, state.abs_snr) = kernel(
            state.h, state.strain[:state.flen], state.norms, q, rho,
            np.int32(valid_start))
        return state.abs_snr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank-sizes', type=int, nargs='+', default=[10, 20])
    parser.add_argument('--repeats', type=int, default=31)
    parser.add_argument('--output', type=Path, default=Path('/tmp/paired-searches.json'))
    args = parser.parse_args()
    if args.repeats < 1 or min(args.bank_sizes) < 1:
        parser.error('positive repeats and bank sizes required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixture.OUT_DIR = str(args.output.parent / (args.output.stem + '_fixtures'))
    rows = fixture.draw_rows(max(args.bank_sizes), np.random.default_rng(fixture.SEED))
    report = dict(seed=fixture.SEED, repeats=args.repeats, cases=[],
        scope='Same-process interleaved warm calls; only fused/split surface differs. '
              'Both offline adapters copy only the valid window to CPU clustering.')
    with CPUScheme(num_threads=1):
        bank, templates, df = fixture.build_templates(rows)
        ntime = (len(templates[0]) - 1) * 2
        psd = fixture.analytic_psd(len(templates[0]), df)
        noise = np.asarray(fixture.noise_from_string(
            fixture.PSD_NAME, ntime, 1 / fixture.SAMPLE_RATE,
            seed=fixture.SEED, low_frequency_cutoff=fixture.F_LOW))
        temporary = fixture.make_reader(fixture.overwhiten(noise, psd))
        start, _ = fixture.valid_bounds(templates[0], temporary)
        injection, _, _ = fixture.injection_waveform(templates[0], psd, ntime, start)
        for n in args.bank_sizes:
            engines = {}
            for name, constructor in [('cpu', LiveBatchMatchedFilter),
                ('fused', FusedCPUReference), ('split', JaxLiveBatchMatchedFilter)]:
                sg = SingleDetSGChisq(bank, num_bins='8', snr_threshold=4,
                                     chisq_locations=fixture.SG_LOCATIONS)
                engines[name] = constructor(list(templates[:n]), snr_threshold=4.5,
                                            chisq_bins='8', sg_chisq=sg)
            for dataset, data in [('noise', noise), ('injection', noise + injection)]:
                reader = fixture.make_reader(fixture.overwhiten(data, psd))
                for workflow in ['live', 'offline']:
                    if workflow == 'live':
                        calls = {name: (lambda e=e: e.process_data(reader))
                                 for name, e in engines.items()}
                    else:
                        calls = {'cpu': offline_cpu(templates[:n], reader,
                                        engines['cpu'].sg_chisq, 4.5, 2048)}
                        calls.update({name: offline_jax(engines[name], reader, 4.5, 2048)
                                      for name in ['fused', 'split']})
                    for call in calls.values():
                        call()  # Compile, initialize PSD weights, and warm up.
                    names = list(calls)
                    samples = {name: [] for name in names}
                    products = {}
                    for repeat in range(args.repeats):
                        order = names[repeat % len(names):] + names[:repeat % len(names)]
                        for name in order:
                            started = time.perf_counter()
                            products[name] = calls[name]()
                            samples[name].append(time.perf_counter() - started)
                    parity = {name: compare(products['cpu'], products[name], reader.sample_rate)
                              for name in ['fused', 'split']}
                    item = dict(workflow=workflow, templates=n, dataset=dataset,
                        parity=parity, timings={name: dict(median_s=statistics.median(v),
                            min_s=min(v), max_s=max(v)) for name, v in samples.items()})
                    item['speedup'] = item['timings']['fused']['median_s'] / item['timings']['split']['median_s']
                    report['cases'].append(item)
                    print(json.dumps(item), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
