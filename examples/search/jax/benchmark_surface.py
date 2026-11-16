#!/usr/bin/env python
"""Interleave original fused and split CPU SNR-surface timings.

This isolates device computation. Whole-search timing remains in
compare_searches.py; these are not whole-search speedups.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pycbc.filter.jax_matchedfilter import (
    _cpu_surface, _normalize_surface_kernel, _surface_kernel, configure_jax)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 2, 4, 10, 20])
    parser.add_argument('--repeats', type=int, default=21)
    parser.add_argument('--output', type=Path, default=Path('/tmp/surface-benchmark.json'))
    args = parser.parse_args()
    if args.repeats < 1 or min(args.batches) < 1:
        parser.error('positive batch sizes and repeats required')
    device, _ = configure_jax('cpu')
    import jax
    import jax.numpy as jnp
    ntime, valid_len = 65536, 16384
    start = np.int32(ntime - valid_len - 512)
    rng = np.random.default_rng(48)
    report = dict(device=str(device), jax_version=jax.__version__, seed=48,
                  fft_length=ntime, valid_length=valid_len, cases=[])
    for batch in args.batches:
        with jax.default_device(device):
            h = jnp.asarray((rng.normal(size=(batch, ntime//2+1))
                             + 1j*rng.normal(size=(batch, ntime//2+1))).astype('complex64'))
            strain = h[0]
            norms = jnp.asarray(rng.uniform(0.1, 1, batch).astype('float32'))
            q = jnp.zeros((batch, ntime), dtype=jnp.complex64)
            rho = jnp.zeros_like(q)
        fused = _surface_kernel(valid_len)
        def original():
            nonlocal q, rho
            result = fused(h, strain, norms, q, rho, start)
            q, rho = result[:2]
            return result
        def split_peaks():
            return _cpu_surface(h, strain, norms, start, valid_len)
        def split_with_surface():
            product, raw, *peaks = split_peaks()
            return product, _normalize_surface_kernel()(raw, norms), *peaks
        modes = dict(fused=original, split_peaks=split_peaks,
                     split_full_surface=split_with_surface)
        for call in modes.values():
            jax.block_until_ready(call())
        reference = tuple(np.array(value, copy=True) for value in original())
        actual = tuple(np.array(value, copy=True) for value in split_with_surface())
        np.testing.assert_array_equal(actual[4], reference[4])
        for i in [0, 1, 2, 3, 5]:
            np.testing.assert_allclose(actual[i], reference[i], rtol=1e-5, atol=1e-4)
        samples = {name: [] for name in modes}
        names = list(modes)
        for repeat in range(args.repeats):
            # Rotate the first mode so no variant always benefits from order.
            order = names[repeat % len(names):] + names[:repeat % len(names)]
            for name in order:
                started = time.perf_counter()
                jax.block_until_ready(modes[name]())
                samples[name].append(time.perf_counter() - started)
        case = dict(templates=batch, timings={name: dict(
            median_s=statistics.median(values), min_s=min(values), max_s=max(values))
            for name, values in samples.items()})
        report['cases'].append(case)
        print(json.dumps(case), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
