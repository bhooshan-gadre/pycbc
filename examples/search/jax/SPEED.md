# CPU SNR-surface optimization follow-up

The optimized CPU surface is now selected automatically by
`JaxLiveBatchMatchedFilter`. GPU execution retains the fused kernel. This
follow-up changes CPU surface evaluation and the example's offline window
handling; the previously validated veto accumulation precision is preserved.

## Why CPU JAX was slower

The warm timings exclude JIT compilation, so startup compilation does not
explain the steady-state gap. Investigation identified three concrete costs:

1. On this JAX 0.10.2 CPU runtime, one compiled kernel returning the correlation
   product and FFT-derived surface was slower than dispatching the product,
   FFT, and peak reduction as separate compiled calls. This is a measured
   execution/scheduling effect, not a claim about all XLA versions or GPUs.
2. The live path normalized the entire 32-second padded SNR surface to retain
   just one peak per template. Offline also normalized/copied that full
   buffer even though only eight seconds are valid for clustering.
3. JAX pays dispatch/synchronization costs at each CPU template chunk and
   uses a float64 phase/prefix scan for the power veto. The existing CPU
   veto uses a compiled rotation recurrence (see `point_chisq_code` in
   `pycbc/vetoes/chisq_cpu.pyx`), avoiding a separate transcendental evaluation
   at every frequency. The two implementations have different work and
   memory costs even though their scientific outputs agree.

Profiler entries in `device_get` and `numpy.asarray` include waiting for
asynchronous computation. They do not prove that copies alone dominate.

## Changes retained

- Separate compiled correlation, FFT, and peak stages on CPU. Keep the GPU
  fused path and its donated buffers.
- Find the first maximum in the unnormalized valid window, as the CPU
  reference does, and normalize only the retained peaks for live triggers.
- Materialize the full normalized `state.snr` only when requested, then release
  the redundant raw time series. Existing diagnostic callers can still read
  the full normalized surface.
- Add `state.snr_window(start, length)`. The offline example normalizes and
  copies only its valid window; it preserves the same CPU clustering.

No FFT samples are removed and no veto precision is lowered. Negative
frequency padding, normalization, phase, time indices, and trigger/veto
semantics remain covered by parity checks.

## Timings and validation

The most useful timing comparison is `results/cpu_paired_surface.json`:
31 warmed calls per mode in one process, with the execution order rotated
on every repetition. PyCBC CPU, original fused JAX, and optimized JAX share
identical banks, data, PSDs, thresholds, clustering, and vetoes. The reference
class changes only surface evaluation. Both offline adapters use the valid
window, so that comparison isolates the surface change rather than adding
an artificial full-buffer copy to the old mode. Setup and compilation are
excluded; host trigger products and veto completion are included. The host
has other activity, so these are observed medians, not guaranteed latency.

| Workflow | Templates | Data | PyCBC CPU (ms) | Fused JAX (ms) | Optimized JAX (ms) | JAX speedup |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| live | 10 | noise | 5.26 | 8.74 | 5.13 | 1.71x |
| offline | 10 | noise | 4.97 | 8.46 | 5.64 | 1.50x |
| live | 10 | injection | 5.82 | 9.45 | 7.13 | 1.33x |
| offline | 10 | injection | 6.51 | 11.66 | 9.29 | 1.26x |
| live | 20 | noise | 9.67 | 15.96 | 11.10 | 1.44x |
| offline | 20 | noise | 8.65 | 16.46 | 11.04 | 1.49x |
| live | 20 | injection | 11.25 | 18.27 | 14.20 | 1.29x |
| offline | 20 | injection | 11.69 | 20.92 | 17.02 | 1.23x |

Every paired case passes the same parity gate. The complete serial/batched
live/offline example also passes all eight cases, with new results and
profiles in `results/cpu_after_surface.json` and `.profile.txt`. The earlier
sequential baseline is retained in `results/cpu_before_surface.json`.
Paired timings are preferred over ratios across those separate runs.

Across the updated full example, the maximum differences from PyCBC are:
SNR 4.56e-4, peak time zero samples, wrapped peak phase 3.31e-4 radians,
reduced power chi-squared 8.89e-4, and SG chi-squared 8.99e-4. Trigger IDs,
counts, and degrees of freedom agree. The automated suite passes 16 tests,
including TaylorF2, SPAtmplt, and IMRPhenomD live parity, the offline example,
partial valid windows, equal-maximum tie handling, lazy-surface lifetime,
PSD/data refreshes, and existing candidate-selection/veto checks.

`results/surface_stages.json` isolates the device work with 21 interleaved
calls: at 20 templates the fused surface took 13.18 ms, split peak evaluation
6.37 ms, and split evaluation plus full-surface normalization 8.65 ms. Those
are kernel timings, not whole-search speedups. CPU binding was also tested;
`surface_stages_single_core.json` shows that forcing one core reduced some
single-template overhead but hurt the larger split batches. No affinity or
thread-pool policy is silently imposed by the implementation.

Alternative chunk-prefix reductions and an FFTW Python callback were slower
in exploratory microbenchmarks, so neither was adopted. An experimental
factored phase calculation showed only a small isolated improvement and was
also left out of this patch. The retained changes have full workflow parity
and timing evidence.

## Reproduce

Run from the repository root in the PyCBC/JAX environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  python examples/search/jax/benchmark_surface.py \
    --output examples/search/jax/results/surface_stages.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  python examples/search/jax/benchmark_paired_searches.py \
    --output examples/search/jax/results/cpu_paired_surface.json --repeats 31

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  python examples/search/jax/compare_searches.py \
    --output examples/search/jax/results/cpu_after_surface.json --repeats 15 --profile

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  python -m pytest -q test/test_jax_search_workflows.py test/test_jax_live_filter.py
```

JAX batching is now closer to native CPU performance, but PyCBC CPU generally
remains faster for these small banks. Serial JAX still incurs per-template
dispatch costs and does not show a consistent whole-workflow improvement.
Next useful work is a specialized CPU power-veto implementation and real-GPU
profiling at representative bank sizes. No GPU speed or parity claim is made
from this CPU-only run. MPI/production offline CLI coverage remains as
specified in README.md; this work does not expand those integration claims.
