# Small JAX live and offline search checks

For native/JAX CPU/GPU serial and batch comparisons with ripplegw PhenomD and
regular LAL TaylorF2, see [GPU_TESTS.md](GPU_TESTS.md). That runner produces
eight-second SNR surfaces, sampled veto curves, difference plots, CSV timings,
and numerical acceptance reports. It includes P100-specific installation.

See [SPEED.md](SPEED.md) for the subsequent CPU surface optimization, 16-test
validation, and paired timing results. The timings below record the earlier
implementation and are retained as historical measurements.

Run from the repository root in an environment containing PyCBC, JAX, ripplegw,
and pytest:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  python examples/search/jax/compare_searches.py \
    --output examples/search/jax/results/cpu.json --repeats 11 --profile

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  python -m pytest -q test/test_jax_live_filter.py test/test_jax_search_workflows.py
```

`--jax-platform gpu` selects GPU execution and fails explicitly if no GPU is
visible. The saved results here use JAX 0.10.2 on CPU only. A GPU benchmark
must use a CUDA-enabled JAX installation on an actual GPU machine.

## What is tested

The example draws 20 aligned-spin IMRPhenomD templates with seed 20261116,
uses the first 10 as the smaller bank, and analyzes eight valid seconds at
2048 Hz with a 20 Hz low-frequency cutoff and a shared 32-second padded FFT
grid. It checks colored Gaussian noise and the same noise plus a bank
injection (optimal SNR 20). The SNR threshold is 4.5; power chi-squared has
eight bins, and sine-Gaussian activation uses reweighted SNR >= 4.

- **Live reference:** `LiveBatchMatchedFilter.process_data`, including its
  per-template peak selection and the actual power/SG veto implementations.
- **Offline reference:** the production `MatchedFilterControl` used by
  `pycbc_inspiral`, with one-template-at-a-time filtering, one-second
  symmetric clustering, and both production vetoes at every retained peak.
- **JAX serial:** `JaxLiveBatchMatchedFilter` constrained to one template per
  duration chunk (`maxelements = FFT length`).
- **JAX batch:** the same implementation with all same-duration templates in
  one chunk. The offline adapter copies the SNR surface to CPU for the same
  PyCBC symmetric clustering, then computes JAX vetoes at all clustered peaks.
  Multiple peaks belonging to the same template retain their individual
  indices. Its implementation is in `offline_jax` in `compare_searches.py`.

CPU references use their original bank waveforms. JAX performs its existing
waveform generation and startup comparison; the references are not replaced
with JAX waveforms to improve agreement. Both use identical data and PSDs.
The same fixture supports empty trigger sets at higher SNR thresholds.

This is filtering-workflow validation, not a complete production search:
`bin/pycbc_live` is not launched under MPI, no coincidences or uploads are
made, and the Pegasus `examples/search/master.sh` workflow is not executed.
There is no JAX CLI backend for `bin/pycbc_inspiral` in this change: the JAX
offline example is an explicit adapter. `mpi4py` is absent from the test
Python environment. Full MPI and production offline integration remain
separate acceptance checks, as do real GPU timing and GPU numerical parity.

## Results from this machine

All eight cases passed for both JAX serial and batch, matching trigger counts,
template IDs, veto degrees of freedom, and peak sample indices. Live returned
3/10/5/18 triggers (10-template noise/injection, 20-template noise/injection);
offline returned 3/13/5/23. The additional offline peaks are expected and
explicitly checked rather than reduced to one peak per template.

Across both workflows the largest differences were:

- SNR: 4.57e-4; peak time: zero samples; wrapped peak phase: 3.31e-4 radians.
- Reduced power chi-squared: 8.88e-4; sine-Gaussian chi-squared: 8.99e-4.

The gate uses relative SNR tolerance 1e-4 (plus 1e-3 absolute for SNR < 5),
and 1e-3 absolute + 1e-4 relative for both vetoes. Matching is by template ID
and peak time, not by output row position. The reference must recover the
injected template, so empty outputs cannot falsely pass the injection checks.
The original TaylorF2, SPAtmplt, and IMRPhenomD live parity example also passes.
The automated suite contains 11 passing tests, including fresh data blocks,
new PSD objects, caps, aborts, inactive SG vetoes, serial strain reuse,
frequency-support truncation, and repeated template rows at different offline
time indices.

Median warm-call times in milliseconds (11 synchronized repetitions):

| Injection case | Existing CPU | JAX serial | JAX batch |
| --- | ---: | ---: | ---: |
| Live, 10 templates | 4.09 | 12.71 | 8.41 |
| Live, 20 templates | 9.51 | 23.92 | 16.65 |
| Offline, 10 templates | 5.19 | 16.46 | 12.97 |
| Offline, 20 templates | 9.81 | 27.22 | 20.34 |

`results/cpu.json` contains all timings (first call, median, min, max), setup
times, parity residuals, and the actual trigger columns from every mode.
`results/cpu.profile.txt` contains profiles of 20 further warm batch calls per
case. First calls include compilation; warm-call timings exclude template
construction and JIT compilation. They include trigger construction, host
synchronization, and vetoes; offline also includes surface copying and CPU
clustering. Data generation and conditioning are excluded in every mode.
Modes run sequentially and share caches, so setup times are descriptive,
not independent cold-start benchmarks. CPU PyCBC uses one OpenMP thread;
JAX may use its own internal thread pool. These are latency comparisons,
not controlled equal-core CPU-throughput measurements.

## Implemented efficiency changes

1. Transfer strain once per FFT grid per data block, even for serial chunks;
   remove a redundant scatter copy. Reset this cache for every new block.
2. Transfer the small vectors of peaks and indices together, without issuing
   a separate device gather for each output column.
3. Evaluate power chi-squared only for retained candidate rows (after the live
   trigger cap), using power-of-two padding to limit the number of compiled
   candidate shapes. Repeated padded rows are discarded before output.
4. Sum only the frequency support up to the last bin edge, skipping the zero
   FFT half, while keeping the **original FFT length** in the phase shift.
   Preserve the float64 phase/prefix accumulation needed for veto parity.
5. Apply the reweighted-SNR SG gate before evaluating tiles. Fuse selected-row
   gathers with SG computation. The offline example similarly fuses gathers
   and power/SG evaluation for arbitrary candidate time indices.

The isolated power-veto benchmark compares the previous full-group algorithm
with the candidate-only kernel using the same device inputs and accumulation
precision. For the 20-template group, retaining 1, 4, or all 20 rows took
0.46, 1.48, or 3.70 ms, versus 14.77 ms for full-group evaluation over the
65536-point buffer (32x, 10x, or 4x in this run). The optimized support is
32768 frequency samples. All selected raw veto values matched the full
calculation exactly in the saved run. These are **isolated kernel savings**,
not whole-search speedups; the full SNR surface remains a substantial cost.

## Next optimizations to measure

The profiles include asynchronous kernel waiting inside `device_get` and
`numpy.asarray`; those entries must not be interpreted purely as PCIe copy
cost. On CPU they primarily include waiting for surface/veto computation.
JAX batching is faster than JAX serial here, but FFTW-based PyCBC still wins
for these small CPU banks.

The next useful experiments are:

1. Benchmark representative bank sizes and duration distributions on a GPU,
   including data-transfer, PSD-refresh, startup, and steady-state costs;
   tune chunk sizes to measured memory and latency limits rather than a
   single fixed maximum. Measure CPU and GPU crossover before routing work.
2. Add a peak-only live kernel that avoids retaining the normalized full
   time/template surface when diagnostics do not need it. Keep the existing
   surface kernel for diagnostics and offline clustering.
3. Implement and validate offline thresholding/clustering on the device.
   The current offline adapter copies the full surface to the CPU. Transfer
   only retained candidates after clustering and keep the peak vetoes there.
4. Batch waveform startup by shared frequency grid and cache compiled
   generators. Current startup generates and compares templates individually.
5. Explore segmented bin reductions to replace the float64 prefix scan,
   keeping the present CPU-veto accuracy tests and cancellation-sensitive
   injections as acceptance gates. Do not lower precision solely for speed.

Multibanding, heterodyning, and reduced-order template representations require
additional scientific validation and are not introduced by these checks.
