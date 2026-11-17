# Native, JAX CPU, and JAX GPU live-search comparison

Use the same Python environment for all modes, with PyCBC installed from this
branch, MPI/mpi4py, LAL frame support, FFTW, matplotlib, JAX and ripplegw.
The P100-compatible CUDA/JAX installation and scratch/micromamba setup are in
[GPU_TESTS.md](../search/jax/GPU_TESTS.md). Do not install a current CUDA 13
wheel on P100; use the compatible pinned CUDA 12 stack documented there.
Run from the repository root, with the activated environment's Python:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pycbc-matplotlib"
# Do not leave JAX_PLATFORMS=cpu set when testing GPUs.
unset JAX_PLATFORMS

python examples/live/benchmark_imrphenomd.py \
  --output "$SCRATCH/pycbc-live-imrd-1gpu" \
  --templates 20 --workers 1 --repeats 3 --gpu required --autotune
```

This generates one shared seeded H1/L1 Gaussian-noise frame fixture, an aligned
spin IMRPhenomD bank, and one separate injection time per template (optimal SNR
40). Each backend runs the **actual `bin/pycbc_live` through MPI** on those same
files. Each injected template must recover SNR >=10 in both detectors. One CPU
coordinator is additional to `--workers`. Native filtering uses FFTW. Searches
use eight-second analysis chunks, power chi-square and sine-Gaussian vetoes.
There is no network data download, GraceDB upload, coincidence background,
parameter inference or sky localization in this filtering comparison.
Choose a fresh output directory for each invocation; existing runs are refused.
`--gpu off` explicitly requests only native/JAX CPU validation.

`live_summary.json` contains executed commands, return codes, every wall-time
sample, median speedups and parity failures. `*_live_comparison.png` compares
SNR, vetoes, peak times and phases and plots their differences. Matching uses
analysis-block filename, detector and template ID; missing and extra triggers
fail. Output timing is **end-to-end process wall time**, including MPI startup,
conditioning, bank preparation, JIT compilation, tuning and disk output. It is
not steady-state filter throughput. A failed parity gate returns nonzero and
preserves logs and products. Phase comparisons are wrapped modulo 2π.

Measure waveform generation and matched filtering **separately**:

```bash
python examples/search/jax/validate_cpu_gpu.py \
  --gpu required --fft-backend fftw --approximants IMRPhenomD \
  --bank-sizes 10 20 --repeats 11 --autotune --tuning-repeats 5 \
  --benchmark-components --output "$SCRATCH/pycbc-imrd-components"
```

This second command tests independent eight-second valid blocks: noise and a
separate injection for **every** template (all recovered SNRs must be >=10).
It is a live-filter API test, not an MPI search. Each bank directory contains
`waveform_timings.json` and `component_timings.png`. Waveform times compare
native LAL IMRPhenomD with the production ripplegw generator on CPU and GPU,
using the same grid, distance and host-ready complex64 outputs. These generators
loop serially over templates; filtering batches do not imply waveform batching.
First-use times can include compilation, but are not guaranteed clean-process
cold compilation measurements. Warm samples exclude the first call.

Each case's `case.json` separates `matched_filter_only` (correlation, IFFT and
peak reduction, preloaded inputs/PSD, **no vetoes**) from the main timings
(filtering + vetoes + host result transfer). Native serial and batch timings
and JAX serial, batch and tuned timings are reported. GPU calls are synchronized;
JAX dispatch alone is not timed. Generation/setup is excluded from warm filtering.
The ripple generator reuses a JIT-compiled executable across bank templates,
with masses/spins and reference frequency as dynamic inputs. Compilation and
host transfer are included in the appropriate first-use/host-ready timings.
The matched-filter-only and waveform speedups use their own native baselines;
never divide full-process wall time by a kernel time.

Per-case `timeseries.png`, `surface.png`, `peaks.png`, `filter_residuals.png` and
`arrays.npz` retain complex SNR, phase/time-optimized peaks and power/SG veto
curves. Exact-waveform native controls distinguish FFT/device errors from
LAL-versus-ripple waveform differences. Known strict cross-model off-template
veto failures remain failures; thresholds are not relaxed to manufacture a pass.

## Multiple GPUs and batch tuning

For two GPUs on one node:

```bash
# Prefer scheduler-provided visibility; do not overwrite it with this example.
# When setting manually, select only GPUs allocated to this job.
export CUDA_VISIBLE_DEVICES=0,1
python examples/live/benchmark_imrphenomd.py \
  --output "$SCRATCH/pycbc-live-imrd-2gpu" \
  --templates 20 --workers 2 --repeats 3 --gpu required --autotune
```

`pycbc_live` uses the existing disjoint MPI bank shards. The new collective
binding chooses one distinct visible GPU per local filtering rank **before
importing JAX**, respecting each rank's existing scheduler visibility mask.
Rank 0 and CPU filtering ranks consume no GPU assignment. Each GPU process is
restricted to its selected identifier and uses logical device 0. The log records
the assignment. This is MPI bank parallelism, not a single JAX multi-device array
or a guarantee of balanced work for heterogeneous templates. More workers than
visible GPUs fail unless `--jax-allow-gpu-sharing` is explicitly passed. A
per-node mixture of numeric and UUID masks, or ambiguous UUID prefixes, fails
rather than risk accidental sharing. For scheduler-isolated namespaces whose
numeric indices are not comparable, use full GPU UUID masks.

Direct `pycbc_live` controls:

```text
--filter-backend jax --jax-platform gpu --jax-device-id auto
--jax-autotune --jax-tuning-candidates 1 2 4 8 16 32 64
--jax-tuning-repeats 5 --jax-memory-budget-mb 8000
--jax-max-batch-latency-ms 500
```

`--jax-device-id` accepts `auto` or an index **within the existing visibility
mask**. It replaces `--processing-device-id` for JAX MPI assignment. CPU/GPU
mixed filtering uses `--rank-backend cpu,jax` and `--jax-platform gpu`.
All nodes must have the checkout, environment and shared frame/bank/output paths.
The benchmark driver launches a local MPI job; use the recorded command with
your scheduler's launcher for a multi-node deployment.

Each worker independently tunes once on its first actual block. Candidates are
whole-bank layouts with different maximum template counts per chunk. The tuner
excludes compilation/first-use PSD preparation, then times synchronized complete
filtering plus vetoes for the actual and forced-all-template trigger workloads.
It minimizes the worse median; an optional latency bound applies to the worse
p95 **for the entire local bank**, not just one chunk. Decisions, raw samples,
layout and tuning cost appear in JAX logs and API `case.json` reports. Tuning
cost remains included in real live wall time. Reuse the engine for long runs.
There is no persistent cross-job cache or continuous automatic retuning.

All templates and surface buffers still reside on the assigned device.
The memory estimate includes the complete resident bank and a scratch allowance,
using 60% of reported free GPU memory (50% of available CPU memory), or an
explicit budget. It is a preflight estimate, not an exact XLA allocator bound;
SG tile sizes and compiler workspaces can vary. Leave allocation headroom,
use more GPU workers for a large bank, and supply a conservative explicit budget
when device memory statistics are unavailable. A budget or latency failure is
explicit. The best measured candidate is not a guarantee of optimal future
performance; PSD changes and trigger load can change costs.

To check complete time surfaces on each GPU as well as the combined MPI run,
repeat the component command with `--gpu-index 0` and `--gpu-index 1`, using
separate output directories. The MPI benchmark compares distributed peak/veto
products; it does not dump distributed full time surfaces.

Regression checks (the last command uses simulated GPU masks, not CUDA):

```bash
python -m pytest -q test/test_jax_devices_tuning.py test/test_jax_live_benchmark.py
PYCBC_TEST_MPI=1 python -m pytest -q test/test_jax_devices_mpi.py
```
