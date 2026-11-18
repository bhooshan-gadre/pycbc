# Diagnose and reduce JAX live-search latency

For strict compiled/batched ripple IMRPhenomD, native-compatible CUDA vetoes,
the multi-detector PSD cache and updated GPU validation commands, see
[COMPILED_RIPPLE_GPU.md](COMPILED_RIPPLE_GPU.md).

The supplied native-example measurements were:

| Backend | First run (s) | Next run (s) | Third run (s) |
|---|---:|---:|---:|
| Native | 157.111 | 158.782 | 153.526 |
| JAX CPU | 337.023 | 232.814 | 232.087 |
| JAX GPU | 844.244 | 225.961 | 224.515 |

The later GPU runs are about 1.44 times slower than the median native run and
only a little faster than JAX CPU. The first GPU run adds about 619 seconds
relative to the later runs. These wall times establish cold-run overhead and
an additional warmed-run gap, but cannot tell us how much is compilation,
conditioning, I/O, device transfer, filtering or veto evaluation. Do not call
the whole 619 seconds "FFT compilation" without a profile. No GPU acceleration
factor for the new code is claimed until it is run on the target hardware.

## Changes

GPU live filtering now reduces the **unnormalized** IFFT to peaks and normalizes
only the selected samples. It no longer materializes a normalized full SNR
surface for a peaks-only live call. Native peak selection is also performed
before normalization. Full SNR surfaces and offline windows remain available
through lazy normalization for diagnostics.

`--jax-preprocessing` enables the device preprocessing backend:

* `off` (default): unchanged native StrainBuffer preprocessing.
* `whiten`: maintain a device circular strain buffer, perform the forward FFT,
  PSD division, inverse FFT, trimming/Tukey taper and final FFT on the selected
  JAX device. Existing PSD preparation stays native. Pass the resulting device
  frequency series directly to the matched filter without a host roundtrip.
* `condition`: also move highpass FIR, LDAS FIR/downsampling, batched Welch
  estimation, inverse-PSD truncation, and the spectral math of automatic glitch
  detection onto that device. FIR responses and compiled functions are cached
  across blocks and shared between detectors where shapes/configuration match.

The resident buffer receives one initial full upload, then only the new/changed
tail and any earlier samples affected by a gate. Its head advances by index;
it is not copied/rolled in full on the GPU every block. PSD replacements
invalidate the spectrum cache. The native host strain view is retained because
native frame I/O, state/DQ bookkeeping, FindChirp clustering, gate decisions,
followup and output still use it. FIR output returns to this view; this is not
a claim that all frame decoding and control flow run on GPU. Raw frame decoding,
CPU interpolation of short PSD vectors, clustering and final coincidence/FAR
calculation remain CPU operations. Native search thresholds are unchanged.

All these operations run on the explicitly selected JAX device. The same code
can be tested on JAX CPU; GPU requests never fall back to CPU. The new
preprocessing modes are opt-in until GPU parity and throughput are checked on
the target system. Small CPU fixtures pass the existing output parity gates;
this does not certify every template bank or GPU.

## Profile the actual native example

Use the existing bank/frames as described in
[NATIVE_BACKEND_COMPARISON.md](NATIVE_BACKEND_COMPARISON.md). From the checkout
root, with the environment active and scheduler GPU allocation preserved:

```bash
python examples/live/run_backend_comparison.py \
  --launcher mpirun \
  --example-dir /path/to/native/fixtures \
  --output /scratch/bug/live-profile-original \
  --modes native jax_gpu --repeats 1 --profile-stages --no-autotune

python examples/live/analyze_backend_performance.py \
  --run-dir /scratch/bug/live-profile-original
```

The first command runs the real MPI search and compares scientific outputs as
usual. Additional `stages/rank*.jsonl` files contain startup and per-detector,
per-block timings for every MPI rank. JAX stage timings explicitly synchronize
whitening/transfers, PSD weights and the filtering surface; veto output is
host-ready. `host_selection_s` is the residual host selection/packing cost.

The analyzer writes `performance/summary.json`, `stages.csv` and `stages.png`.
It records bank/device setup, all-detector conditioning, filtering, root MPI
wait, coincidence/postprocessing, and individual JAX stages. It identifies the
largest measured serial stage in each rank. Times from concurrent ranks are
**not added together**. The root's outside-loop estimate includes imports,
setup and shutdown; it is not a direct compilation measurement. First-call,
median, p95 and max values remain available so cold costs are visible. Excluding
one initial call does not guarantee that later, previously unseen shapes have
finished compiling. Inactive startup blocks are retained in these profiles.
The GPU model and local template count are in each rank's startup record.

Profiling introduces extra synchronization and can reduce overlap. Use it to
attribute costs, then omit `--profile-stages` for final speedup measurements.
The ordinary comparison still writes precise block latencies and process wall
times. Replayed historical GPS lag is never used as processing latency.

## Run with device preprocessing

```bash
python examples/live/run_backend_comparison.py \
  --launcher mpirun \
  --example-dir /path/to/native/fixtures \
  --output /scratch/bug/live-profile-conditioned \
  --modes native jax_gpu --repeats 1 \
  --jax-preprocessing condition --profile-stages --no-autotune

python examples/live/analyze_backend_performance.py \
  --run-dir /scratch/bug/live-profile-conditioned
```

Use a fresh output directory each time. To isolate only the whitening/transfer
change first, use `--jax-preprocessing whiten`. To compare final runtime, rerun
without profiling and use three repeats. Keep batch tuning identical in your
control and treatment runs; `--autotune` includes its cost in total
wall time. Changing bank size/batch size and preprocessing simultaneously will
make attribution difficult.

### Tune batches and remove batch-by-batch host waits

The comparison runner now enables autotuning for GPU modes by default. CPU
tuning requires `--autotune`; `--no-autotune` disables both. Filtering was
already batched without tuning, using the native maximum-elements limit.
The tuner measures synchronized full-bank filtering plus vetoes for candidate
sizes 1 through 256 (powers of two), plus the original largest batch where
memory permits. It chooses the lowest worst-workload median under the supplied
memory/latency constraints. This is a measured choice among candidates, not a
guarantee of a global optimum. Use `--tuning-candidates` and `--tuning-repeats`
to control the sweep.

GPU filtering now enqueues all duration batches before collecting their peak
vectors with one host request. It previously waited for each batch's peaks
before dispatching the next. Profiling still synchronizes individual stages
for attribution; omit profiling to measure the dispatch improvement.

```bash
python examples/live/run_backend_comparison.py \
  --launcher mpirun --example-dir /path/to/native/fixtures \
  --output /scratch/bug/live-tuned --modes native jax_gpu --repeats 2 \
  --jax-preprocessing condition --autotune --profile-stages
python examples/live/analyze_backend_performance.py \
  --run-dir /scratch/bug/live-tuned
```

The analyzer now prints per-rank stage costs, actual batch-size histograms,
tuning startup time, and detailed parity failures. With profiling enabled,
`stages/rankN-tuning.json` preserves every measured candidate. Tuning is paid
again in each new MPI process; compare warm block latencies separately from
complete wall time. A short replay can spend more time tuning than it saves.
The tuning sweep disables per-stage profiling barriers while retaining a
host-ready full-bank result for every measurement, so it selects a production
layout even when the subsequent replay is profiled.

### Recheck existing output without repeating the search

```bash
python examples/live/compare_backend_runs.py --run-dir /scratch/bug/live-tuned
python examples/live/analyze_backend_performance.py --run-dir /scratch/bug/live-tuned
```

The comparator matches candidate directories by timestamp and ignores the
four random lowercase characters appended by `get_candidate_path`. It still
compares every product, and rejects ambiguous duplicate candidate identities.
Structured numeric `stat` records are checked field by field using the existing
numeric tolerances, with integer identifiers/counts checked exactly. These
fix comparator false failures; differing `background/count` values remain a
failure and now show the actual native/JAX counts. Do not infer search parity
from disappearance of the random-path errors alone. Repeating native runs in
the same comparison also checks native reproducibility against the first run.

### Rank 0 can remain the bottleneck

By default rank 0 conditions the same frames on CPU for detector status,
coincidence and followup. Worker-only GPU preprocessing cannot remove that
CPU floor. The new `--coordinator-gpu` option also moves rank-0 conditioning
math to GPU. It requires `--jax-preprocessing condition` and **one additional
allocated GPU** with the default distinct-device binding. With two MPI ranks,
this means two GPUs: one coordinator GPU and one filtering GPU.

```bash
python examples/live/run_backend_comparison.py \
  --launcher mpirun \
  --example-dir /path/to/native/fixtures \
  --output /scratch/bug/live-profile-coordinator-gpu \
  --modes native jax_gpu --repeats 1 \
  --jax-preprocessing condition --coordinator-gpu --profile-stages
```

If only one GPU is allocated, sharing the coordinator and worker GPU requires
explicit `--allow-gpu-sharing`. This adds a second GPU context and competing
work; measure it rather than assuming it helps. The default never silently
oversubscribes a GPU. Even with coordinator conditioning offloaded, frame I/O,
DQ decisions, clustering and coincidence/background remain CPU work; the stage
profiles reveal whether these now limit throughput.

Direct `bin/pycbc_live` equivalents are `--jax-preprocessing condition`,
`--jax-coordinator-preprocessing`, `--jax-allow-gpu-sharing`, and
`--stage-profile-dir DIR`. No global CUDA processing scheme is used: native
buffer metadata stays under the CPU scheme and JAX owns the selected device.

## Numerical verification on P100

```bash
# Exercise device preprocessing against independent native FFTW references.
PYCBC_JAX_TEST_PLATFORM=gpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q test/test_jax_live_preprocessing.py
```

This tests conditioning, resampling, Welch PSDs, whitening, resident-buffer
updates, older gates, PSD changes and peak-only kernel equivalence. It fails if
the GPU backend is unavailable. The native references stay on CPU. For full
SNR/veto time-series/surface plots on the GPU, the existing
`validate_cpu_gpu.py --gpu required` runner exercises the changed filtering
kernel. For the actual preprocessing plus native example, the runner above
compares stored trigger, PSD, veto and coincidence products and returns nonzero
on a parity failure. It does not weaken thresholds to obtain a pass.

Local validation used JAX CPU for the device backend because no GPU is attached
to the development machine. Real MPI testing on a small shared fixture included
PSD recalculation, autogating and coincidence background, and passed native
output parity. Coordinator GPU offload and GPU speedups need validation on the
allocated P100 hardware.
