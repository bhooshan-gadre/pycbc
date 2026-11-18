# Compiled ripple IMRPhenomD live search

Use this mode for native FFTW and JAX filtering of compiled, batched ripplegw
IMRPhenomD on the same template-bank parameters and frame data. The explicit
`--reference-waveforms ripple` option provides identical-model CPU reference
templates. `--reference-waveforms lal` (the default) retains the separate
LAL-versus-ripple agreement study.
The original native example uses SPAtmplt/SEOBNRv4_ROM; selecting IMRPhenomD
changes the approximant for **every backend**, while retaining the bank
parameters, data, thresholds and search settings. The default comparison mode
still preserves the original approximants and exact native bank waveforms.

## Changes driven by the supplied profile

The reported GPU run spent 87.84 s filtering versus 45.53 s native, with about
45.74 s in vetoes across H1/V1/L1, and 13.98 s in PSD-weight setup. Both ranks
also spent about 100 s advancing/conditioning data. These are stage totals;
concurrent ranks must not be added together. The extra triggers, veto errors
and background-count differences were scientific failures, not just filenames.

* The filter now caches three PSD-weight/tile bundles per duration group.
  Returning from V1 to an unchanged H1 PSD reuses its weights instead of
  rebuilding them. Cached PSD objects are retained to prevent identity reuse;
  the cache is bounded and its extra memory is included in tuning estimates.
* Power chi-squared now follows native per-bin float32 rotations and summation.
  A mathematically accurate complex exponential/prefix sum is not the same
  arithmetic as native's long recurrence. The GPU implementation uses two
  compiled CUDA kernels, parallel over candidates and bins, sharing JAX arrays
  through DLPack. It avoids the full complex128 prefix workspace and an XLA
  launch per frequency-loop iteration. It does not copy spectra to the CPU.
  CPU live filtering uses the native Cython point-veto loops for CPU-resident
  correlations; FFT/matched-filter surfaces remain JAX. GPU arrays never use
  this CPU path.
* SG time shifts now use native's float32 recurrence, with double-precision
  trigonometric initialization and resets every 100 frequency bins.
* Strict ripple mode uses `jax.jit(jax.vmap(...))`, float64 waveform evaluation
  and complex64 storage, with dynamic masses/spins/cutoffs/reference frequencies.
  Small banks do not generate 32 unnecessary rows; long grids cap waveform
  batch size to limit float64 intermediates. The final chunk is padded to reuse
  its executable. Generated banks stay on their selected device. A host copy
  is used for startup bank comparison; the CPU control bank is also checked
  against LAL before replacement. There is no waveform fallback in this mode.
* Filtering remains batched with GPU autotuning, measured without artificial
  stage-profiling barriers. Waveforms are generated once, outside the stream
  loop. Startup reports separate LAL bank generation, ripple generation/JIT,
  and waveform validation times. LAL references are intentionally still built
  for validation and that cost stays in complete process wall time.
* Conditioning profiles now separate frame I/O, state/DQ, FIR/downsampling,
  stitching, autogating and initial PSD estimation. A large `advance_s` alone
  is not evidence that GPU arithmetic can remove the time: frame decoding and
  native DQ/control operations remain on CPU.
* Incremental frame reading now reuses unchanged streams instead of reopening
  the same frame and retrieving metadata on every analysis block. Reuse checks
  paths, file identity, size, modification/change times and stream identity;
  file replacements, modifications, missing files and explicit refreshes
  invalidate it. This affects native and JAX I/O equally and preserves samples.

## Environment

Keep the existing working CUDA-12 JAX and ripplegw environment on the P100.
The native-compatible CUDA veto kernels additionally need CuPy/NVRTC:

```bash
# In the active scratch micromamba environment, with CUDA-12 JAX already working:
mkdir -p /scratch/bug/pycbc_gpu/cache/cupy
export CUPY_CACHE_DIR=/scratch/bug/pycbc_gpu/cache/cupy
python -m pip install --no-cache-dir cupy-cuda12x
```

Use only the CuPy package matching the installed CUDA family; see the
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).
NVRTC compiles/caches the kernels without a system `nvcc` invocation. A
missing CuPy/runtime/compiler dependency is an explicit GPU error; no silent
CPU veto fallback is permitted. CPU JAX runs do not need CuPy. The optional
`jax-cuda12` PyCBC extra adds CuPy/ripple dependencies but does not replace an
existing GPU-enabled JAX installation.

## GPU correctness and component timing first

```bash
PYCBC_JAX_TEST_PLATFORM=gpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q test/test_jax_live_preprocessing.py \
    test/test_jax_ripple_bank.py test/test_jax_cuda_source.py

python examples/search/jax/validate_cpu_gpu.py \
  --gpu required --approximants IMRPhenomD --bank-sizes 2 \
  --reference-waveforms ripple \
  --fft-backend fftw --autotune --tuning-repeats 3 --repeats 3 \
  --benchmark-components --veto-samples 33 \
  --output /scratch/bug/pycbc_gpu/ripple-gpu-validation
```

The required-GPU integration test checks actual CUDA launches/DLPack results
against native Cython vetoes, including repeated selected rows and arbitrary
candidate times. With the environment variable unset, that GPU test is skipped;
the host C++ source-arithmetic test is explicitly **not** a GPU execution test.

The validation runner tests noise and a separate injection in every template,
each recovered at SNR >=10. It emits 8 s SNR surfaces/time series, veto samples,
difference/peak plots and machine-readable errors. `waveform_timings.json`
separates serial LAL, serial ripple and compiled batched ripple generation;
`component_timings.png` separates generation from matched filtering without
vetoes. Cold JIT/generation and warm medians are labeled separately. After a
small smoke pass, use `--bank-sizes 10 20` for the larger comparison. Add
`TaylorF2` to `--approximants` to exercise regular native TF2 as well.

## Actual live example

From the checkout root (fixtures default to `examples/live`):

```bash
python examples/live/run_backend_comparison.py \
  --launcher mpirun --prepare \
  --output /scratch/bug/pycbc_gpu/live-ripple-profile \
  --modes native jax_cpu jax_gpu --repeats 2 \
  --waveform-model IMRPhenomD --jax-preprocessing condition \
  --reference-waveforms ripple \
  --autotune --profile-stages

python examples/live/analyze_backend_performance.py \
  --run-dir /scratch/bug/pycbc_gpu/live-ripple-profile
```

`--waveform-model IMRPhenomD` selects strict ripple generation for JAX.
`--reference-waveforms ripple` installs CPU-generated ripple arrays in the
filtering templates on every backend, keeping IDs/parameters and native FFTW
processing. Normalization and veto bins are then computed from those arrays.
The initial LAL comparison and CPU-reference generation/JIT are recorded
separately, and their costs remain in total wall time. The direct live-engine
options are `--bank-waveform-source ripple` and
`--jax-waveform-source ripple`, requiring an IMRPhenomD bank. Verify startup
`jax_waveforms.implementation=ripplegw-jit-vmap` and
`veto_backend=cuda-nvrtc-native-recurrence` on GPU rank 1. The analyzer prints
actual filtering batch sizes and per-candidate tuning measurements are retained.

For the cross-model agreement study, select `--reference-waveforms lal` in
either runner. No SNR/veto amplitude thresholds have been relaxed: the local
10/20-template study with LAL references passed exact-waveform backend checks,
but two injections in the 20-template bank failed the strict LAL-versus-ripple
peak chi-squared gate (differences about 0.0014 and 0.0016). These remain
reported failures in LAL-reference mode. They must not be presented as GPU
arithmetic errors or as successful LAL agreement. Identical-model mode checks
all SNR/veto curves with the existing strict backend tolerances.

Startup waveform validation accepts a time displacement of at most one sample
and a support-cutoff difference of at most one frequency bin. It records the
unaligned match, aligned match and measured offsets; generated waveforms are
not shifted or trimmed for filtering. Peak-time comparisons already allow one
sample, with phase/vetoes evaluated at the same physical sample for adjacent
peaks. These allowances do not waive SNR/veto amplitude errors or background
count differences.

Use a new output directory for every replay. For final throughput, repeat
without `--profile-stages`; profiling intentionally synchronizes stages.
Tuning/JIT are still included in wall time; inspect warm active-block latency
separately. If another GPU is allocated, `--coordinator-gpu` also offloads rank-0
conditioning. It requires a distinct coordinator GPU by default; sharing one
requires explicit `--allow-gpu-sharing` and can create resource contention.

All existing numerical tolerances, trigger identity checks and exact integer
background-count checks remain in place. A pass on a small fixture does not
certify the full native bank, real frame data, or GPU execution. Local tests
used JAX CPU because the development machine has no GPU; P100 correctness and
speedup must be measured with the required-GPU tests and replay above.
