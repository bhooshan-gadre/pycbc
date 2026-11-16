# Native and JAX CPU/GPU validation on a Tesla P100

Run `validate_cpu_gpu.py` from this branch to compare **native serial, native
batch, JAX CPU serial, JAX CPU batch, JAX GPU serial, and JAX GPU batch**.
No MPI launcher, frame files, network downloads, or production search service
is needed. This checks filtering, template generation, peak selection and
vetoes; it does not launch the entire live/offline search executables.

## Install on the GPU machine

Clone the feature branch:

```bash
git clone --branch jax-live-snr-surface --single-branch \
  https://github.com/bhooshan-gadre/pycbc.git pycbc-jax
cd pycbc-jax
```

From an existing clone:

```bash
git fetch origin jax-live-snr-surface
git switch jax-live-snr-surface
git pull --ff-only origin jax-live-snr-surface
```

Use Python 3.11 or 3.12. The CPU development run used Python 3.12, JAX/jaxlib
0.10.2 and ripplegw 0.5.1. A fresh environment avoids mixing CUDA plugin
versions. Install normal PyCBC source-build prerequisites first (on Debian or
Ubuntu: a C/C++ compiler, Python development headers, pkg-config, FFTW3 and
HDF5 development packages). For example:

```bash
python3.12 -m venv .venv-p100
source .venv-p100/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r examples/search/jax/requirements-p100.txt
python -m pip install -e .
python -m pip check
nvidia-smi
python -c "import jax; print(jax.__version__); print(jax.devices('gpu'))"
```

**Select CUDA 12 explicitly.** P100 is compute capability 6.0. JAX's CUDA 12
distribution supports SM >= 5.2; its CUDA 13 distribution requires SM >= 7.5,
which excludes P100. JAX documents Linux driver >= 525 for CUDA 12. The
`jax[cuda12]` extra supplies CUDA user-space libraries; do not substitute
`jax[cuda13]`. An old cluster-provided CUDA/cuDNN in `LD_LIBRARY_PATH` can
override the pip libraries; follow the JAX installation instructions when
using cluster modules. The CUDA wheel has not been executed on a GPU in the
CPU development environment; the explicit GPU preflight below is required.

Sources: [JAX installation](https://docs.jax.dev/en/latest/installation.html),
[NVIDIA legacy GPU capabilities](https://developer.nvidia.com/cuda/gpus/legacy).

## Run

Request a GPU allocation through your site's scheduler first. Then run:

```bash
source .venv-p100/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=.
# Do not set JAX_PLATFORMS=cpu for a GPU run.
unset JAX_PLATFORMS

# Quick GPU smoke test: both models, noise and injection, two templates.
python examples/search/jax/validate_cpu_gpu.py \
  --gpu required --bank-sizes 2 --repeats 3 --veto-samples 17 \
  --output examples/search/jax/validation_results_smoke

# Full requested matrix: 10 and 20 random aligned-spin templates, both models,
# eight valid seconds of noise and noise+injection; 11 warm repetitions.
python examples/search/jax/validate_cpu_gpu.py \
  --gpu required --bank-sizes 10 20 --repeats 11 --veto-samples 65 \
  --output examples/search/jax/validation_results_p100
```

Run directly, not with `mpirun`: the script compares all modes in one process.
The seed is fixed at 20261116; the 10-template bank is a prefix of the
20-template bank. Data sample rate is 2048 Hz and the low-frequency cutoff
is 20 Hz. The FFT buffer includes duration-dependent padding, with **exactly
eight valid seconds** plotted and compared. The injection uses the native
bank template at time 5 seconds with optimal SNR 20. Colored Gaussian noise
and the injection are identical for every mode. No real detector data is
included in this portable GPU test.

The default is `--gpu required`. If no GPU can execute the preflight, this
returns an error rather than reporting a CPU run as a GPU pass. CPU FFT and
FP64 JIT arithmetic are also preflighted; every GPU filter workspace and
veto result is checked for placement on the selected GPU. The script does
not set a global platform to hide CPU devices. Use `--gpu-index 1` for the
second **visible** GPU; scheduler `CUDA_VISIBLE_DEVICES` remapping applies.

For a CPU-only machine:

```bash
python -m pip install 'jax==0.10.2' 'ripplegw==0.5.1' matplotlib pytest
python -m pip install -e .
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  python examples/search/jax/validate_cpu_gpu.py --gpu off \
    --output examples/search/jax/validation_results_cpu
```

`--gpu optional` attempts both and explicitly records a skipped GPU. Neither
optional-without-GPU nor off can produce a CPU/GPU pass status.

Optional Slurm example (adapt account, partition and environment paths to
your site; submit with `sbatch` from the repository root):

```bash
#!/bin/bash
#SBATCH --job-name=pycbc-jax-check
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=pycbc-jax-%j.log
set -euo pipefail
source .venv-p100/bin/activate
unset JAX_PLATFORMS
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python examples/search/jax/validate_cpu_gpu.py --gpu required \
  --output examples/search/jax/validation_results_p100
```

## Exactly what is compared

* **Regular TaylorF2:** LAL TaylorF2 templates from the PyCBC live bank are
  filtered by every implementation. This intentionally bypasses the existing
  JAX SPA template replacement. It tests ordinary TF2 filtering on the GPU;
  it does not introduce a JAX-native TaylorF2 waveform generator.
* **IMRPhenomD:** native references use LAL bank waveforms. JAX modes must
  actually use ripplegw-generated waveforms on their selected device; a
  startup waveform fallback is an error in this runner. Generation uses
  FP64 parameters/phasing and stores complex64 strain. It is startup work,
  reported separately from warm filtering. Unused DC and Nyquist endpoints
  are zeroed consistently with the bank.
* **Serial:** native calls `matched_filter_core` once per template. JAX uses
  one-template chunks in the same production engine used by batch mode.
  **Batch:** duration groups share strain and transform batches together.
* **Vetoes:** production eight-bin power chi-squared and a Q=10,
  zero-frequency-offset sine-Gaussian tile, enabled for all bank locations.
  SG activation follows reweighted SNR >= 4. Returned power chi-squared is
  reduced by `2*8-2`; SG values retain their native normalization. An
  inactive SG veto returns 1, which is not evidence of GPU tile execution;
  active candidates and time samples exercise the tile kernel.
* The full complex SNR surface is retained for all templates, including
  real/imaginary quadratures. `abs(rho)` is the phase-maximized SNR; its
  maximum in the valid block is the final time/phase-optimized SNR.
  `angle(rho_peak)` supplies the optimum coalescence phase.
* Veto curves use 65 uniformly spaced samples plus extra points around the
  representative template's peak. They are **sampled veto time series**,
  not a dense veto computation at every 2048 Hz sample. Use
  `--veto-samples 16384` for dense sampling (substantially more expensive).

## Accuracy gates and model differences

Three comparisons prevent waveform-model errors from being confused with
GPU kernel errors:

1. Every JAX mode has an independent native CPU control filtering its exact
   chosen waveform, with the same bank normalization and veto bins. Strict
   gates apply to the full complex surface, peaks and sampled veto curves.
2. Every JAX CPU/GPU serial/batch result is compared directly to JAX CPU
   serial, including sampled veto curves. This tests device and batching
   parity, including device-specific ripple generation.
3. All modes are also compared with the original native LAL bank. Ripple
   and LAL PhenomD are distinct numerical generators, so this comparison
   includes waveform-model differences. All cross-model residuals are
   plotted and saved; no reference waveform is replaced to make it pass.

The strict complex-SNR gate is `1e-3 + 1e-4 * max(abs(reference), 1)`.
The **cross-model PhenomD surface only** uses absolute tolerance `5e-3`
instead of `1e-3`, reflecting observed waveform differences in low-SNR
sidelobes; exact-waveform and CPU/GPU gates remain strict. Peak SNR uses
relative tolerance 1e-4 with absolute 1e-3 only below SNR 5. Both peak vetoes
and kernel veto curves use `1e-3 + 1e-4 * max(abs(reference), 1)`. Peak phase
must agree within 1e-3 rad and peak time within one sample; degrees of freedom
must agree exactly. In-band nonfinite ripple strain is an error.

Cross-model **off-peak PhenomD veto curves** have reported errors and strict
per-curve diagnostic flags but do not determine acceptance: tiny waveform
differences can amplify in these nonlinear quantities away from a peak.
The corresponding exact-waveform CPU controls and CPU/GPU curve comparisons
are mandatory acceptance gates. Cross-model peak vetoes remain mandatory.
This distinction is explicit in `cross_model_veto_curves_gated` in JSON/CSV.

The SG kernel follows the native CPU single-precision phase increment and
100-frequency-bin phase resets. This avoids the late-time errors caused by
forming one large single-precision phase angle. Double precision is retained
where needed in waveform generation and power-veto accumulation; P100 has
hardware FP64 support.

## Outputs, timing and exit status

The script prints per-mode median milliseconds, maximum complex-SNR error,
peak veto errors, native-batch speed ratio and peak-time displacement.

At the output root:

* `summary.json`: arguments, versions, device names, `nvidia-smi`, git commit,
  per-case acceptance, startup waveform reports and all failures.
* `metrics.csv`: model, bank size, data case, mode and numerical errors.
* `timings.csv`: setup, first call, warm median/minimum/p95 seconds. Setup
  includes waveform generation. First call includes compilation/cache setup.
  Warm calls return host triggers/vetoes and therefore synchronize execution.
  Native FFTW also benefits from warm plans. Diagnostic surface copies,
  exact-waveform CPU controls and plot generation are outside timed calls.

For each `IMRPhenomD/n10/injection/` (and other model/count/data cases):

* `case.json`: full case report, including `implementations.*.filter_only`
  and `jax_cpu_serial_consistency` with their mandatory acceptance flags.
* `timeseries.png`: full-rate SNR magnitude and quadratures, sampled power/SG
  vetoes, and differences from native serial.
* `surface.png`: time/template SNR surfaces and batch-minus-native residuals.
* `peaks.png`: final maximized SNRs and SNR/phase/time/veto peak differences.
* `filter_residuals.png`: JAX minus independent native CPU filtering of the
  **identical waveform**, isolating filtering and veto kernel errors.
* `arrays.npz`: full complex SNR surfaces, exact-waveform control surfaces,
  peak results, veto samples, template IDs and time axes. `--no-arrays` skips
  only this file. Generated output directories are ignored by git.

Exit **0** means all required numeric gates passed; inspect status to see
whether GPU was tested. `PASS_CPU_GPU` requires real GPU execution;
`PASS_CPU_ONLY_GPU_NOT_TESTED` does not validate GPU correctness. Exit **1**
means numerical failure (plots/reports are still saved), and exit **2** means
an environment/runtime error (including missing required GPU/ripplegw).
Use a fresh output directory for each run to avoid stale artifacts after an
early failure. The CPU machine cannot establish P100 speed or correctness.

For regression tests, including failure detection and CPU-only artifact checks:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  python -m pytest -q test/test_jax_live_filter.py \
    test/test_jax_search_workflows.py test/test_jax_cpu_gpu_validation.py
```

Ten or twenty short templates can be too little work to amortize GPU launch
and transfer costs. Start with correctness, then increase `--bank-sizes`
(e.g. `128 512`) and inspect actual `chunks`, setup time and warm time. This
runner measures the production host-trigger workflow; it does not claim a
GPU speedup based solely on an asynchronous FFT launch.
