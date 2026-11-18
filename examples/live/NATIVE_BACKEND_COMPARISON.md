# Run the original live example with native, JAX CPU and JAX GPU filtering

This comparison uses the **original `examples/live/run.sh` fixtures and search
settings**: the 50-template O2 bank subset, simulated H1/L1/V1 frames and native
injections, `SPAtmplt:mtotal<4` / `SEOBNRv4_ROM:else`, dynamic chi-square bins,
SG veto, PSD variation, background estimation, ranking and coincidence output.
It does not substitute the small IMRPhenomD benchmark bank or generate new noise
for each backend. `run.sh` and the new runner source `live_example_options.sh`,
so the native search options have one definition.

Activate your existing micromamba environment with this checkout installed,
JAX, mpi4py, FFTW and the dependencies that already run the native example.
Keep the native example's LAL waveform data configuration (`LAL_DATA_PATH`,
including the SEOBNRv4 ROM data). For P100 use the compatible CUDA 12 JAX stack
in [GPU_TESTS.md](../search/jax/GPU_TESTS.md).

From the checkout root, point at your **existing native fixture directory**:

```bash
export SCRATCH=/path/to/your/scratch
export OMP_NUM_THREADS=4 HDF5_USE_FILE_LOCKING=FALSE
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$SCRATCH/pycbc-matplotlib"

python examples/live/run_backend_comparison.py \
  --example-dir "$SCRATCH/native-live-example" \
  --output "$SCRATCH/live-backend-comparison" \
  --repeats 3
```

`--example-dir` must contain `template_bank.hdf`, `injections.hdf`, `strain/`,
the detector fit files, `single_significance_fits.hdf` and `stat*.hdf` from the
native example. To use fixtures generated previously in the checkout, pass
`--example-dir "$PWD/examples/live"` instead. Existing `output/` is preserved.
The bank, frame and statistic files actually read by the search are hashed
before and after the comparison; changed, missing or additional input files
fail the integrity check.

If the fixtures have not been generated yet, add `--prepare`:

```bash
python examples/live/run_backend_comparison.py \
  --example-dir "$SCRATCH/native-live-example" \
  --output "$SCRATCH/live-backend-comparison" \
  --prepare --repeats 3
```

Preparation invokes the original example once, in preparation-only mode, before
any timed search. It can download the large bank and create frames/statistic
files. It exits before deleting or running the native `output/` directory.
If existing strain has no corresponding `injections.hdf`, preparation refuses
to replace it. Supply the matching complete native fixtures in that case.
An already populated comparison output directory is refused.

The default modes are `native jax_cpu jax_gpu`; the same number of MPI ranks and
search settings are used for all. Rank 0 is the coordinator; default total ranks
is 2, as in the native example. `--mpi-ranks 3` uses two filtering workers and
therefore requires two allocated GPUs for the GPU mode. Respect your scheduler's
`CUDA_VISIBLE_DEVICES`; do not expose unallocated GPUs. Each GPU filtering rank
is assigned a distinct visible GPU automatically. GPU failures stay failures.
For a CPU-only check explicitly pass `--modes native jax_cpu`.

For MPICH/Hydra use `--launcher mpiexec` (or `--launcher mpiexec.mpich` when
both MPI implementations are installed). The default comparison command uses
only the common `-n` launcher option. For **Open MPI only**, binding can be
disabled with `--launcher-arg=--bind-to --launcher-arg=none`; do not pass these
options or Open MPI's `-x` options to Hydra. The runner uses the active
Python and this checkout's `bin/pycbc_live`; exported environment variables are
inherited by local MPI workers. Multi-node jobs require the same checkout,
environment and fixture paths on each node and scheduler-appropriate launch
arguments. Optional `--autotune` tunes JAX batch layouts; its cost remains in
process wall time. The default keeps the example's fixed batch-size limit.
`--plan-only` validates inputs and writes exact commands without running MPI.
`--args-json` accepts a custom fixed native argv list for other replay fixtures.

## Keep waveform and followup choices explicit

JAX runs pass `--jax-waveform-source bank`. Both CPU and GPU filter **the exact
native-generated frequency series**, including SEOBNRv4_ROM, with native
normalization. There is no replacement of SEOBNRv4_ROM by IMRPhenomD. Waveform
creation and conditioning remain on CPU for this test; GPU filtering/veto
arithmetic is what is compared. Use the separate IMRPhenomD component benchmark
to measure ripplegw waveform generation.

Asynchronous SNR optimization is disabled uniformly in the three runs. Bayestar,
which the original shell script runs after `pycbc_live`, is not launched here.
Their separate followup processes could otherwise overlap the next backend and
contaminate latency measurements. Background estimation, ranking, coincidence
XML production and the example's single-detector settings remain enabled.
No GraceDB uploads are enabled by these options.

## Compare the logs and final outputs

Run this independently after the searches (or rerun after inspecting results):

```bash
python examples/live/compare_backend_runs.py \
  --run-dir "$SCRATCH/live-backend-comparison" \
  --warmup-blocks 1
```

The runner invokes the comparison automatically after the searches. You can
rerun it independently with the command above. **PASS means the numerical and
latency-coverage checks passed**. A failed GPU run,
missing blocks/triggers/candidates, changed input files or a parity failure
returns nonzero. All raw outputs and logs remain available.

The comparison creates `comparison/` with:

* `summary.json`: job failures, matched/missing/extra rows, numerical error
  maxima, candidate XML checks, per-run latency statistics and wall speedups.
* `*_triggers.csv` and `*_results.png`: SNR, power chi-square, SG chi-square,
  phase and time comparisons/differences, keyed by analysis block, detector and
  template ID. All per-trigger scalar fields are checked; integer template
  hashes/IDs are compared exactly without float conversion. Phase errors wrap
  modulo 2π. Missing rows are never silently dropped.
* `latency.csv`, `latency.png` and `latency_blocks.png`: process wall time,
  warm block median/p95/max, initial active block latency and the block sequence.

The script also compares HDF block coverage, PSD/gate/coincidence datasets and
candidate XML scientific tables. XML process/event identifiers are excluded;
scientific values and candidate counts are checked. Candidate XML filenames
must match; a change in the candidate set is reported rather than paired away.
The first successful native run is the reference, including for subsequent
native repeats, which can reveal repeat-to-repeat differences in the pipeline.
The native example has injection-recovery checks in `check_results.py`; this
backend comparison checks equivalence and does not impose a new injection-SNR
requirement on its original, unchanged injection set.

Complete process wall time includes MPI startup, native waveform creation,
conditioning, JIT, optional tuning, filtering/vetoes and output. Preparation and
input hashing are outside the timed jobs. Backend order rotates across repeats.
The runner uses a consistent `PYTHONHASHSEED` (default 0) for all modes and
records package versions, MPI command lines and GPU visibility in `runs.json`.

The new optional `pycbc_live --latency-log FILE` writes rank-0 JSONL records with
`time.perf_counter()` timing for each analysis block: conditioning, MPI wait,
filtering/veto results, background/coincidence and writing outputs are included.
The comparator checks JSONL block counts against coordinator log messages and
compares block/detector coverage between backends. Inactive blocks are reported
but excluded from active processing statistics. The initial active block is
reported separately; `--warmup-blocks` excludes initial active blocks only from
warm statistics, never from output parity or total runtime. Later PSD/JIT work
is still included. These are processing latencies, not acquisition/network or
astrophysical-alert latencies. Replayed historical GPS **lag is not latency**.
For older logs without JSONL the script accepts coordinator `Took` messages,
labels their lower 0.01-second precision and does not mix in worker timings.

GPU execution requires validation on your hardware. The new orchestration was
smoke-tested locally through real MPI on a small shared fixture with native and
JAX CPU modes; this does not claim a full O2-bank/native-example or GPU run.

`--plan-only` validates the fixtures and writes the exact commands without
launching jobs. Choose a different output directory for the subsequent real run.
For a CPU-only smoke check, use `--modes native jax_cpu`. GPU mode is required
by default and never silently skipped. The default two MPI ranks reproduce the
native example's one coordinator plus one filtering worker. Use `--mpi-ranks 3`
for two filtering workers/two GPUs, with the same layout for every backend;
that changes the native example's original MPI partition. Multi-node jobs need
shared fixture/output paths and identical environments on the participating nodes.


### Native run.sh with MPICH/Hydra

An error such as `HYDU_parse_array: unrecognized argument x` means that Hydra
was given Open MPI's `-x` option. The updated `run.sh` uses `mpiexec -n 2`,
inherits exported environment variables, and adds `--bind-to none` only when
the launcher identifies itself as Open MPI/OpenRTE. It no longer forces local
hosts. If needed, choose the launcher explicitly:

```bash
cd examples/live
PYCBC_MPIEXEC=mpiexec bash run.sh
# On machines with both MPI installations:
PYCBC_MPIEXEC=mpiexec.mpich bash run.sh
```

Use the launcher belonging to the MPI library loaded by your environment's
`mpi4py`. A two-rank check should print ranks 0 and 1, each with size 2:

```bash
mpiexec -n 2 python -c 'from mpi4py import MPI; print(MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size, MPI.get_vendor())'
```
