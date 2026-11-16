# CPU acceptance run for the P100 test runner

This is a **CPU-only** run, not a P100 benchmark or GPU validation.

Command from the repository root (Python 3.12 virtual environment):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  python examples/search/jax/validate_cpu_gpu.py --gpu off --repeats 11 \
    --output /tmp/pycbc-p100-final
```

All eight model/bank/data cases passed: IMRPhenomD and regular TaylorF2,
10 and 20 templates, Gaussian noise and noise plus injection. Each case
compared native serial, native batch, JAX CPU serial and JAX CPU batch over
eight valid seconds. No failures; status `PASS_CPU_ONLY_GPU_NOT_TESTED`.

The independent native CPU controls filtering identical chosen waveforms
gave maximum complex-SNR error 4.08e-6, zero peak-time displacement, peak
reduced-power error <=6.52e-4 and peak SG error <=4.73e-4. Sampled power/SG
curve errors were <=7.13e-4 / 1.48e-5. These are maxima across all cases and
both JAX modes. Direct JAX serial/batch comparisons also passed.

The original LAL versus ripple PhenomD comparison includes waveform-model
differences: its maximum complex-SNR difference was 2.53e-3. See
[../../GPU_TESTS.md](../../GPU_TESTS.md) for the separate kernel and waveform
acceptance rules; reported cross-model off-peak veto flags can be false
without a kernel failure.

JSON/CSV files cover all cases. The plots retained here show the 20-template
injection cases for each model. Full-rate NPZ arrays, generated banks and
the remaining plots were generated in `/tmp/pycbc-p100-final` and are not
committed; regenerate with the command above. Report `git_commit` records
the parent feature commit: this measurement ran with the changes in the
following validation/optimization commit present in the working tree.
The regression suite also passed: **24 tests** in the three JAX test modules.

Timing is machine-specific; these small CPU banks do not establish a GPU
speedup. Use `--gpu required` on the P100 to produce the missing GPU evidence.
