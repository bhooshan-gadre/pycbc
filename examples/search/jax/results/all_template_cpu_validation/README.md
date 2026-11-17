# Per-template injection coverage: CPU validation

**All injected signals passed the recovered-SNR >=10 requirement. The overall
run is FAIL because two cross-model PhenomD power-chi-squared comparisons
exceed the existing strict tolerance. This is not a GPU test.**

The final run used FFTW for the native reference and independent JAX CPU
serial/batch workflows, with both IMRPhenomD and regular TaylorF2:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  python examples/search/jax/validate_cpu_gpu.py --gpu off \
    --bank-sizes 10 20 --injection-templates all --injection-snr 20 \
    --repeats 1 --veto-samples 9 --no-arrays \
    --output /tmp/pycbc-every-template-final
```

There are 64 eight-second cases: 60 separate injections (one per bank
template for each model and bank size) and four noise cases. Every case
uses the entire corresponding bank, with a common noise realization and
the same injected data in all four CPU modes. No signals are injected
simultaneously. The injected template ID is explicit in every report.

* All **240 recovery checks** passed, including both recovered peak SNR and
  SNR at the known injection sample. Minimum peak SNR was **17.81902695**;
  maximum was **22.61836624**. Minimum known-time SNR was **17.81902885**.
* All **128 independent exact-waveform CPU/JAX kernel controls** passed.
  Maximum full-surface complex-SNR residual was **4.26496120e-6**.
* All direct JAX CPU serial/batch consistency checks passed. Their surfaces
  were numerically identical at stored precision, using independent arrays.
  This legitimately gives zero residual. The native self-comparison is
  explicitly labelled N/A in the log; it is not GPU evidence.
* The two failing cases are `IMRPhenomD/n10/injection_t0008` and
  `IMRPhenomD/n20/injection_t0008`. JAX-versus-original-LAL reduced-power
  errors reach **1.11332110e-3** and **1.28032480e-3**, respectively. Both
  serial and batch flag these failures. Their exact-waveform kernel controls
  pass. The waveform/model tolerance has not been relaxed to hide this.

An adjacent, nearly tied peak in the TaylorF2 bank was checked in a separate
targeted recheck (`adjacent_peak_recheck.json.gz`). For
`n20/injection_t0015`, template 10 selects adjacent samples in native serial
and batch filtering. Original peak phase differs by about 0.15955 rad;
same-time comparison reduces the JAX/native phase residual to 1.20e-7 rad
and the reduced-power residual to 1.79e-4. Original peak differences remain
reported; the allowed timing displacement is still at most one sample.

`acceptance.json` contains coverage IDs, aggregate counts and failures.
`metrics.csv` and `timings.csv` retain all 256 mode/case rows, including
injection SNR checks and raw/same-time comparison metrics. `summary.json.gz`
is the complete original JSON report, compressed to avoid storing repeated
startup waveform reports as a large text diff. Read it with:

```bash
python - <<'PY'
import gzip, json
with gzip.open('examples/search/jax/results/all_template_cpu_validation/summary.json.gz', 'rt') as f:
    report = json.load(f)
print(report['status'], report['failures'])
PY
```

All plots were generated under `/tmp/pycbc-every-template-final`; regenerate
them with the command above. Full NPZ arrays were disabled for this sweep.
One timing repeat is for correctness coverage, not a performance claim.
The measurement ran with this feature's working-tree changes before the
commit containing these artifacts; its git metadata records the parent
commit. The original full-run exit code is 1, not a claimed acceptance pass.

The regression suite also covers the SNR-10 amplitude boost, sub-10 rejection,
all-template output coverage, independent exact-zero surfaces, adjacent-peak
comparisons, FFTW/NumPy references and CPU-only/GPU-required behavior.
All **38 regression tests passed**.
