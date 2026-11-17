# CPU execution evidence (2026-11-17)

GPU hardware was unavailable; no GPU performance or correctness is claimed.
The component run used IMRPhenomD banks of 10 and 20 templates, noise and only
template 0 injected separately for this quick timing study, 11 warm repeats,
17 veto samples and 5 tuning repeats. All four API cases and all exact-waveform
controls passed. The runner defaults to injecting every template for acceptance.

The real MPI run used 10 templates, two filtering workers plus one coordinator,
two fresh-process repeats, identical seeded H1/L1 frame data and a separate
SNR-40 injection time for every template. All injections recovered above 10.
All 260 trigger keys match and SNR error is below 0.00074, but strict phase/veto
parity FAILS: same-time LAL/ripple power-chi-square differences and one one-sample
peak shift remain. See preserved per-row CSVs and plots. No gates were weakened.

Full process wall-time medians (including compilation, tuning and output):
native 4.188 s; JAX CPU 8.244 s. These short runs do not show an end-to-end
CPU speedup. Warm filtering and generation are recorded separately in component
CSVs and JSON; their speedups must not be substituted for process speedup.

`components_summary.json.gz` preserves candidate measurements and all parity
metrics. `live_summary.json.gz` preserves exact executed commands, timings and
strict failures. Local absolute paths in those commands describe this run and
must be replaced when reproducing. Source version is the parent commit plus
the implementation saved alongside this evidence. Python 3.12 / JAX 0.10.2 /
ripplegw 0.5.1, native FFTW, one CPU thread per worker.

Commands:

```bash
python examples/search/jax/validate_cpu_gpu.py --gpu off --approximants IMRPhenomD \
  --bank-sizes 10 20 --injection-templates first --repeats 11 --veto-samples 17 \
  --autotune --tuning-repeats 5 --benchmark-components --output /tmp/new-components
python examples/live/benchmark_imrphenomd.py --gpu off --templates 10 --workers 2 \
  --repeats 2 --autotune --output /tmp/new-live
```

See [running GPU tests](../../../../live/IMRPHENOMD_BENCHMARK.md).
