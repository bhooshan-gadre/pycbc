# Serial matched filter versus the JAX SNR surface

Regenerate the figures with:

    /home/bug/work/pycbc_live/.venv/bin/python examples/live/plot_batch_vs_serial.py

The script draws 20 aligned-spin IMRPhenomD templates (seed 20261116, also written in comparison_summary.txt), keeps the first 10 as the smaller bank, and filters one 8 s valid block. The two datasets are aLIGOZeroDetHighPower colored Gaussian noise and that noise plus one bank injection. Sample rate 2048 Hz, f_lower 20 Hz.

Serial evaluation calls pycbc.filter.matched_filter_core once per template on the shared overwhitened frequency series. Batch evaluation is JaxLiveBatchMatchedFilter on CPU. Both correlate the complex64 series the batch filter stores, with the bank normalization 4 * df / sqrt(sigmasq). The block is already overwhitened, so the serial call does not divide by the PSD again. Timings are a second call, after FFTW planning and JAX compilation.
