# Shared native search argv; sourced by run.sh and run_backend_comparison.py.
# Paths remain relative to the fixture directory. Keep waveform choices intact.
gps_start_time=${gps_start_time:-1272790000}
gps_end_time=${gps_end_time:-1272790512}
f_min=${f_min:-18}
live_example_args=(
--bank-file template_bank.hdf \
--sample-rate 2048 \
--enable-bank-start-frequency \
--low-frequency-cutoff ${f_min} \
--max-length 512 \
--approximant "SPAtmplt:mtotal<4" "SEOBNRv4_ROM:else" \
--chisq-bins "0.72*get_freq('fSEOBNRv4Peak',params.mass1,params.mass2,params.spin1z,params.spin2z)**0.7" \
--snr-abort-threshold 500 \
--snr-threshold 4.5 \
--newsnr-threshold 4.5 \
--max-triggers-in-batch 30 \
--store-loudest-index 50 \
--analysis-chunk 8 \
--autogating-threshold 50 \
--autogating-pad 0.5 \
--autogating-cluster 1 \
--autogating-width 0.25 \
--autogating-taper 0.25 \
--highpass-frequency 13 \
--highpass-bandwidth 5 \
--highpass-reduction 200 \
--psd-samples 30 \
--max-psd-abort-distance 600 \
--min-psd-abort-distance 20 \
--psd-abort-difference .15 \
--psd-recalculate-difference .01 \
--psd-inverse-length 3.5 \
--psd-segment-length 4 \
--trim-padding .5 \
--store-psd \
--increment-update-cache \
    H1:strain/H1 \
    L1:strain/L1 \
    V1:strain/V1 \
--frame-src \
    H1:strain/H1/* \
    L1:strain/L1/* \
    V1:strain/V1/* \
--frame-read-timeout 10 \
--channel-name \
    H1:SIMULATED_STRAIN \
    L1:SIMULATED_STRAIN \
    V1:SIMULATED_STRAIN \
--processing-scheme cpu:4 \
--fftw-measure-level 0 \
--fftw-threads-backend openmp \
--increment 8 \
--max-batch-size 16777216 \
--output-path output \
--day-hour-output-prefix \
--sngl-ranking newsnr_sgveto_psdvar_threshold \
--ranking-statistic \
  exp_fit \
--statistic-features \
  phasetd \
  sensitive_volume \
  normalize_fit_rate \
--statistic-keywords \
  alpha_below_thresh:6 \
--statistic-files \
  statHL.hdf \
  statHV.hdf \
  statLV.hdf \
  H1-fit_coeffs.hdf \
  L1-fit_coeffs.hdf \
  V1-fit_coeffs.hdf \
--sgchisq-snr-threshold 4 \
--sgchisq-locations "mtotal>40:20-30,20-45,20-60,20-75,20-90,20-105,20-120" \
--enable-background-estimation \
--background-ifar-limit 100 \
--timeslide-interval 0.1 \
--pvalue-combination-livetime 0.0005 \
--ifar-double-followup-threshold 0.0001 \
--ifar-upload-threshold 0.0001 \
--round-start-time 4 \
--start-time $gps_start_time \
--end-time $gps_end_time \
--src-class-mchirp-to-delta 0.01 \
--src-class-eff-to-lum-distance 0.74899 \
--src-class-lum-distance-to-delta -0.51557 -0.32195 \
--run-snr-optimization \
--snr-opt-extra-opts \
    "--snr-opt-method differential_evolution \
    --snr-opt-di-maxiter 50 \
    --snr-opt-di-popsize 100 \
    --snr-opt-include-candidate " \
--sngl-ifar-est-dist conservative \
--single-ranking-threshold 9 \
--single-duration-threshold 7 \
--single-reduced-chisq-threshold 2 \
--single-fit-file single_significance_fits.hdf \
--single-maximum-ifar 100 \
--psd-variation \
--verbose
)
