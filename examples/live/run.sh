#!/bin/bash

# example/test of running PyCBC Live on simulated data
#
# Optional JAX filter (does not replace this SEOBNRv4 path). Install with
#   pip install 'jax' 'ripplegw'
# and see run_jax_parity.sh. Extra flags: --filter-backend {cpu,jax},
# --jax-platform {cpu,gpu}, --rank-backend cpu,jax,jax
# Rank 0 stays on CPU. --jax-device-id selects the JAX GPU.
# See run_backend_comparison.py to reuse these exact fixtures and settings.

set -e

LIVE_SOURCE="$(cd "$(dirname "$0")" && pwd)"
source "$LIVE_SOURCE/live_example_options.sh"

export OMP_NUM_THREADS=4
export HDF5_USE_FILE_LOCKING="FALSE"



# test if there is a template bank. If not, make one

if [[ ! -f template_bank.hdf ]]
then
    echo -e "\\n\\n>> [`date`] Making template bank"
    curl \
        --remote-name \
        --silent \
        --show-error \
        https://raw.githubusercontent.com/gwastro/pycbc-config/710dbfd3590bd93d7679d7822da59fcb6b6fac0f/O2/bank/H1L1-HYPERBANK_SEOBNRv4v2_VARFLOW_THORNE-1163174417-604800.xml.gz

    pycbc_coinc_bank2hdf \
        --bank-file H1L1-HYPERBANK_SEOBNRv4v2_VARFLOW_THORNE-1163174417-604800.xml.gz \
        --output-file template_bank_full.hdf

    rm -f H1L1-HYPERBANK_SEOBNRv4v2_VARFLOW_THORNE-1163174417-604800.xml.gz

    pycbc_hdf5_splitbank \
        --bank-file template_bank_full.hdf \
        --output-prefix template_bank_ \
        --random-sort \
        --random-seed 831486 \
        --templates-per-bank 50

    mv template_bank_0.hdf template_bank.hdf
    rm -f template_bank_*.hdf
else
    echo -e "\\n\\n>> [`date`] Pre-existing template bank found"
fi

# test if there is a single fits file. If not, make a representative one
if [[ ! -f single_significance_fits.hdf ]]
then
    echo -e "\\n\\n>> [`date`] Making single significance fits file"
    python "$LIVE_SOURCE/make_singles_significance_fits.py"
else
    echo -e "\\n\\n>> [`date`] Pre-existing single significance fits file found"
fi

# test if there are fit_coeffs files for each detector.
# If not, make some representative ones
if [[ `ls -1 H1-fit_coeffs.hdf L1-fit_coeffs.hdf V1-fit_coeffs.hdf 2> /dev/null | wc -l` -lt 3 ]]
then
    echo -e "\\n\\n>> [`date`] Making fit coefficient files"
    python "$LIVE_SOURCE/make_fit_coeffs.py"
else
    echo -e "\\n\\n>> [`date`] All needed fit coeffs files found"
fi

# test if there is a injection file.
# If not, make one and delete any existing strain

if [[ -f injections.hdf ]]
then
    echo -e "\\n\\n>> [`date`] Pre-existing injections found"
else
    echo -e "\\n\\n>> [`date`] Generating injections"

    if [[ "${PYCBC_LIVE_PREPARE_ONLY:-0}" == "1" && -d ./strain ]]; then
        echo "Existing strain has no injections.hdf; refusing to replace it during preparation" >&2
        exit 1
    fi
    rm -rf ./strain

    python "$LIVE_SOURCE/generate_injections.py"
fi


# test if strain files exist. If they don't, make them

if [[ ! -d ./strain ]]
then
    echo -e "\\n\\n>> [`date`] Generating simulated strain"

    function simulate_strain { # detector PSD_model random_seed
        mkdir -p strain/$1

        out_path="strain/$1/$1-SIMULATED_STRAIN-{start}-{duration}.gwf"

        pycbc_condition_strain \
            --fake-strain $2 \
            --fake-strain-seed $3 \
            --output-strain-file $out_path \
            --gps-start-time $gps_start_time \
            --gps-end-time $4 \
            --sample-rate 16384 \
            --low-frequency-cutoff 10 \
            --channel-name $1:SIMULATED_STRAIN \
            --frame-duration 32 \
            --injection-file injections.hdf
    }
    # L1 ends 32s later, so that we can inject in single-detector time
    simulate_strain H1 aLIGOMidLowSensitivityP1200087 1234 $((gps_end_time - 32))
    simulate_strain L1 aLIGOMidLowSensitivityP1200087 2345 $gps_end_time
    simulate_strain V1 AdVEarlyLowSensitivityP1200087 3456 $((gps_end_time - 32))

else
    echo -e "\\n\\n>> [`date`] Pre-existing strain data found"
fi


# make phase-time-amplitude histogram files, if needed

if [[ ! -f statHL.hdf ]]
then
    echo -e "\\n\\n>> [`date`] Making phase-time-amplitude files"

    bash "$LIVE_SOURCE/../search/stats.sh"
else
    echo -e "\\n\\n>> [`date`] Pre-existing phase-time-amplitude files found"
fi


# Prepare fixtures without running or deleting any existing search outputs.
if [[ "${PYCBC_LIVE_PREPARE_ONLY:-0}" == "1" ]]; then
    exit 0
fi

# delete old outputs if they exist
rm -rf ./output


echo -e "\\n\\n>> [`date`] Running PyCBC Live"


mpirun \
-host localhost,localhost \
-n 2 \
--bind-to none \
 -x PYTHONPATH -x LD_LIBRARY_PATH -x OMP_NUM_THREADS -x VIRTUAL_ENV -x PATH -x HDF5_USE_FILE_LOCKING \
\
python -m mpi4py `which pycbc_live` \
"${live_example_args[@]}"

# If you would like to use the pso optimizer, change --optimizer to pso
#  and include these arguments while removing other optimizer args.
#  You will need to install the pyswarms package into your environment.
# --snr-opt-extra-opts \
#   "--snr-opt-method pso \
#   --snr-opt-pso-iters 5 \
#   --snr-opt-pso-particles 250 \
#   --snr-opt-pso-c1 0.5 \
#   --snr-opt-pso-c2 2.0 \
#   --snr-opt-pso-w 0.01 \
#   --snr-opt-include-candidate " \

# note that, at this point, some SNR optimization processes may still be
# running, so the checks below may ignore their results

# cat the logs of pycbc_optimize_snr so we can check them
for opt_snr_log in `find output -type f -name optimize_snr.log | sort`
do
    echo -e "\\n\\n>> [`date`] Showing log of SNR optimizer, ${opt_snr_log}"
    cat ${opt_snr_log}
done

echo -e "\\n\\n>> [`date`] Checking results"
python "$LIVE_SOURCE/check_results.py" \
    --gps-start ${gps_start_time} \
    --gps-end ${gps_end_time} \
    --f-min ${f_min} \
    --bank template_bank.hdf \
    --injections injections.hdf \
    --detectors H1 L1 V1

echo -e "\\n\\n>> [`date`] Running Bayestar"
for XMLFIL in `find output -type f -name \*.xml\* | sort`
do
    pushd `dirname ${XMLFIL}`
    bayestar-localize-coincs --f-low ${f_min} `basename ${XMLFIL}` `basename ${XMLFIL}`
    test -f 0.fits
    popd
done
