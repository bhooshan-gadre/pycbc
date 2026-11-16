#!/bin/bash
# Parity gate for the JAX PyCBC Live matched filter.
#
# Optional packages, not required by a CPU-only PyCBC install:
#   pip install 'jax' 'ripplegw'
# or: pip install 'pycbc[jax]'
#
# CPU ranks use LiveBatchMatchedFilter. JAX ranks use JaxLiveBatchMatchedFilter.
# A mixed job cycles --rank-backend across filtering ranks, for example:
#   mpirun -np 3 pycbc_live ... --rank-backend cpu,jax --jax-platform cpu
# GPU ranks need a visible device and --jax-platform gpu. This script runs the
# CPU comparison, then the GPU comparison when JAX can see a GPU.
#
# The comparison writes single-detector HDF files and runs check_results.py
# on them. Full pycbc_live still accepts --filter-backend on the command in
# run.sh; point --bank-file at the TaylorF2 or IMRPhenomD bank from this run
# rather than the SEOBNRv4 example bank.

set -euo pipefail
cd "$(dirname "$0")"
SOURCE="$(cd ../.. && pwd)"
export PYTHONPATH="${SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"

python compare_jax_live.py --jax-platform cpu "$@"

if python -c "import jax; jax.devices('gpu')" >/dev/null 2>&1; then
    python compare_jax_live.py --jax-platform gpu "$@"
else
    echo "No JAX GPU device; GPU parity run skipped"
fi
