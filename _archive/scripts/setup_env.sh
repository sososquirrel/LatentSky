#!/bin/bash
# Create the latentsky-env conda environment on Casper/Derecho and register
# it as a Jupyter kernel so it appears in JupyterHub.
#
# Usage:  bash scripts/setup_env.sh
#
# Re-runs are safe: if the env exists, mamba update is used.
set -euo pipefail

ENV_NAME="latentsky-env"
ENV_PREFIX="/glade/work/${USER}/conda-envs/${ENV_NAME}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# NCAR conda module
module --force purge
module load ncarenv/24.12
module load conda/latest
eval "$(conda shell.bash hook)"

mkdir -p "$(dirname "$ENV_PREFIX")"

if [ -d "$ENV_PREFIX" ]; then
    echo "[setup_env] $ENV_PREFIX exists — updating from environment.yaml"
    mamba env update -p "$ENV_PREFIX" -f "$REPO_ROOT/environment.yaml"
else
    echo "[setup_env] creating $ENV_PREFIX"
    mamba env create -p "$ENV_PREFIX" -f "$REPO_ROOT/environment.yaml"
fi

conda activate "$ENV_PREFIX"

# Register as Jupyter kernel so it shows up in JupyterHub.
python -m ipykernel install --user \
    --name "$ENV_NAME" \
    --display-name "Python (latentsky-env)"

echo
echo "[setup_env] Done."
echo "  Activate:    conda activate $ENV_PREFIX"
echo "  Kernel name: $ENV_NAME  (visible in JupyterHub)"
