#!/bin/bash
#PBS -N ls_<name>
#PBS -A UCLB0066
#PBS -q main
#PBS -l select=1:ncpus=8:mem=128GB:ngpus=1:gpu_type=a100
#PBS -l walltime=04:00:00
#PBS -j oe
# Short jobs (<1 h): qsub -q develop for fast GPU slots.
set -euo pipefail
cd $PBS_O_WORKDIR

export TMPDIR=/glade/derecho/scratch/$USER/tmp
mkdir -p "$TMPDIR"

ENV_PY=/glade/work/sabramian/conda-envs/latentsky-env/bin/python
_NV=/glade/work/sabramian/conda-envs/latentsky-env/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=\
${_NV}/cublas/lib:${_NV}/cuda_cupti/lib:${_NV}/cuda_nvrtc/lib:\
${_NV}/cuda_runtime/lib:${_NV}/cudnn/lib:${_NV}/cufft/lib:\
${_NV}/curand/lib:${_NV}/cusolver/lib:${_NV}/cusparse/lib:\
${_NV}/cusparselt/lib:${_NV}/nccl/lib:${_NV}/nvjitlink/lib:\
${_NV}/nvtx/lib:${LD_LIBRARY_PATH:-}

$ENV_PY run.py --config config.yaml
