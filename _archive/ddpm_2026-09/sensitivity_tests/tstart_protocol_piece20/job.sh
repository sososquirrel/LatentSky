#!/bin/bash
#PBS -N ls_tstart_sens_p20
#PBS -A UCLB0066
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=64GB:ngpus=1:gpu_type=a100
#PBS -l walltime=00:45:00
#PBS -j oe
#PBS -o /glade/derecho/scratch/sabramian/logs/ls_tstart_sens_p20.log
set -euo pipefail
cd /glade/u/home/sabramian/LatentSky

export TMPDIR=/glade/derecho/scratch/sabramian/tmp
ENV_PY=/glade/work/sabramian/conda-envs/latentsky-env/bin/python
_NV=/glade/work/sabramian/conda-envs/latentsky-env/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=\
${_NV}/cublas/lib:${_NV}/cuda_cupti/lib:${_NV}/cuda_nvrtc/lib:\
${_NV}/cuda_runtime/lib:${_NV}/cudnn/lib:${_NV}/cufft/lib:\
${_NV}/curand/lib:${_NV}/cusolver/lib:${_NV}/cusparse/lib:\
${_NV}/cusparselt/lib:${_NV}/nccl/lib:${_NV}/nvjitlink/lib:\
${_NV}/nvtx/lib:${LD_LIBRARY_PATH:-}

$ENV_PY sensitivity_tests/tstart_protocol_piece20/run.py
