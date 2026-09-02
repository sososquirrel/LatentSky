#!/bin/bash
#PBS -N ls_<name>
#PBS -A <project code>
#PBS -q main
#PBS -l select=1:ncpus=64:ngpus=1
#PBS -l walltime=04:00:00
#PBS -j oe
cd $PBS_O_WORKDIR
source ../../.venv/bin/activate
python run.py --config config.yaml
