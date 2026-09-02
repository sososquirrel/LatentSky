#!/bin/bash
# Submit every config in configs/ as its own PBS job.
# Job name and CONFIG env var are derived from the YAML filename.
set -e
cd "$(dirname "$0")/.."
for cfg in configs/*.yaml; do
    name=$(basename "$cfg" .yaml)
    echo "Submitting $name -> $cfg"
    qsub -v CONFIG="$cfg" -N "ls_${name}" submit/submit.pbs
done
