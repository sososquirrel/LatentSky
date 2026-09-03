#!/bin/bash
# Submit the five remaining flow-matching leads on DERECHO as a serial chain.
# Run from any Derecho login node:  bash experiments/flow_matching_6h/submit_derecho_chain.sh
set -euo pipefail
cd /glade/u/home/sabramian/LatentSky
prev=""
for L in 1 2 3 0.25 0.5; do
  if [ -z "$prev" ]; then
    jid=$(qsub experiments/flow_matching_6h/job_lead${L}h_derecho.sh)
  else
    jid=$(qsub -W depend=afterany:$prev experiments/flow_matching_6h/job_lead${L}h_derecho.sh)
  fi
  echo "lead ${L}h -> $jid"
  prev=$jid
done
echo "chain submitted; logs: /glade/derecho/scratch/sabramian/logs/ls_flow{L}h.log"
