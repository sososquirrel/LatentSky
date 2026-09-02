#!/usr/bin/env bash
# Run from the root of the latentsky repo, on a fresh branch.
# Creates the new tree and moves everything else into _archive/.
set -euo pipefail

KEEP=(.git .gitignore CLAUDE.md .claude docs src experiments sensitivity_tests exploration wb2_benchmark results journal _archive data pyproject.toml README.md Makefile)

mkdir -p src/latentsky/{models,training,metrics,evaluation,comparison} \
         experiments sensitivity_tests exploration wb2_benchmark results docs journal _archive .claude/skills
touch src/latentsky/__init__.py src/latentsky/{models,training,metrics,evaluation,comparison}/__init__.py

for f in * .*; do
  [[ "$f" == "." || "$f" == ".." ]] && continue
  keep=0
  for k in "${KEEP[@]}"; do [[ "$f" == "$k" ]] && keep=1; done
  if [[ $keep -eq 0 ]]; then
    echo "archive: $f"
    git mv -k "$f" _archive/ 2>/dev/null || mv "$f" _archive/
  fi
done

# data lives on scratch, never in git
if [[ ! -e data ]]; then
  ln -s "${SCRATCH:-/glade/derecho/scratch/$USER}/latentsky" data
fi
grep -qx "data" .gitignore 2>/dev/null || echo "data" >> .gitignore

echo "Done. Next: git add -A && git commit -m 'reset: archive legacy tree, new structure'"
