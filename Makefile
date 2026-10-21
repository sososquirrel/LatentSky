PY := /glade/work/sabramian/conda-envs/latentsky-env/bin/python
NV := /glade/work/sabramian/conda-envs/latentsky-env/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH := $(NV)/cublas/lib:$(NV)/cuda_cupti/lib:$(NV)/cuda_nvrtc/lib:$(NV)/cuda_runtime/lib:$(NV)/cudnn/lib:$(NV)/cufft/lib:$(NV)/curand/lib:$(NV)/cusolver/lib:$(NV)/cusparse/lib:$(NV)/cusparselt/lib:$(NV)/nccl/lib:$(NV)/nvjitlink/lib:$(NV)/nvtx/lib:$(LD_LIBRARY_PATH)

.PHONY: results test lint paper overleaf-pull overleaf-push
results:
	for f in results/fig_R*.py; do $(PY) $$f; done
test:
	$(PY) -m pytest tests/
lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .

# --- paper: home is paper/, Overleaf mirrors that folder only ------------------
# one-time: git remote add overleaf https://git.overleaf.com/<project-id>
OVERLEAF_BRANCH := master
paper:
	cd paper && module load texlive/20250308 2>/dev/null; \
	pdflatex -interaction=nonstopmode main.tex >/dev/null; bibtex main >/dev/null; \
	pdflatex -interaction=nonstopmode main.tex >/dev/null; pdflatex -interaction=nonstopmode main.tex | tail -2
# bring the co-author's Overleaf edits into paper/ (commit your own work first)
overleaf-pull:
	git subtree pull --prefix paper overleaf $(OVERLEAF_BRANCH) -m "paper: merge Overleaf edits"
# send paper/ to Overleaf (pulls first, so nobody's edits are overwritten)
overleaf-push: overleaf-pull
	git subtree push --prefix paper overleaf $(OVERLEAF_BRANCH)
