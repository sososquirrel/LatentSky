PY := /glade/work/sabramian/conda-envs/latentsky-env/bin/python
NV := /glade/work/sabramian/conda-envs/latentsky-env/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH := $(NV)/cublas/lib:$(NV)/cuda_cupti/lib:$(NV)/cuda_nvrtc/lib:$(NV)/cuda_runtime/lib:$(NV)/cudnn/lib:$(NV)/cufft/lib:$(NV)/curand/lib:$(NV)/cusolver/lib:$(NV)/cusparse/lib:$(NV)/cusparselt/lib:$(NV)/nccl/lib:$(NV)/nvjitlink/lib:$(NV)/nvtx/lib:$(LD_LIBRARY_PATH)

.PHONY: results test lint
results:
	for f in results/fig_R*.py; do $(PY) $$f; done
test:
	$(PY) -m pytest tests/
lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .
