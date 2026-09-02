.PHONY: results test lint
results:
	for f in results/fig_R*.py; do python $$f; done
test:
	pytest tests/
lint:
	ruff check . && ruff format --check .
