PYTHON ?= python

.PHONY: test debug pipeline figures tables clean-debug

test:
	$(PYTHON) -m compileall -q src scripts
	$(PYTHON) -m pytest

debug:
	./scripts/run_debug.sh

pipeline:
	./scripts/run_pipeline.sh

figures:
	$(PYTHON) -m src.make_plots --results-dir results/main --output-dir results/figures

tables:
	$(PYTHON) -m src.make_tables --results-dir results/main --output-dir results/tables

clean-debug:
	rm -rf data/debug results/debug
