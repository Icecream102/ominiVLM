PYTHON ?= python
CONFIG ?= configs/full_pipeline_5090.json

.PHONY: test lint-config dry-run summarize

test:
	$(PYTHON) -m pytest -q

lint-config:
	$(PYTHON) -m json.tool $(CONFIG) >/dev/null
	$(PYTHON) -m compileall -q dataset evaluation model scripts trainer

dry-run:
	$(PYTHON) scripts/run_experiment.py --config $(CONFIG) --dry_run

summarize:
	$(PYTHON) scripts/summarize_experiment.py benchmark_results/full_pipeline_20260811 \
		--output benchmark_results/full_pipeline_20260811/summary.json
