# New Metric PR Guide

Use this when an existing task gets a new reported score definition. This is
not for alternate backends of an existing metric.

## Required Changes

- Add or reuse a scoring node under `src/sure_eval/evaluation/nodes/scoring/`.
- Add the metric to the task `manifest.yaml`.
- Add or update the metric input contract.
- Add a route in `routes.yaml`.
- Update task pipeline dispatch and report details.
- Document the metric in `docs/tasks/<task>.md`.
- Update `scripts/generate_pipeline_catalog.py`.
- Regenerate `docs/pipeline_catalog.jsonl`.

## Required Tests

- Scorer correctness on small fixtures.
- Aggregation edge cases, including missing keys and empty references when
  applicable.
- Script and CLI `describe -> run -> report` tests.
- Report shape tests for `report.json` and `pipeline_description.json`.
- Regression tests when replacing or aliasing old behavior.

## PR Must State

- Mathematical definition and denominator.
- Whether higher or lower is better.
- Reported `metric` name.
- Accepted `execution_metrics`, if any.
- Added `pipeline_id` values.
- Score comparability versus existing metrics.
