# New Pipeline Route PR Guide

Use this when the reported metric stays the same, but the computation route
changes. Examples include a new normalizer, transcription model, scorer
backend, provider, or toolkit chain.

## Required Changes

- Add a route in `tasks/<task>/routes.yaml`.
- Keep `metric` canonical, such as `cer`, `wer`, `spk_sim`, or `dnsmos`.
- Put method selectors in `execution_metrics`, such as `sim/ecapa-tdnn`.
- Add or update task pipeline selection logic only as needed.
- Preserve the default route unless the PR explicitly changes it.
- Update task docs with reported metric, selector, exact `pipeline_id`, and
  node chain.
- Regenerate `docs/pipeline_catalog.jsonl`.

## Required Tests

- Default route remains unchanged, unless intentionally changed.
- New route can be selected by exact `pipeline_id`.
- `metric routes` lists the route under the correct task, language, and
  canonical metric.
- `describe -> run -> report` preserves `pipeline_id`, `pipeline_kind`, member
  IDs, and computation nodes.
- Illegal route combinations fail with clear errors.
- Report trace contains the selected nodes.
- `env setup --pipeline ... --dry-run` selects only the route's declared
  optional environments.

## PR Must State

- Existing reported metric.
- New route selector or method.
- Added exact `pipeline_id`.
- Whether the default route changed.
- Score comparability boundary.
- Small fixture used for the exact route run.
