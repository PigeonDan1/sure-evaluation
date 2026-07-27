# New Task PR Guide

Use this when the contribution introduces new input roles, row format,
alignment semantics, aggregation policy, or report structure.

## Required Changes

- Add `src/sure_eval/evaluation/tasks/<task>/`.
- Add `manifest.yaml`, `routes.yaml`, `pipeline.py`, and task-local `README.md`.
- Add a script adapter in `src/sure_eval/evaluation/scripts/<task>.py`.
- Register the task in script dispatch and CLI/agent surfaces.
- Reuse existing nodes when possible; add new nodes only for reusable stages.
- Add `node_env.yaml` for heavyweight dependencies, binaries, models, or
  checkpoints.
- Add `docs/tasks/<task>.md` and link it from `docs/tasks/README.md`.
- Update `README.md`, `README_ZH.md`, and `scripts/generate_pipeline_catalog.py`.
- Regenerate `docs/pipeline_catalog.jsonl`.

## Required Tests

- Task pipeline tests on small fixtures.
- Script `describe_pipeline` and `run_task` tests.
- CLI `metric describe -> metric run` tests.
- Input contract tests for missing and malformed roles.
- Env check tests for node-local environments.

## PR Must State

- New task name and supported languages.
- Required and optional input roles.
- Main reported metric and aggregation policy.
- Added `pipeline_id` values.
- Small fixture used for `describe -> run -> report`.
