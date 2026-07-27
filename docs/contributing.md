# Contributing

SURE-EVAL contributions must be reproducible from declarations, route identity,
and tests. Before changing evaluation behavior, classify the PR with
[Add Evaluation Capabilities](./add_a_metric.md).

## Contributor Flow

1. Classify the change.

   Use one category: new task, new metric for an existing task, new pipeline
   route for an existing metric, or node/tool/version change. Do not describe a
   route selector as a new metric.

2. Update declarations first.

   Task behavior belongs in `src/sure_eval/evaluation/tasks/<task>/`.
   Reusable work belongs in `src/sure_eval/evaluation/nodes/<stage>/<name>/`.
   Route selection belongs in `routes.yaml`; public metric names belong in the
   task manifest and docs.

3. Preserve identity.

   Every user-visible route must have a stable `pipeline_id` in the form
   `task.language.metric.node_version...`. The reported `metric` is canonical.
   Method-specific selectors such as `sim/wavlm-large` belong in
   `execution_metrics`, not in the reported metric name.

4. Validate describe, run, and report together.

   For route changes, generate a pipeline JSON with the exact `pipeline_id`,
   run it on a small fixture, and confirm `report.json` and
   `pipeline_description.json` keep the same identity.

5. Fill the PR template.

   State the contribution category, changed `pipeline_id` values, default route
   impact, score comparability impact, docs updated, and checks run.

## Files To Update

Update only the files that match the change:

- task manifest: `src/sure_eval/evaluation/tasks/<task>/manifest.yaml`
- task routes: `src/sure_eval/evaluation/tasks/<task>/routes.yaml`
- task pipeline code: `src/sure_eval/evaluation/tasks/<task>/pipeline.py`
- script adapter: `src/sure_eval/evaluation/scripts/<task>.py`
- node manifest or environment: `src/sure_eval/evaluation/nodes/<stage>/<name>/`
- task guide: `docs/tasks/<task>.md`
- task-local README: `src/sure_eval/evaluation/tasks/<task>/README.md`
- catalog generator: `scripts/generate_pipeline_catalog.py`
- generated catalog: `docs/pipeline_catalog.jsonl`
- public docs: `README.md`, `docs/agent_contract.md`, or
  `docs/pipeline_catalog.md` when workflow or schema semantics change

## Required Checks

Run focused checks for the changed task, then broader checks when shared code is
touched:

```bash
PYTHONPATH=src uv run pytest -q <focused tests>
PYTHONPATH=src uv run ruff check .
git diff --check
```

Regenerate the catalog after route changes:

```bash
PYTHONPATH=src uv run python scripts/generate_pipeline_catalog.py
git diff -- docs/pipeline_catalog.jsonl
```

For heavyweight nodes, also run:

```bash
sure-eval env check --node <node-id>
sure-eval env setup --node <node-id> --dry-run
```

If real inference is required but unsuitable for CI, run one local smoke test
and record the node id, backend, checkpoint or model id, input fixture, and
score in the PR.

## Do Not Commit

- virtual environments
- checkpoints or model weights
- generated reports
- local caches
- private absolute paths
- API keys or credentials

Use `node_env.yaml` and environment variables to describe heavyweight runtime
requirements without committing local assets.
