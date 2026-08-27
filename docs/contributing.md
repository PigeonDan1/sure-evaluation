# Contributing

Start here for every PR. SURE-EVAL contributions are reviewed by route
identity, reproducibility, and tests.

## Pick One Guide

Use [Add Evaluation Capabilities](./add_a_metric.md) if you are unsure which
category applies.

| PR type | Use when | Guide |
|:--|:--|:--|
| New task | New input roles, row format, alignment, or aggregation | [New Task](./pr_guides/new_task.md) |
| New metric | New reported score definition for an existing task | [New Metric](./pr_guides/new_metric.md) |
| New pipeline route | Same reported metric, different node chain or backend | [New Pipeline Route](./pr_guides/new_route.md) |
| Node/tool/version change | One route stage changes implementation, environment, or version | [Node Change](./pr_guides/node_change.md) |
| Bug, docs, tests only | No user-visible scoring route change | [Maintenance](./pr_guides/maintenance.md) |

## Common Rules

- Keep `metric` canonical, such as `cer`, `wer`, `spk_sim`, or `dnsmos`.
- Put method or compatibility selectors in `execution_metrics`.
- Select same-metric variants with exact `pipeline_id`.
- `sure-eval metric routes` must discover every registered atomic route; do not
  maintain a second hand-written list for route variants.
- For every node/tool/version change, update the node README using
  [Node README Template](./node_readme_template.md).
- Do not change a default route without saying so in the PR.
- Do not commit model weights, checkpoints, `.venv`, caches, reports, secrets,
  or private absolute paths.

## Common Checks

Run focused tests for the changed task, then broader checks when shared code is
touched:

```bash
PYTHONPATH=src uv run pytest -q <focused tests>
PYTHONPATH=src uv run ruff check .
git diff --check
```

After route changes, regenerate and inspect the catalog:

```bash
PYTHONPATH=src uv run python scripts/generate_pipeline_catalog.py
git diff -- docs/pipeline_catalog.jsonl
python scripts/generate_pipeline_atlas.py
```

The generator discovers atomic rows from `routes.yaml`. Only new task
registration or a deliberately curated multi-metric bundle should require a
generator code change.

For heavyweight nodes, also run:

```bash
sure-eval env check --node <node-id>
sure-eval env setup --node <node-id> --dry-run
```

For route-facing changes, also verify exact selection:

```bash
sure-eval metric routes <task> --language <lang> --metric <metric> --json
sure-eval metric describe <task> --pipeline-id <pipeline-id> --output pipeline.json
sure-eval env setup --pipeline pipeline.json --dry-run --json
```

## CI Gates

Required PR checks are intentionally portable: they must pass on a clean GitHub
runner without model checkpoints, secrets, or node-local virtual environments.
They run `ruff`, whitespace checks, catalog freshness, package build/install
smoke tests, CLI smoke tests, and the test suite excluding checkpoint-backed
environment assertions.

Use this local approximation before opening a broad PR:

```bash
uv run --extra dev ruff check .
git diff --check
uv run --extra dev --extra audio --extra canonical pytest -q --ignore=tests/test_evaluation_env_check.py
uv run --extra dev --extra audio --extra canonical pytest tests/test_evaluation_env_check.py -q -k "in_process or pip_runtime or dry_run or doctor or cache_dir or se_pip_runtime_nodes_are_declared"
uv run --extra dev --extra canonical python scripts/generate_pipeline_catalog.py
git diff --exit-code docs/pipeline_catalog.jsonl
```

Checkpoint-backed node-local checks live in the optional Heavy Nodes workflow.
Run it manually with `run_real_checks=true` only on a prepared self-hosted
runner labeled `sure-eval-heavy`.

## Open The PR

Use the PR template and link the guide you followed. For route or metric
changes, include a small `describe -> run -> report` check that proves the
selected `pipeline_id` is preserved.
