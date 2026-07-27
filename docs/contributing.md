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
```

For heavyweight nodes, also run:

```bash
sure-eval env check --node <node-id>
sure-eval env setup --node <node-id> --dry-run
```

## Open The PR

Use the PR template and link the guide you followed. For route or metric
changes, include a small `describe -> run -> report` check that proves the
selected `pipeline_id` is preserved.
