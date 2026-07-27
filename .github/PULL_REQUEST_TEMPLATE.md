## Summary

## Contribution Type

Select one category from `docs/add_a_metric.md`:

- [ ] New task
- [ ] New metric for an existing task
- [ ] New pipeline route for an existing metric
- [ ] Node, tool, environment, or version change
- [ ] Bug fix / docs / tests only

## Evaluation Identity

- Reported metric(s):
- Execution selector(s), if different:
- Added or changed `pipeline_id` values:
- Default route changed: yes / no
- Score comparability changed: yes / no

## Route And Node Changes

- Task manifest updated: yes / no / n/a
- Task routes updated: yes / no / n/a
- Node manifest or `node_env.yaml` updated: yes / no / n/a
- Runtime assets required:
- External toolkit, model, or checkpoint:

## Documentation

- [ ] Task guide updated when user-visible behavior changes
- [ ] Node README updated for new or changed nodes
- [ ] `scripts/generate_pipeline_catalog.py` updated when routes change
- [ ] `docs/pipeline_catalog.jsonl` regenerated when routes change
- [ ] README or agent docs updated when public workflow changes

## Validation

Paste the exact checks run:

```bash

```

For route changes, include at least one `describe -> run -> report` check using
the exact `pipeline_id`.

For heavyweight nodes, include the real smoke test or explain why it is skipped:

```bash

```

## Safety Checklist

- [ ] No `.venv`, checkpoints, generated reports, local cache paths, or credentials committed
- [ ] No private absolute paths in committed config or docs
- [ ] New score-affecting behavior is declared in route/node identity
- [ ] Existing default route behavior is covered by tests or explicitly changed
