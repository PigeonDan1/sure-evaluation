## Start Here

Read [Contributing](docs/contributing.md). If the category is unclear, use
[Add Evaluation Capabilities](docs/add_a_metric.md).

## Summary

## PR Type

Select one guide:

- [ ] [New task](docs/pr_guides/new_task.md)
- [ ] [New metric](docs/pr_guides/new_metric.md)
- [ ] [New pipeline route](docs/pr_guides/new_route.md)
- [ ] [Node/tool/version change](docs/pr_guides/node_change.md)
- [ ] [Maintenance](docs/pr_guides/maintenance.md)

Guide followed:

## Evaluation Identity

- Reported `metric`:
- `execution_metrics`, if different:
- Added or changed `pipeline_id`:
- Default route changed: yes / no
- Score comparability changed: yes / no

## Changed Declarations

- Task manifest: yes / no / n/a
- Task routes: yes / no / n/a
- Node manifest: yes / no / n/a
- `node_env.yaml`: yes / no / n/a
- Catalog generator and `docs/pipeline_catalog.jsonl`: yes / no / n/a

## Validation

Paste exact commands run:

```bash

```

For route or metric PRs, include one exact `pipeline_id` check:

```bash
# sure-eval metric describe ... --pipeline-id ...
# sure-eval metric run --pipeline ...
```

For heavyweight nodes, include a real smoke test or explain the skip:

```bash

```

## Safety

- [ ] No `.venv`, checkpoints, generated reports, local cache paths, or credentials committed
- [ ] No private absolute paths in committed config or docs
- [ ] Score-affecting behavior is declared in route/node identity
- [ ] Existing default route behavior is tested or explicitly changed
