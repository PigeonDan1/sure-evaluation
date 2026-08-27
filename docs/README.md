# Documentation

## Use SURE-EVALUATION

- [Installation](installation.md): clean source installation and base smoke test
- [Task Guides](tasks/README.md): metrics, input contracts, exact pipelines, and examples
- [Environment Management](environment.md): prepare only selected optional nodes
- [Pipeline Catalog](pipeline_catalog.md): committed route-to-node inventory
- [Pipeline Atlas](atlas/index.html): the catalog drawn as one interactive map from task to report
- [Reproducibility](reproducibility.md): identity, reports, locks, and runtime assets

The standard user flow is:

```bash
sure-eval metric routes <task> --language <lang> --metric <metric>
sure-eval metric describe <task> --pipeline-id <pipeline-id> --output pipeline.json
sure-eval env setup --pipeline pipeline.json --dry-run
sure-eval env check --pipeline pipeline.json
sure-eval metric run --pipeline pipeline.json ...
```

## Contribute

Start with [Contributing](contributing.md). It classifies the PR and links to a
focused guide for a new task, metric, route, node/tool version, or maintenance
change. Use [Add Evaluation Capabilities](add_a_metric.md) when the category is
unclear.

## Agent Integration

Agents and evaluation harnesses should follow the
[Agent Contract](agent_contract.md). Machine consumers should prefer
`metric routes --json`, exact pipeline IDs, pipeline-based environment
commands, and the structured run artifacts.
