# Reproducibility

Every run writes:

- `report.json`
- `pipeline_description.json`

The pipeline description records `pipeline_id`, canonical `metric`,
`execution_metrics`, `pipeline_kind`, `member_pipeline_ids`, selected nodes,
versions, config paths, and input contracts. `computation_node_ids` includes
score-affecting conversions.

For heavyweight metrics, reproducibility also requires:

- `node_env.yaml`
- model provider and model id
- revision or checkpoint path when available
- environment-variable override
- license and citation notes
- setup or download command

Use dry-run commands before preparing large assets:

```bash
sure-eval env setup --node scoring/dnsmos --dry-run --json
sure-eval env download --node scoring/dnsmos --dry-run --json
```
