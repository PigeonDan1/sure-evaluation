# Reproducibility

Every run writes:

- `report.json`
- `pipeline_description.json`

The pipeline description records `pipeline_id`, canonical `metric`,
`execution_metrics`, `pipeline_kind`, `member_pipeline_ids`, selected nodes,
versions, config paths, and input contracts. `computation_node_ids` includes
score-affecting conversions.

When a run starts from a pipeline JSON, `pipeline_id` is the selected execution
identity. The run path validates that the returned report still matches the
described `pipeline_id`, `pipeline_kind`, member IDs, and computation nodes.

For heavyweight metrics, reproducibility also requires:

- `node_env.yaml`
- model provider and model id
- revision or checkpoint path when available
- environment-variable override
- license and citation notes
- setup or download command
- runtime-managed preprocessing details when a node owns decode, resampling, or
  text extraction internally

Use dry-run commands before preparing large assets:

```bash
sure-eval env setup --node scoring/dnsmos --dry-run --json
sure-eval env download --node scoring/dnsmos --dry-run --json
```
