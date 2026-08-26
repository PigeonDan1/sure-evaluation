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
Environment setup and checking rebuild the same registered identity before
resolving optional nodes.

For heavyweight metrics, reproducibility also requires:

- `node_env.yaml`
- committed dependency locks for frozen node-local environments
- immutable source revision and source-tree metadata for fetched code
- model provider and model id
- revision or checkpoint path when available
- environment-variable override
- license and citation notes
- setup or download command
- runtime-managed preprocessing details when a node owns decode, resampling, or
  text extraction internally

Use dry-run commands before preparing large assets:

```bash
sure-eval metric routes tts --language zh --metric dnsmos --json
sure-eval metric describe tts \
  --pipeline-id tts.zh.dnsmos.dnsmos_v1 --output pipeline.json
sure-eval env setup --pipeline pipeline.json --dry-run --json
sure-eval env setup --node scoring/dnsmos --dry-run --json
sure-eval env download --node scoring/dnsmos --dry-run --json
sure-eval env setup --node normalization/funasr_itn --dry-run --json
sure-eval env setup --node normalization/nemo_norm --dry-run --json
```

Fetched runtime code must not track a moving branch. Store its repository,
commit SHA, and source subdirectory in a packaged lock file; verify the checked
out revision during setup and record the resolved source tree in the node
trace or runtime metadata.
