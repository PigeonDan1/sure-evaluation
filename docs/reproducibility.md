# Reproducibility

Every run writes:

- `report.json`
- `pipeline_description.json`

The pipeline description records selected nodes, versions, route ids, config
paths, and input contracts.

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

