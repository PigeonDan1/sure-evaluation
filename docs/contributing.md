# Contributing

Every metric contribution should be reproducible from declarations and tests.

Before opening a PR:

```bash
sure-eval doctor
sure-eval metric describe asr --language zh --metric cer --json
pytest -q
```

Do not commit:

- virtual environments
- checkpoints
- generated reports
- local caches
- private absolute paths
- API keys or credentials

If a node depends on a model, external binary, or heavyweight package stack, add
or update `node_env.yaml`.

