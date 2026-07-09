# Add A Metric

1. Add a node under `src/sure_eval/evaluation/nodes/<stage>/<name>/`.
2. Add `manifest.yaml` with id, version, stage, input schema, output schema, and implementation.
3. Add `node_env.yaml` if the node has heavyweight dependencies, checkpoints, external binaries, or model downloads.
4. Add or update a route in `src/sure_eval/evaluation/tasks/<task>/routes.yaml`.
5. Add focused tests for the node and route.
6. Verify with:

```bash
sure-eval metric describe <task> --metric <metric> --json
sure-eval env setup --task <task> --metric <metric> --dry-run
```

Wrappers around external toolkits should keep upstream behavior unchanged unless
the change is intentional and documented.

