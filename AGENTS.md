# Repository Agent Notes

This repository is the standalone, versioned SURE evaluation engine. It is not
the harness repository and not a model onboarding workspace.

For agent or TUI usage:

1. Discover exact routes first:

   ```bash
   sure-eval metric routes <task> --language <lang> --metric <metric> --json
   ```

2. Describe one exact `pipeline_id`, then prepare only that pipeline's optional
   node environments:

   ```bash
   sure-eval metric describe <task> --pipeline-id <pipeline-id> --output pipeline.json
   sure-eval env setup --pipeline pipeline.json --dry-run
   sure-eval env check --pipeline pipeline.json
   ```

3. Run scoring through the pipeline JSON:

   ```bash
   sure-eval metric run --pipeline pipeline.json ...
   ```

4. Keep generated reports, checkpoints, model weights, caches, and node-local
   virtual environments out of git.

Agent contract details live in `docs/agent_contract.md`.
