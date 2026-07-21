# Repository Agent Notes

This repository is the standalone deterministic SURE evaluation engine. It is
not the harness repository and not a model onboarding workspace.

For agent or TUI usage:

1. Inspect the route and environment plan first:

   ```bash
   sure-eval agent plan <task> --language <lang> --metric <metric> --json
   ```

2. Prepare only the node environments selected by that plan. Do not create all
   optional environments up front.
3. Run scoring through the existing deterministic commands:

   ```bash
   sure-eval metric describe <task> --language <lang> --metric <metric> --output pipeline.json
   sure-eval metric run --pipeline pipeline.json ...
   ```

4. Keep generated reports, checkpoints, model weights, caches, and node-local
   virtual environments out of git.

Agent contract details live in `docs/agent_contract.md`.
