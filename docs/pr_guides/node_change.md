# Node Change PR Guide

Use this when one route stage changes implementation, dependency handling,
runtime setup, model provider, toolkit wrapper, or version.

## Required Changes

- Update or add files under `src/sure_eval/evaluation/nodes/<stage>/<name>/`.
- Update node `manifest.yaml` with version, stage, implementation, schemas, and
  internal stages.
- Update `node_env.yaml` for dependencies, binaries, checkpoints, downloads, or
  environment variables.
- Update node README when runtime behavior is user-visible.
- Add a new node version, node id, or route when scores are expected to change.
- Keep runtime assets local; do not commit model files or generated outputs.

## Required Tests

- Node-level tests for the changed behavior.
- Route-level regression tests when the node is used by a route.
- Env check tests for node-local setup.
- Heavyweight smoke test when practical.
- Skip policy or local smoke evidence when CI cannot run real inference.

## PR Must State

- Node id and stage.
- Whether scores are expected to change.
- Runtime assets and environment variables.
- External toolkit, model id, checkpoint, and license constraints.
- Smoke test command and result for heavyweight nodes.
