# Node Change PR Guide

Use this when one route stage changes implementation, dependency handling,
runtime setup, model provider, toolkit wrapper, or version.

## Required Changes

- Update or add files under `src/sure_eval/evaluation/nodes/<stage>/<name>/`.
- Update node `manifest.yaml` with version, stage, implementation, schemas, and
  internal stages.
- Update `node_env.yaml` for dependencies, binaries, checkpoints, downloads, or
  environment variables.
- Update node README using
  [Node README Template](../node_readme_template.md). At minimum, document
  task scenarios, input/output schemas, versioned computation, runtime assets,
  source/reference links, and known limitations.
- Add a new node version, node id, or route when scores are expected to change.
- Keep runtime assets local; do not commit model files or generated outputs.
- For fetched source code, commit an immutable revision lock and package the
  setup script. Do not fetch a moving branch or depend on a contributor's
  checkout path.
- For a frozen uv runtime, commit `uv.lock`, declare `frozen: true`, and verify
  that the wheel contains the project, lock, and any setup scripts.

## Required Tests

- Node-level tests for the changed behavior.
- Route-level regression tests when the node is used by a route.
- Env check tests for node-local setup.
- Source-install and built-wheel checks when node setup depends on packaged
  files.
- Heavyweight smoke test when practical.
- Skip policy or local smoke evidence when CI cannot run real inference.

## PR Must State

- Node id and stage.
- Whether scores are expected to change.
- Runtime assets and environment variables.
- External toolkit, model id, checkpoint, and license constraints.
- Node README sections updated and any unknown external source explicitly
  marked.
- Smoke test command and result for heavyweight nodes.
