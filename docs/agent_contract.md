# Agent Contract

This document defines how Codex, Kimi Code, and other TUI agents should use
SURE-EVAL as a deterministic evaluation skill.

## Scope

SURE-EVAL owns metric routing, node version selection, node environment
diagnostics, and deterministic scoring. It does not own model inference,
dataset sampling, harness state machines, or model onboarding.

An agent should treat this repository as the scoring engine behind a larger
workflow.

## Two-Phase Flow

1. Plan and validate.

   ```bash
   sure-eval agent plan asr --language zh --metric cer --json
   sure-eval agent plan tts --language zh --metrics tts_cer,sim/wavlm-large --json
   ```

   The plan resolves the configured route from `tasks/<task>/routes.yaml`,
   expands the selected versioned nodes, checks only those node environments,
   and returns setup commands for missing requirements.

2. Score.

   ```bash
   sure-eval metric describe asr --language zh --metric cer --output pipeline.json
   sure-eval metric run --pipeline pipeline.json \
     --ref-file ref.txt --hyp-file hyp.txt --output-dir eval_out \
     --validate-env
   ```

   `metric run` remains the only deterministic scoring entrypoint. `agent plan`
   is a readiness and routing interface, not a second evaluator.

## Plan Payload

`sure-eval agent plan --json` emits `schema=sure.eval.agent_plan.v1` with:

- `task`, `language`, `metrics`: normalized user selection.
- `root_env`: Python/package/cache checks needed before any route can run.
- `selected_routes`: one entry per requested metric.
- `selected_routes[].pipeline_id`: concrete deterministic pipeline id.
- `selected_routes[].nodes`: ordered versioned node ids and runtimes.
- `selected_routes[].required_roles`: input roles required by the scorer.
- `selected_routes[].env_checks`: node-level status, fix, and setup hints.
- `can_run_now`: true only when all required root and selected node checks pass.
- `blocking_issues`: concise reasons an agent must resolve before scoring.
- `next_steps`: setup commands or the next metric command to run.

## Environment Timing

Install the root package before route inspection:

```bash
pip install -e .
sure-eval doctor
```

Prepare optional node environments only after route selection:

```bash
sure-eval agent plan asr --language zh --metric cer --json
sure-eval env setup --task asr --language zh --metric cer --dry-run
sure-eval agent plan tts --language zh --metrics tts_cer,dnsmos --json
sure-eval env setup --task tts --language zh --metrics tts_cer,dnsmos --dry-run
sure-eval env setup --task tts --language zh --metrics tts_cer,dnsmos
```

Node-local virtual environments, heavy models, and checkpoints remain local
assets. They must not be committed.

## Route Configuration

Agents should not guess metric behavior from names. The source of truth is:

- task manifests: `src/sure_eval/evaluation/tasks/<task>/manifest.yaml`
- task routes: `src/sure_eval/evaluation/tasks/<task>/routes.yaml`
- node manifests: `src/sure_eval/evaluation/nodes/<stage>/<name>/manifest.yaml`
- node environments: `src/sure_eval/evaluation/nodes/<stage>/<name>/node_env.yaml`

If a collaborator adds a new task, metric, route, or node version, they should
update those declarations, tests, and docs. Agents should validate the change
with both `agent plan` and `metric describe`.
