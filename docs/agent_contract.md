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
   sure-eval agent plan tts --language zh --metrics cer,spk_sim --json
   ```

   The plan resolves the configured pipeline from `tasks/<task>/routes.yaml`,
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
- `selected_routes[].resolved_metric`: canonical reported metric for this selection.
- `selected_routes[].pipeline_kind`: `atomic` or `bundle`.
- `selected_routes[].member_pipeline_ids`: atomic member IDs for bundle selections.
- `selected_routes[].computation_node_ids`: score-affecting nodes, including conversions.
- `selected_routes[].route_config_path`: repository-relative task route file.
- `selected_routes[].describe_entrypoint`: dotted Python entrypoint for route
  description.
- `selected_routes[].script_entrypoint`: dotted Python entrypoint for
  deterministic scoring.
- `selected_routes[].executor`: dotted task executor called by the route script.
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
sure-eval agent plan tts --language zh --metrics cer,dnsmos --json
sure-eval env setup --task tts --language zh --metrics cer,dnsmos --dry-run
sure-eval env setup --task tts --language zh --metrics cer,dnsmos
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

## Identity Rules

`pipeline_id` names the computation:
`task.language.metric.node_version...`. The `metric` field in reports and the
catalog is canonical. When one metric has multiple route variants, agents
should select the exact `pipeline_id`. Compatibility aliases and method
selectors are recorded as `execution_metrics`; for example,
`sim/wavlm-large` executes a `spk_sim` pipeline through the WavLM node.
Multi-metric requests are `pipeline_kind=bundle` and list atomic members in
`member_pipeline_ids`.
