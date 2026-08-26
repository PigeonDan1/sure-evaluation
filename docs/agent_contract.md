# Agent Contract

This document defines how Codex, Kimi Code, and other agents should use
SURE-EVALUATION as a versioned, reviewable evaluation engine.

## Scope

SURE-EVALUATION owns metric routing, node version selection, node environment
diagnostics, score execution, and structured evaluation reports. It does not
own the system being evaluated, dataset sampling, harness state machines, or
model onboarding.

An agent should treat this repository as the scoring engine behind a larger
workflow.

## Exact-Pipeline Flow

1. Discover registered alternatives. Never infer a route from a metric name.

   ```bash
   sure-eval metric routes asr --language zh --metric cer --json
   ```

   The response gives exact pipeline IDs, the default marker, ordered
   computation nodes, required roles, route selectors, and declared runtimes.

2. Describe, prepare, and validate the selected identity.

   ```bash
   sure-eval metric describe asr \
     --pipeline-id asr.zh.cer.wetext_norm_zh_itn_v1.wenet_cer_v1 \
     --output pipeline.json
   sure-eval env setup --pipeline pipeline.json --dry-run --json
   sure-eval env setup --pipeline pipeline.json
   sure-eval env check --pipeline pipeline.json --json
   ```

3. Score and retain both output artifacts.

   ```bash
   sure-eval metric run --pipeline pipeline.json \
     --ref-file ref.txt --hyp-file hyp.txt --output-dir eval_out \
     --validate-env
   ```

   `metric run` is the scoring entrypoint. `metric routes`, `metric describe`,
   environment commands, and `agent plan` do not calculate a score.

Use `sure-eval agent plan ... --json` when a harness needs route resolution and
environment readiness in one payload. For same-metric alternatives, pass an
exact `--pipeline-id`; use `metric routes` first to obtain that ID.

## Plan Payload

`sure-eval agent plan --json` emits `schema=sure.eval.agent_plan.v1` with:

- `task`, `language`, `metrics`: normalized user selection.
- `root_env`: Python/package/cache checks needed before any route can run.
- `selected_routes`: one entry per requested metric.
- `selected_routes[].pipeline_id`: concrete versioned pipeline id.
- `selected_routes[].resolved_metric`: canonical reported metric for this selection.
- `selected_routes[].pipeline_kind`: `atomic` or `bundle`.
- `selected_routes[].member_pipeline_ids`: atomic member IDs for bundle selections.
- `selected_routes[].computation_node_ids`: score-affecting nodes, including conversions.
- `selected_routes[].route_config_path`: repository-relative task route file.
- `selected_routes[].describe_entrypoint`: dotted Python entrypoint for route
  description.
- `selected_routes[].script_entrypoint`: dotted Python scoring entrypoint.
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

Prepare optional node environments only after exact route selection:

```bash
sure-eval metric routes asr --language ar --metric cer --json
sure-eval metric describe asr \
  --pipeline-id asr.ar.cer.nemo_norm_ar_tn_v1.wenet_cer_v1 \
  --output pipeline.json
sure-eval env setup --pipeline pipeline.json --dry-run
sure-eval env setup --pipeline pipeline.json
sure-eval env check --pipeline pipeline.json --json
```

Node-local virtual environments, heavy models, and checkpoints remain local
assets. They must not be committed.

An agent must review the dry-run and preserve the selected pipeline identity.
A node may use a
committed lock file and a packaged post-setup script to prepare immutable
upstream source; agents must not replace its revision with a branch head or a
machine-local source path.

## Route Configuration

Agents should not guess metric behavior from names. The source of truth is:

- task manifests: `src/sure_eval/evaluation/tasks/<task>/manifest.yaml`
- task routes: `src/sure_eval/evaluation/tasks/<task>/routes.yaml`
- node manifests: `src/sure_eval/evaluation/nodes/<stage>/<name>/manifest.yaml`
- node environments: `src/sure_eval/evaluation/nodes/<stage>/<name>/node_env.yaml`

If a collaborator adds a new task, metric, route, or node version, they should
update those declarations, tests, and docs. Agents should validate the change
with `metric routes`, exact `metric describe`, pipeline-based environment
checks, and `metric run` where practical.

## Identity Rules

`pipeline_id` names the computation:
`task.language.metric.node_version...`. The `metric` field in reports and the
catalog is canonical. When one metric has multiple route variants, agents
should select the exact `pipeline_id`. Compatibility aliases and method
selectors are recorded as `execution_metrics`; for example,
`sim/wavlm-large` executes a `spk_sim` pipeline through the WavLM node.
For example, TTS `cer` defaults to the Paraformer-ZH route, while
`tts.zh.cer.qwen3_asr_1_7b_v1.punctuation_strip_norm_v1.wenet_cer_v1`
selects the Qwen3-ASR-1.7B transcription route for the same reported metric.
Likewise, `spk_sim` variants such as WavLM, ECAPA-TDNN, and ERes2Net are
distinguished by exact `pipeline_id`, not by changing the canonical `metric`.
Arabic ASR uses the canonical metric `cer`; its exact default identity is
`asr.ar.cer.nemo_norm_ar_tn_v1.wenet_cer_v1`.
After `metric describe` writes a pipeline JSON, `metric run --pipeline` must
execute that selected identity and reject reports whose `pipeline_id`,
`pipeline_kind`, member IDs, or computation nodes diverge from the description.
`env setup --pipeline` and `env check --pipeline` must reject stale or edited
pipeline identity fields before resolving node environments.
Multi-metric requests are `pipeline_kind=bundle` and list atomic members in
`member_pipeline_ids`.
