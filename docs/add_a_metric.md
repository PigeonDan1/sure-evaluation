# Add Evaluation Capabilities

Use this decision tree before opening a PR. The goal is to match the change to
the smallest extension point that preserves reproducibility and keeps metric
semantics clear.

In this repository, a **metric family** is the reported evaluation quantity
such as CER, WER, MER, BLEU, DER, accuracy, SIM, or MOS. A **route selector**
is the CLI/API value that selects a concrete pipeline for that metric family.
For example, `cer_canonical` is a route selector for CER with canonical
normalization. It is not a new metric family.

## Decision Tree

1. Does the contribution introduce a new input/output evaluation problem?
   Add a **new task**.

2. Does it introduce a new reported score definition for an existing task?
   Add a **new metric family for an existing task**.

3. Does it keep the same reported metric family but use a different
   normalization, transcription, scorer, or backend chain?
   Add a **new route for an existing metric family**.

4. Does it change one node inside an existing route, such as a new normalizer
   ruleset, model provider, external toolkit wrapper, or checkpoint-backed
   implementation?
   Add or update a **node/tool/version**.

When in doubt, prefer the narrowest category that matches the actual scientific
claim. Do not create a new task for a new scorer, and do not document a route
selector as a new metric family.

## Category 1: New Task

Use this when the input roles, row format, alignment semantics, or aggregation
policy do not fit an existing task.

Examples:

- a new audio captioning task;
- a new multimodal task with its own required files;
- a new benchmark family whose reports cannot be represented by existing task
  inputs.

Required work:

- Add `src/sure_eval/evaluation/tasks/<task>/`.
- Add `manifest.yaml`, `routes.yaml`, `pipeline.py`, and `README.md`.
- Define required and optional roles in the task manifest.
- Define row format, alignment key, aggregation policy, and main report score.
- Add reusable nodes under `nodes/<stage>/<name>/` only when existing nodes are
  not sufficient.
- Add `node_env.yaml` for heavyweight dependencies, external binaries, model
  downloads, or checkpoints.
- Wire `sure_eval.evaluation.scripts.describe_pipeline(...)` and
  `run_task(...)` only after the task route is stable.
- Update docs:
  - `docs/tasks/<task>.md`
  - `docs/tasks/README.md`
  - `README.md`
  - `scripts/generate_pipeline_catalog.py`
  - regenerate `docs/pipeline_catalog.jsonl`

Minimum tests:

- task pipeline tests;
- script-level describe/run tests;
- input contract tests;
- env check tests for node-local environments;
- fixture-based report shape tests.

## Category 2: New Metric Family For An Existing Task

Use this when the reported score definition changes. This is a real new metric,
not just a new route.

Examples:

- adding a semantic ASR score that is not CER/WER/MER;
- adding a new MT quality score for S2TT;
- adding a new classification-style aggregate with a different denominator or
  decision rule.

Required work:

- Add or reuse a scoring node under `nodes/scoring/<backend>/`.
- Add the metric family to `tasks/<task>/manifest.yaml`.
- Add or update the metric input contract.
- Add a route in `tasks/<task>/routes.yaml`.
- Update `tasks/<task>/pipeline.py` dispatch and report details.
- Make the score key and aggregation policy explicit.
- Update task docs and the pipeline catalog.

Minimum tests:

- scorer correctness on small fixtures;
- aggregation edge cases, including empty reference and missing keys;
- route describe/run tests;
- script output tests for `report.json` and `pipeline_description.json`;
- regression tests against legacy behavior if replacing an old path.

Documentation rule:

- Put it in the metric family table as a new reported metric.
- State the mathematical definition and denominator.
- State whether higher or lower is better when this is not obvious.

## Category 3: New Route For An Existing Metric Family

Use this when CER remains CER, WER remains WER, MER remains MER, etc., but the
route changes. This usually means a different normalization, transcription, or
scoring backend chain.

Examples:

- `asr.zh.cer_canonical.canonical_itn.token_cer` for CER with canonical
  normalization;
- an explicit `wetext_norm` route for an existing ASR metric;
- an SCTK-backed WER route alongside an existing WER route.

Required work:

- Add a route entry in `tasks/<task>/routes.yaml`.
- Update `tasks/<task>/pipeline.py` to validate legal normalizer/scorer
  combinations.
- Preserve the default route unless the PR explicitly changes it and includes a
  compatibility justification.
- Ensure `EvaluationReport.metric` and route selector semantics are documented
  clearly.
- Regenerate `docs/pipeline_catalog.jsonl`.
- Update task docs with separate columns for reported metric and route
  selector when needed.

Minimum tests:

- default route remains unchanged;
- new route can be selected explicitly;
- illegal combinations are rejected with clear errors;
- `describe_pipeline(...)` returns the expected `pipeline_id` and node IDs;
- end-to-end route run produces the expected report trace.

Documentation rule:

- Do not call the route selector a new metric family.
- Use language such as "canonical-normalized CER route" or "alternate WER
  route".
- If a route changes comparability, document the comparability boundary.

Canonical ASR example:

- reported metric family: CER / WER / MER;
- route selectors: `cer_canonical`, `wer_canonical`, `mer_canonical`;
- new node: `normalization/canonical_itn`;
- new scorer route: `scoring/token_cer` or `scoring/token_mer`;
- scientific intent: change normalization and tokenization route, not create a
  new reported metric family.

## Category 4: Node, Tool, Or Version Change Inside A Route

Use this when the PR changes one stage of a route.

Examples:

- a new normalizer rules version;
- a new WavLM provider or checkpoint-backed speaker embedding implementation;
- a wrapper around an external scoring toolkit;
- a dependency or environment validation change for a node.

Required work:

- Update or add the node under `nodes/<stage>/<name>/`.
- Update `manifest.yaml` with `version`, `stage`, schemas, implementation, and
  internal stages.
- Update `node_env.yaml` for dependencies, binaries, checkpoints, or downloads.
- Keep runtime assets local. Never commit `.venv/`, `checkpoints/`, model
  weights, caches, generated reports, or private paths.
- Decide whether behavior is score-compatible:
  - compatible bugfix/performance/env changes can keep the same route;
  - score-affecting changes need a new node version, new node id, or explicit
    route selector, depending on the comparability impact.
- Update node README and task docs if user-visible behavior changes.

Minimum tests:

- node-level tests;
- route-level regression tests;
- env check tests;
- full inference or toolkit smoke tests when practical for heavyweight nodes;
- skipped-heavy-test policy when real inference cannot run in CI.

Documentation rule:

- State whether the change is expected to alter scores.
- State required runtime assets and environment variables.
- State model/checkpoint provenance without committing the asset.

## Naming And Trace Rules

- `task`: canonical task name, such as `ASR`, `TTS`, or `VC`.
- `metric`: CLI route selector used by `describe_pipeline` and `run_task`.
- reported metric family: the scientific score family, such as CER or WER.
- `pipeline_id`: should expose task, language, route selector, and meaningful
  node choices.
- `node_id`: should be stable and stage-qualified, such as
  `normalization/canonical_itn` or `scoring/wavlm_large_sim`.
- node trace: must include `node_id`, `version`, and meaningful internal
  stages.

When a selector differs from the reported metric family, document that
distinction in the task guide and in PR notes.

## Documentation Checklist

Update all applicable files:

- `docs/tasks/<task>.md`
- `src/sure_eval/evaluation/tasks/<task>/README.md`
- `src/sure_eval/evaluation/tasks/<task>/manifest.yaml`
- `src/sure_eval/evaluation/tasks/<task>/routes.yaml`
- `docs/pipeline_catalog.jsonl`
- `docs/pipeline_catalog.md` when schema semantics change
- `README.md` for user-visible task, metric, or route changes
- node README files for new or changed nodes

Regenerate the catalog after changing supported routes:

```bash
PYTHONPATH=src python scripts/generate_pipeline_catalog.py
```

## Validation Checklist

Run the narrowest meaningful checks plus any task-specific tests:

```bash
PYTHONPATH=src python -m pytest <focused tests> -q
python -m ruff check src tests scripts/generate_pipeline_catalog.py
git diff --check
git ls-files | rg '(\.venv/|checkpoints?/|\.pth$|\.pt$|\.ckpt$|\.onnx$|\.safetensors$|\.bin$)' || true
```

For heavyweight nodes, also run:

```bash
sure-eval env check --node <node-id>
sure-eval env setup --node <node-id> --dry-run
```

If the node depends on a real model path and the PR changes scoring logic, run
at least one real inference smoke test outside CI and record the backend,
checkpoint, input fixture, and score in the PR description.

## PR Description Checklist

Every PR should state:

- category from the decision tree;
- scientific intent;
- reported metric family versus route selector;
- default route impact;
- score comparability impact;
- new or changed nodes;
- required dependencies and runtime assets;
- docs updated;
- tests and smoke checks run.

## Anti-Patterns

- Calling a route selector a new metric family.
- Changing a default route without saying so.
- Hiding score-affecting behavior behind a patch-level node change.
- Committing `.venv/`, checkpoints, downloaded model files, caches, or reports.
- Adding metric logic directly to `scripts/` instead of task/node layers.
- Using broad string manipulation when the task has structured inputs.
- Wrapping an external toolkit but silently changing its upstream scoring
  behavior.
