# Add Evaluation Capabilities

Use this page to classify an evaluation PR before editing code. Then follow the
matching guide.

## Key Terms

- `metric`: canonical reported score, such as `cer`, `wer`, `spk_sim`,
  `dnsmos`, or `macro_recall`.
- `execution_metrics`: CLI/API selectors or method selectors used to run a
  route, such as `tts_cer`, `sim/wavlm-large`, or `macro-recall`.
- `pipeline_id`: exact computation identity in the form
  `task.language.metric.node_version...`.

Use `sure-eval metric routes <task> ...` to inspect existing identities before
creating a new declaration. A different implementation of an existing metric
is normally a new route, not a new metric.

Do not create a new metric name when only the route, backend, normalizer, or
tool changes.

## Decision Tree

1. New input/output problem, row format, alignment, or aggregation?
   Follow [New Task](./pr_guides/new_task.md).

2. Existing task, but a new reported score definition?
   Follow [New Metric](./pr_guides/new_metric.md).

3. Same reported metric, but different normalization, transcription, scorer,
   backend, or node chain?
   Follow [New Pipeline Route](./pr_guides/new_route.md).

4. One node/tool/environment/version changes inside a route?
   Follow [Node Change](./pr_guides/node_change.md).

5. No evaluation behavior change?
   Follow [Maintenance](./pr_guides/maintenance.md).

When unsure, choose the narrowest category that matches the scientific claim.

## Quick Examples

| Change | Category |
|:--|:--|
| Add VC-style task with new sample roles | New task |
| Add a new S2TT reported quality score | New metric |
| Add Qwen3-ASR route for existing TTS CER | New pipeline route |
| Add ECAPA provider for existing `spk_sim` family | New pipeline route or node change, depending on whether route identity changes |
| Update `node_env.yaml` for a model dependency | Node change |

## Required PR Statement

Every evaluation PR must state:

- category and guide followed;
- reported `metric`;
- `execution_metrics`, if different;
- added or changed `pipeline_id`;
- default route impact;
- score comparability impact;
- tests and smoke checks run.

The PR template asks for these fields explicitly.
