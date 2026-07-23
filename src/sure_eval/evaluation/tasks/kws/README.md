# KWS Evaluation Task

KWS uses a task route plus a WekWS-style scoring node:

```text
sure_eval.evaluation.tasks.kws.pipeline.evaluate_kws_samples
sure_eval.evaluation.tasks.kws.pipeline.evaluate_kws_files
```

The scoring backend is:

```text
src/sure_eval/evaluation/nodes/scoring/wekws_det
```

`sure_eval.evaluation.tasks.kws` exposes the task route, loader helpers, and
the runner-compatible `KWSMetricPipeline` wrapper used by
`scripts/run_kws_metric_pipeline.py`.

## Input Contracts

Supported input modes:

| `input_mode` | Required roles | Notes |
| --- | --- | --- |
| `sure_json` | `reference_jsonl`, `sample_output` | wrapper outputs `detected`, `keyword`, and `score` |
| `wekws_score_ctc` | `wekws_label_file`, `wekws_score_file`, `keyword` | accepts `score_ctc.py` output |
| `wekws_frame_score` | `wekws_label_file`, `wekws_frame_score_file`, `keyword` | uses max frame score as utterance score |

All input modes are converted to aligned `KWSSample` rows before scoring.
`evaluate_kws_files()` records the selected `input_mode`, `input_contract`, and
`input_files` in the returned `EvaluationReport.details`.

`scripts/run_kws_metric_pipeline.py` calls `evaluate_kws_files()` directly and
keeps the legacy JSON fields:

```text
ok
input_mode
metrics
rows
summary
```

It also records the task-route metadata:

```text
pipeline_id
input_contract
input_files
pipeline_trace
```

## Metrics

The route reports `accuracy` by default and can also use `macro-recall` as a primary score. Reported metrics include:

- `precision`
- `recall`
- `macro-recall`
- `f1`
- `false_reject_rate`
- `false_alarm_rate`
- `false_alarm_per_hour`
- `det_curve`

The pipeline id records the scoring version:

```text
kws.default.accuracy.wekws_det
kws.default.macro-recall.wekws_det
```

`macro-recall` is the maximum true detect rate on the DET threshold grid with
`false_alarms <= macro_recall_false_alarms` (default `0`).
