# WekWS DET KWS Scoring

This node wraps the KWS metric behavior previously exposed through
`sure_eval.evaluation.tasks.kws.KWSMetricPipeline`.

The node keeps WekWS-style threshold sweep semantics:

- positive samples below threshold count as false rejects;
- positive samples with the wrong keyword count as false rejects;
- negative samples above threshold count as false alarms;
- false alarms per hour use negative audio duration when available;
- `macro-recall` selects the maximum true detect rate under a fixed false-alarm-count budget.

## Input

The node receives aligned `KWSSample` objects. Required fields:

| Field | Meaning |
| --- | --- |
| `key` | stable sample id shared by reference and prediction |
| `expected_detected` | whether the reference contains the target keyword |
| `detected` | whether the model triggered |
| `score` or `scores` | scalar score or frame scores used for threshold decisions |

Optional but important fields:

| Field | Meaning |
| --- | --- |
| `expected_keyword` | target keyword for positive samples |
| `predicted_keyword` | keyword returned by the model |
| `duration` | audio duration in seconds, used for false alarms per hour |

If `expected_keyword` is set and the model triggers a different keyword, the
sample is counted as `wrong_keyword`, which contributes to false reject rate.

## Metrics

The node reports threshold metrics at the selected threshold plus DET-derived
metrics:

- `accuracy`
- `precision`
- `recall`
- `macro-recall`
- `f1`
- `false_reject_rate`
- `false_alarm_rate`
- `false_alarm_per_hour`
- `det_curve`

`macro-recall` is computed on the same DET threshold grid as `det_curve`:

```text
max true_detect_rate(threshold)
where false_alarms(threshold) <= macro_recall_false_alarms
```

The default false-alarm-count budget is `0`.

## Runtime

Use this scoring node as the uv project:

```bash
env \
  UV_CACHE_DIR=src/sure_eval/evaluation/nodes/scoring/wekws_det/.cache/uv \
  UV_PROJECT_ENVIRONMENT=src/sure_eval/evaluation/nodes/scoring/wekws_det/.venv \
  UV_LINK_MODE=copy \
  PYTHONPATH=src \
  uv run --project src/sure_eval/evaluation/nodes/scoring/wekws_det \
  python -m pytest tests/test_kws_metrics.py tests/test_kws_pipeline_nodes.py
```

Do not keep the persistent KWS metric cache in `/tmp`.
