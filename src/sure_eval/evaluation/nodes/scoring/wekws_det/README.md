# WekWS DET KWS Scoring

## Purpose

`scoring/wekws_det` computes keyword spotting metrics using WekWS-style
threshold sweep semantics. It exposes both operating-point metrics and
DET-derived summary metrics.

## Task Scenarios

- KWS accuracy, precision, recall, F1, FRR, FAR, false alarms per hour, DET
  curve, and macro-recall.
- Routes that include
  `conversion_kws_sure_json_to_samples_v1.wekws_det_v1` when conversion from
  SURE JSONL sample format is needed.

## Input

- Schema: `kws_samples`.
- Required fields:
  - `key`: stable sample id.
  - `expected_detected`: whether the reference contains the target keyword.
  - `detected`: whether the model triggered.
  - `score` or `scores`: scalar score or frame scores used for thresholds.
- Optional fields:
  - `expected_keyword`
  - `predicted_keyword`
  - `duration` in seconds for false alarms per hour.

If `expected_keyword` is set and the model triggers a different keyword, the
sample is counted as `wrong_keyword`, which contributes to false rejects.

## Output

- Schema: `kws_metric_report`.
- Metrics:
  - `accuracy`
  - `precision`
  - `recall`
  - `macro-recall`
  - `f1`
  - `false_reject_rate`
  - `false_alarm_rate`
  - `false_alarm_per_hour`
  - `det_curve`
- Higher is better for accuracy/precision/recall/F1/macro-recall. Lower is
  better for false alarm and false reject metrics.

## Versioned Computation

- Node id: `scoring/wekws_det`.
- Version: `v1`.
- Internal stages:
  - `keyword_normalization`
  - `threshold_decision`
  - `det_curve`
  - `operating_point_summary`
- Macro-recall:

```text
max true_detect_rate(threshold)
where false_alarms(threshold) <= macro_recall_false_alarms
```

The default false-alarm-count budget is `0`.

## Runtime and Assets

- Runtime: optional node-local `uv` project.
- Python: `3.11`.
- GPU: false.
- No model checkpoint is needed for metric computation.

Node-local test command:

```bash
UV_CACHE_DIR=src/sure_eval/evaluation/nodes/scoring/wekws_det/.cache/uv \
UV_PROJECT_ENVIRONMENT=src/sure_eval/evaluation/nodes/scoring/wekws_det/.venv \
UV_LINK_MODE=copy \
PYTHONPATH=src uv run --project src/sure_eval/evaluation/nodes/scoring/wekws_det \
  python -m pytest tests/test_kws_metrics.py tests/test_kws_pipeline_nodes.py
```

## Source and References

- WeKWS: https://github.com/wenet-e2e/wekws
- Local metric implementation:
  `src/sure_eval/evaluation/nodes/scoring/wekws_det/metrics.py`

## Limitations

- DET summary depends on score calibration and threshold grid behavior.
- False alarms per hour requires negative sample duration to be meaningful.
