# VAD — Voice Activity Detection

Evaluate voice activity detection outputs as speech time-segment detection on a
fixed seconds timebase.

## Metrics

| Canonical metric | Pipeline ID | Nodes | Higher is better |
|:-----------------|:------------|:------|:-----------------|
| `f1` | `vad.any.f1.vad_contract_v1.vad_timebase_strict_v1.vad_detection_duration_v1` | `validation/vad_contract` → `normalization/vad_timebase` → `scoring/vad_detection_duration` | yes |
| `p_fa` | `vad.any.p_fa.vad_contract_v1.vad_timebase_strict_v1.vad_detection_duration_v1` | `validation/vad_contract` → `normalization/vad_timebase` → `scoring/vad_detection_duration` | no |
| `p_miss` | `vad.any.p_miss.vad_contract_v1.vad_timebase_strict_v1.vad_detection_duration_v1` | `validation/vad_contract` → `normalization/vad_timebase` → `scoring/vad_detection_duration` | no |
| `dcf_nist` | `vad.any.dcf_nist.vad_contract_v1.vad_timebase_strict_v1.vad_detection_duration_v1` | `validation/vad_contract` → `normalization/vad_timebase` → `scoring/vad_detection_duration` | no |
| `auc_roc` | `vad.any.auc_roc.vad_contract_v1.vad_timebase_strict_v1.vad_auc_roc_v1` | `validation/vad_contract` → `normalization/vad_timebase` → `scoring/vad_auc_roc` | yes |

Default route:

`vad.any.f1.vad_contract_v1.vad_timebase_strict_v1.vad_detection_duration_v1`

## Input Contract

Required roles:

| Role | Required | Format |
|:-----|:---------|:-------|
| `reference_jsonl` | yes | one JSON object per line |
| `sample_output` | yes | one JSON object per line |

Reference rows require exactly `key`, `duration`, and `speech_segments`:

```json
{"key": "utt1", "duration": 2.465, "speech_segments": [{"start": 0.3, "end": 0.838}]}
```

Prediction rows require `key` and may include only `speech_segments` and
`frame_scores`. Duration-based routes require `speech_segments`; the AUC route
requires `frame_scores`:

```json
{"key": "utt1", "speech_segments": [{"start": 0.3, "end": 0.9}], "frame_scores": [{"start": 0.0, "end": 0.01, "score": 0.03}]}
```

Strict contract rules:

- Reference and prediction keys must align exactly.
- `speech_segments` intervals use `start` and `end` in seconds.
- `frame_scores` intervals use `start`, `end`, and `score`.
- Reference `duration` must be positive and defines the scored time region.
- All intervals must satisfy `0 <= start < end <= duration`.
- Speech intervals and frame-score intervals must not overlap within a row.
- Prediction score aliases such as `scores`, `probs`, and `speech_probabilities` are not accepted.
- Prediction-side duration metadata such as `audio_duration` is not accepted.
- Missing fields required by the selected route fail validation.

## Scoring

`validation/vad_contract` parses JSONL, rejects malformed fields, aligns keys,
and records metric availability per row.

`normalization/vad_timebase` applies the strict profile on the reference
`duration` timebase. Because the contract rejects invalid or out-of-range
intervals, normalization is only responsible for stable ordering and scored
duration summaries. The first VAD route version uses zero collar and zero
boundary exclusion.

`scoring/vad_detection_duration` pools duration counts over rows with
prediction `speech_segments`:

- `TP_sec = duration(reference intersection prediction)`
- `FP_sec = duration(prediction minus reference)`
- `FN_sec = duration(reference minus prediction)`
- `precision = TP / (TP + FP)`
- `recall = TP / (TP + FN)`
- `f1 = 2PR / (P + R)`
- `p_fa = FP / nonspeech_scored_sec`
- `p_miss = FN / speech_scored_sec`
- `dcf_nist = 0.25 * p_fa + 0.75 * p_miss`

`scoring/vad_auc_roc` samples frame centers at `frame_shift_sec=0.01`. Labels
come from reference speech segments, scores come from prediction `frame_scores`
covering each frame center, and uncovered frames do not participate. If all
participating frames have one label class, `auc_roc` is reported as `null` with
a skip reason. Predicted `speech_segments` are never used as a hard-label AUC
fallback. `frame_shift_sec` is route configuration recorded in report details
and `pipeline_description.json`; it is not part of the pipeline ID.

## CLI Usage

```bash
sure-eval metric describe vad --metric f1 --output /tmp/vad.json
sure-eval metric run --pipeline /tmp/vad.json \
  --reference-jsonl ref.jsonl --sample-output pred.jsonl \
  --output-dir /tmp/vad_eval
```

Select AUC explicitly:

```bash
sure-eval metric describe vad --metric auc_roc --output /tmp/vad_auc.json
sure-eval metric run --pipeline /tmp/vad_auc.json \
  --reference-jsonl ref.jsonl --sample-output pred.jsonl \
  --output-dir /tmp/vad_auc_eval
```

## Python API

```python
from sure_eval.evaluation.scripts import run_task

report = run_task(
    "vad",
    reference_jsonl="ref.jsonl",
    sample_output="pred.jsonl",
    metric="f1",
    output_dir="/tmp/vad_eval",
)
print(report.metric, report.score)
```

## Output

- `report.json` — `score` for the selected primary metric, `details.results`
  for all metrics produced by the selected scoring node, auxiliary duration or
  AUC counts, skipped metrics, and per-row details.
- `pipeline_description.json` — canonical `metric`, selected `pipeline_id`,
  `execution_metrics`, `computation_node_ids`, relative `task_config_path` /
  `route_config_path`, `script_entrypoint`, `executor`, and node versions.

## Environment Notes

All VAD nodes are in-process and require only the base Python environment. The
evaluator does not run VAD inference, load audio, or resample audio; prediction
generation must produce the documented seconds-timebase JSONL.
