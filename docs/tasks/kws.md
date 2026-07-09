# KWS — Keyword Spotting

Evaluate keyword spotting outputs with accuracy, precision, recall, F1, and false-alarm/reject rates.

## Metrics

| Primary metric | Pipeline ID | Nodes |
|:---------------|:------------|:------|
| `accuracy` | `kws.<input_mode>.accuracy.wekws_det` | `scoring/wekws_det` |

Reported metrics:

- `accuracy`
- `precision`
- `recall`
- `f1`
- `false_reject_rate`
- `false_alarm_rate`
- `false_alarm_per_hour`
- `det_curve`

## Input Modes

| Mode | Required roles | Description |
|:-----|:---------------|:------------|
| `sure_json` | `reference_jsonl`, `sample_output` | SURE-style JSONL with `detected`, `keyword`, `score` |
| `wekws_score_ctc` | `wekws_label_file`, `wekws_score_file`, `keyword` | WeKWS CTC score output |
| `wekws_frame_score` | `wekws_label_file`, `wekws_frame_score_file`, `keyword` | WeKWS frame score output |

All modes are converted to aligned `KWSSample` rows before scoring.

## CLI Usage

### SURE JSON mode

```bash
sure-eval metric describe kws --metric accuracy --output /tmp/kws.json
sure-eval metric run --pipeline /tmp/kws.json \
  --reference-jsonl ref.jsonl --sample-output pred.jsonl \
  --output-dir /tmp/kws_eval
```

### WeKWS CTC score mode

```bash
sure-eval metric describe kws --metric accuracy --output /tmp/kws.json
sure-eval metric run --pipeline /tmp/kws.json \
  --wekws-label-file labels.txt --wekws-score-file scores.txt --keyword "hello" \
  --output-dir /tmp/kws_eval
```

## Python API

```python
from sure_eval.evaluation.scripts import run_task

report = run_task(
    "kws",
    reference_jsonl="ref.jsonl",
    sample_output="pred.jsonl",
    metric="accuracy",
    output_dir="/tmp/kws_eval",
)
print(report.score)
```

## Output

- `report.json` — `score` (accuracy), plus precision, recall, F1, FRR, FAR, and per-row details.
- `pipeline_description.json` — selected route, `input_mode`, node versions.

## Environment Notes

`scoring/wekws_det` runs in a node-local environment by default. Set it up with:

```bash
sure-eval env setup --node scoring/wekws_det
```
