# KWS — Keyword Spotting

Evaluate keyword spotting outputs with accuracy, macro_recall, precision, recall, F1, and false-alarm/reject rates.

## Metrics

| Primary metric | Execution selector | Pipeline ID | Nodes |
|:---------------|:-----------------|:------------|:------|
| `accuracy` | `accuracy` | `kws.any.accuracy.conversion_kws_sure_json_to_samples_v1.wekws_det_v1` | `conversion/kws_sure_json_to_samples` → `scoring/wekws_det` |
| `macro_recall` | `macro-recall` or `macro_recall` | `kws.any.macro_recall.conversion_kws_sure_json_to_samples_v1.wekws_det_v1` | `conversion/kws_sure_json_to_samples` → `scoring/wekws_det` |

Each file input mode has its own conversion node in `pipeline_id`, followed by
the shared `scoring/wekws_det` node. The direct in-memory sample path uses
`kws.any.<metric>.wekws_det_v1`.

Reported metrics:

- `accuracy`
- `precision`
- `recall`
- `macro_recall` (`macro-recall` in node-level DET details)
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

For recall under a fixed false-alarm-count budget, select `macro_recall`.
The compatibility selector `macro-recall` is also accepted:

```bash
sure-eval metric describe kws --metric macro_recall --output /tmp/kws.json
sure-eval metric run --pipeline /tmp/kws.json \
  --reference-jsonl ref.jsonl --sample-output pred.jsonl \
  --macro-recall-false-alarms 0 \
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
    metric="macro_recall",
    output_dir="/tmp/kws_eval",
)
print(report.score)
```

## Output

- `report.json` — `score` for the selected primary metric, plus precision, recall, macro_recall, F1, FRR, FAR, DET curve, and per-row details.
- `macro_recall` is computed from the node-level `macro-recall` DET result as the maximum true detect rate with `false_alarms <= macro_recall_false_alarms` (default `0`).
- `pipeline_description.json` — canonical `metric`, selected `pipeline_id`,
  `execution_metrics`, `computation_node_ids` including conversion nodes,
  input mode, relative `task_config_path` / `route_config_path`,
  `script_entrypoint`, `executor`, and node versions.

## Environment Notes

`scoring/wekws_det` runs in a node-local environment by default. Set it up with:

```bash
sure-eval env setup --node scoring/wekws_det
```
