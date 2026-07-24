# SD — Speaker Diarization

Evaluate speaker diarization outputs with Diarization Error Rate (DER).

## Metrics

| Metric | Pipeline ID | Nodes | Params |
|:-------|:------------|:------|:-------|
| `der` | `sd.any.der.meeteval_v1` | `scoring/meeteval` | `collar: 0.25` |

## Input Format

Reference and hypothesis annotation files in a MeetEval-supported format:

- RTTM
- STM
- CTM
- SegLST

Example RTTM line:

```text
SPEAKER meeting_001 1 12.34 2.50 <NA> <NA> speaker_A <NA>
```

## CLI Usage

```bash
sure-eval metric describe sd --metric der --output /tmp/sd.json
sure-eval metric run --pipeline /tmp/sd.json \
  --ref-file ref.rttm --hyp-file hyp.rttm --output-dir /tmp/sd_eval
```

## Python API

```python
from sure_eval.evaluation.scripts import run_task

report = run_task(
    "sd",
    ref_file="ref.rttm",
    hyp_file="hyp.rttm",
    metric="der",
    output_dir="/tmp/sd_eval",
)
print(report.score)
```

## Output

- `report.json` — `score` (DER), `num_sessions`, missed/false-alarm/speaker-error times.
- `pipeline_description.json` — canonical `metric`, selected `pipeline_id`,
  `execution_metrics`, `computation_node_ids`, relative `task_config_path` /
  `route_config_path`, `script_entrypoint`, `executor`, and node versions.

## Environment Notes

SD requires `meeteval`. Install with:

```bash
pip install "sure-evaluation[diarization]"
```

Or ensure the `scoring/meeteval` node environment is set up:

```bash
sure-eval env setup --node scoring/meeteval
```
