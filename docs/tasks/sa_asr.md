# SA-ASR — Speaker-Aware ASR

Evaluate multi-speaker ASR outputs. Reports cpWER as the main metric and DER as a companion metric.

## Metrics

| Metric | Pipeline ID | Nodes | Params |
|:-------|:------------|:------|:-------|
| `cpwer` (main) | `sa_asr.en.cpwer.conversion_sa_asr_cpwer_v1.gstar_norm_v1.meeteval_v1` | `conversion/sa_asr__cpwer` → `normalization/gstar_norm` → `scoring/meeteval` | `collar: 0.5` |
| `der` (companion) | same | same | recorded alongside cpWER |

## Input Format

STM six-field rows:

```text
session_id channel speaker start end <transcript>
```

Example:

```text
meeting_001 1 spk_A 12.34 14.80 hello world
```

The pipeline converts STM to key-text for normalization, normalizes with `gstar_norm`, then converts back to STM before MeetEval scoring.

## CLI Usage

```bash
sure-eval metric describe sa_asr --metric cpwer --output /tmp/sa_asr.json
sure-eval metric run --pipeline /tmp/sa_asr.json \
  --ref-file ref.stm --hyp-file hyp.stm --output-dir /tmp/sa_asr_eval
```

## Python API

```python
from sure_eval.evaluation.scripts import run_task

report = run_task(
    "sa_asr",
    ref_file="ref.stm",
    hyp_file="hyp.stm",
    metric="cpwer",
    output_dir="/tmp/sa_asr_eval",
)
print(report.score)  # cpWER
```

## Output

- `report.json` — `score` (cpWER), `der`, `num_sessions`.
- `pipeline_description.json` — canonical `metric`, selected `pipeline_id`,
  `execution_metrics`, `computation_node_ids` including conversion nodes,
  relative `task_config_path` / `route_config_path`, `script_entrypoint`,
  `executor`, conversion trace, and node versions.

## Environment Notes

SA-ASR requires `meeteval`. Install with:

```bash
pip install "sure-evaluation[diarization]"
```
