# SLU — Spoken Language Understanding

Evaluate prompt-based spoken language understanding answers with accuracy.

## Metrics

| Output mode | Metric | Pipeline ID | Nodes |
|:------------|:-------|:------------|:------|
| `choice_id` | `accuracy` | `slu.any.accuracy.prompt_norm_choice_id_v1.classify_v1` | `normalization/prompt_norm` → `scoring/classify` |
| `choice_text` | `accuracy` | `slu.any.accuracy.prompt_norm_choice_text_v1.classify_v1` | `normalization/prompt_norm` → `scoring/classify` |

## Input Format

Three aligned files:

### Reference answers (`--ref-file`)

```text
utt_001<TAB>A
utt_002<TAB>B
```

### Hypothesis answers (`--hyp-file`)

```text
utt_001<TAB>A. hello world
utt_002<TAB>B
```

### Prompt JSONL (`--prompt-jsonl`)

```jsonl
{"key":"utt_001","prompt":"A. hello world\nB. goodbye\nC. maybe"}
{"key":"utt_002","prompt":"A. yes\nB. no"}
```

The prompt normalization node supports arbitrary choice ids and counts. Legacy `A. option` text prompts remain supported as a fallback.

## CLI Usage

```bash
sure-eval metric describe slu --output /tmp/slu.json
sure-eval metric run --pipeline /tmp/slu.json \
  --ref-file ref.txt --hyp-file hyp.txt \
  --prompt-jsonl prompt.jsonl --output-dir /tmp/slu_eval
```

## Python API

```python
from sure_eval.evaluation.scripts import run_task

report = run_task(
    "slu",
    ref_file="ref.txt",
    hyp_file="hyp.txt",
    prompt_jsonl="prompt.jsonl",
    output_dir="/tmp/slu_eval",
)
print(report.score)
```

## Output

- `report.json` — `score` (accuracy), correct/total counts.
- `pipeline_description.json` — canonical `metric`, selected `pipeline_id`,
  `execution_metrics`, `computation_node_ids`, `output_mode`, relative
  `task_config_path` / `route_config_path`, `script_entrypoint`, `executor`,
  and node versions.

## Output Modes

- `choice_id` — compare normalized choice ids (`A`, `B`, ...).
- `choice_text` — compare normalized choice text (`hello world`, ...).
