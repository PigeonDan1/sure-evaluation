# S2TT — Speech-to-Text Translation

Score speech-to-text translation hypotheses against references.

## Metrics

| Metric | Languages | Pipeline ID | Nodes |
|:-------|:----------|:------------|:------|
| `bleu` | `zh`, `en` | `s2tt.<lang>.bleu.sacrebleu` | `scoring/sacrebleu` |
| `bleu_char` | `zh`, `en` | `s2tt.<lang>.bleu_char.sacrebleu` | `scoring/sacrebleu` |
| `chrf` | `zh`, `en` | `s2tt.<lang>.chrf.sacrebleu` | `scoring/sacrebleu` |
| `xcomet_xl` | `zh`, `en` | `s2tt.<lang>.xcomet_xl.xcomet_xl` | `scoring/xcomet_xl` |
| `bleurt_20` | `zh`, `en` | `s2tt.<lang>.bleurt_20.bleurt_20` | `scoring/bleurt_20` |

SacreBLEU is lightweight and included in the base install. XCOMET-XL and BLEURT-20 require optional node-local environments.

## Input Format

Tab-separated key-text files:

```text
utt_001<TAB>你好世界
```

| Role | Required for | Format |
|:-----|:-------------|:-------|
| `ref` | all metrics | `key<TAB>text` |
| `hyp` | all metrics | `key<TAB>text` |
| `src` | `xcomet_xl` | `key<TAB>text` |

## CLI Usage

### Lightweight metrics

```bash
sure-eval metric describe s2tt --language zh --metric bleu --output /tmp/s2tt.json
sure-eval metric run --pipeline /tmp/s2tt.json \
  --ref-file ref.txt --hyp-file hyp.txt --output-dir /tmp/s2tt_eval
```

### Heavy metrics

```bash
# Inspect environment requirements first
sure-eval env setup --task s2tt --language zh --metrics xcomet_xl --dry-run

# Set up the node
sure-eval env setup --task s2tt --language zh --metrics xcomet_xl

# Run
sure-eval metric describe s2tt --language zh --metric xcomet_xl --output /tmp/s2tt.json
sure-eval metric run --pipeline /tmp/s2tt.json \
  --ref-file ref.txt --hyp-file hyp.txt --src-file src.txt \
  --output-dir /tmp/s2tt_eval --validate-env
```

## Python API

```python
from sure_eval.evaluation.scripts import run_task

report = run_task(
    "s2tt",
    ref_file="ref.txt",
    hyp_file="hyp.txt",
    src_file="src.txt",
    language="zh",
    metric="bleu",
    output_dir="/tmp/s2tt_eval",
)
print(report.score)
```

## Output

- `report.json` — `score`, `bleu`, `bleu_char`, `chrf` (SacreBLEU), or segment-mean XCOMET/BLEURT scores.
- `pipeline_description.json` — selected route and node versions.

## Environment Notes

- `scoring/sacrebleu` runs in the base install.
- `scoring/xcomet_xl` needs `unbabel-comet`, PyTorch, and the `Unbabel/XCOMET-XL` checkpoint.
- `scoring/bleurt_20` needs TensorFlow 2.13 and the BLEURT-20 checkpoint.
