# ASR — Automatic Speech Recognition

Evaluate text transcripts against references.

## Metrics

| Metric | Language | Pipeline ID | Nodes |
|:-------|:---------|:------------|:------|
| `cer` | `zh` | `asr.zh.cer.aispeech_norm.wenet_cer` | `normalization/aispeech_norm` → `scoring/wenet_cer` |
| `wer` | `en` (Whisper norm) | `asr.en.wer.whisper_norm.wenet_wer` | `normalization/whisper_norm` → `scoring/wenet_wer` |
| `wer` | `en` (legacy AISpeech norm) | `asr.en.wer.aispeech_norm.wenet_wer` | `normalization/aispeech_norm` → `scoring/wenet_wer` |
| `mer` | `cs` (code-switching) | `asr.cs.mer.aispeech_norm.wenet_mer` | `normalization/aispeech_norm` → `scoring/wenet_mer` |

## Input Format

Tab-separated key-text files:

```text
utt_001<TAB>你好世界
utt_002<TAB>今天天气不错
```

Both `--ref-file` and `--hyp-file` use the same format, aligned by key.

## CLI Usage

```bash
# Describe the route
sure-eval metric describe asr --language zh --metric cer --output /tmp/asr.json

# Run
sure-eval metric run --pipeline /tmp/asr.json \
  --ref-file ref.txt --hyp-file hyp.txt --output-dir /tmp/asr_eval
```

## Python API

```python
from sure_eval.evaluation.scripts import run_task

report = run_task(
    "asr",
    ref_file="ref.txt",
    hyp_file="hyp.txt",
    language="zh",
    metric="cer",
    output_dir="/tmp/asr_eval",
)
print(report.score)  # CER
```

## Output

- `report.json` — `score`, `cer` or `wer`, edit counts (`all`, `cor`, `sub`, `ins`, `del`).
- `pipeline_description.json` — selected route and node versions.

## Optional Tools

- `normalization/wetext_norm` — select with `normalizer="wetext:zh_tn"` or `normalizer="wetext:en_itn"`.
- `scoring/sctk_sclite` — select with `scorer="sctk_sclite"`.

These are not part of the default routes and must be requested explicitly.
