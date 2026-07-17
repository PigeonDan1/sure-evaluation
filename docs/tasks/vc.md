# VC — Voice Conversion

Evaluate converted speech for content preservation, speaker similarity, and quality.

## Metrics

### Semantic error rate

| Metric | Language | Reference mode | Pipeline ID | Nodes |
|:-------|:---------|:---------------|:------------|:------|
| `vc_cer` | `zh` | text | `vc.zh.vc_cer.funasr_loader_16k_mono.paraformer_zh.punctuation_strip_norm.wenet_cer` | `frontend/funasr_loader_16k_mono` → `transcription/paraformer_zh` → `normalization/punctuation_strip_norm` → `scoring/wenet_cer` |
| `vc_cer` | `zh` | audio | `vc.zh.vc_cer.funasr_loader_16k_mono.paraformer_zh.punctuation_strip_norm.wenet_cer` | transcribes both converted and reference audio, then applies `normalization/punctuation_strip_norm` → `scoring/wenet_cer` |
| `vc_wer` | `en` | text | `vc.en.vc_wer.whisper_large_v3.whisper_norm.wenet_wer` | `transcription/whisper_large_v3` → `normalization/whisper_norm` → `scoring/wenet_wer` |
| `vc_wer` | `en` | audio | `vc.en.vc_wer.whisper_large_v3.whisper_norm.wenet_wer` | transcribes both converted and reference audio |

When `reference_text` is absent, the pipeline transcribes `reference_audio` and uses that as the reference.

### Speaker similarity

| Metric | Scoring node |
|:-------|:-------------|
| `sim/wavlm-large` | `scoring/wavlm_large_sim` |
| `sim/ecapa-tdnn` | `scoring/ecapa_tdnn_sim` |
| `sim/eres2net` | `scoring/eres2net_sim` |

### MOS

| Metric | Scoring node |
|:-------|:-------------|
| `dnsmos` | `scoring/dnsmos` |
| `wv-mos` | `scoring/wv_mos` |
| `utmos` | `scoring/utmos` |

## Input Format

JSONL sample manifest:

```json
{
  "sample_id": "vc_001",
  "converted_audio": "converted.wav",
  "source_audio": "source.wav",
  "reference_audio": "speaker.wav",
  "reference_text": "你好世界",
  "language": "zh"
}
```

| Field | Required for | Description |
|:------|:-------------|:------------|
| `sample_id` | all | Unique sample identifier |
| `converted_audio` | all | Path to converted audio |
| `reference_text` | semantic (text mode) | Ground-truth text |
| `reference_audio` | semantic (audio mode), speaker, MOS | Reference audio |
| `source_audio` | optional | Original source audio |
| `language` | semantic metrics | `zh`, `en`, etc. |

## CLI Usage

```bash
# Inspect required environments
sure-eval env setup --task vc --language zh --metrics vc_cer,sim/wavlm-large --dry-run

# Set up environments
sure-eval env setup --task vc --language zh --metrics vc_cer,sim/wavlm-large

# Describe and run
sure-eval metric describe vc --language zh --metrics vc_cer,sim/wavlm-large \
  --output /tmp/vc.json
sure-eval metric run --pipeline /tmp/vc.json \
  --samples-jsonl samples.jsonl \
  --output-dir /tmp/vc_eval \
  --device cuda \
  --validate-env
```

## Python API

```python
from sure_eval.evaluation.scripts import run_task

report = run_task(
    "vc",
    samples_jsonl="samples.jsonl",
    language="zh",
    metrics="vc_cer,sim/wavlm-large",
    output_dir="/tmp/vc_eval",
    device="cuda",
)
print(report.score)
```

## Output

- `report.json` — `score` (primary semantic metric), plus `sim`, MOS scores, and per-sample details.
- `pipeline_description.json` — selected route and node versions.

## Environment Notes

- Semantic metrics need ASR transcription nodes.
- Speaker similarity and MOS require their respective scoring nodes.
- Use `export SURE_EVAL_CACHE_DIR=/path/to/large/disk` for model checkpoints and caches.
