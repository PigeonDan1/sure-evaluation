# TTS — Text-to-Speech

Evaluate synthesized speech for intelligibility, speaker similarity, and quality.

## Metrics

### Semantic error rate

| Metric | Language | Pipeline ID | Nodes |
|:-------|:---------|:------------|:------|
| `tts_cer` | `zh` / `cmn` / `yue` | `tts.zh.tts_cer.funasr_loader_16k_mono.paraformer_zh.punctuation_strip_norm.wenet_cer` | `frontend/funasr_loader_16k_mono` → `transcription/paraformer_zh` → `normalization/punctuation_strip_norm` → `scoring/wenet_cer` |
| `tts_wer` | `en` | `tts.en.tts_wer.whisper_large_v3.whisper_norm.wenet_wer` | `transcription/whisper_large_v3` → `normalization/whisper_norm` → `scoring/wenet_wer` |

Mandarin semantic CER strips punctuation only before WeNet CER. It does not
use `normalization/aispeech_norm` by default, so numbers, case, and
non-punctuation text are preserved.

### Speaker similarity

| Metric | Scoring node |
|:-------|:-------------|
| `sim/wavlm-large` | `scoring/wavlm_large_sim` |
| `sim/ecapa-tdnn` | `scoring/ecapa_tdnn_sim` |
| `sim/eres2net` | `scoring/eres2net_sim` |

Report-level `sim` is an aggregate over selected speaker backends.

### MOS (Mean Opinion Score)

| Metric | Scoring node |
|:-------|:-------------|
| `dnsmos` | `scoring/dnsmos` |
| `wv-mos` | `scoring/wv_mos` |
| `utmos` | `scoring/utmos` |

## Input Format

JSONL sample manifest. Each row is one sample:

```json
{
  "sample_id": "tts_001",
  "prediction_audio": "out.wav",
  "reference_text": "你好世界",
  "reference_audio": "speaker.wav",
  "language": "zh"
}
```

| Field | Required for | Description |
|:------|:-------------|:------------|
| `sample_id` | all | Unique sample identifier |
| `prediction_audio` | all | Path to synthesized audio |
| `reference_text` | semantic metrics | Ground-truth text |
| `reference_audio` | speaker / MOS metrics | Reference speaker audio |
| `language` | semantic metrics | `zh`, `en`, etc. |

## CLI Usage

```bash
# Inspect required environments
sure-eval env setup --task tts --language zh --metrics tts_cer,dnsmos --dry-run

# Set up environments
sure-eval env setup --task tts --language zh --metrics tts_cer,dnsmos

# Describe and run
sure-eval metric describe tts --language zh --metrics tts_cer,dnsmos \
  --output /tmp/tts.json
sure-eval metric run --pipeline /tmp/tts.json \
  --samples-jsonl samples.jsonl \
  --output-dir /tmp/tts_eval \
  --device cuda \
  --cache-dir /tmp/sure_eval_cache \
  --validate-env
```

## Python API

```python
from sure_eval.evaluation.scripts import run_task

report = run_task(
    "tts",
    samples_jsonl="samples.jsonl",
    language="zh",
    metrics="tts_cer,dnsmos",
    output_dir="/tmp/tts_eval",
    device="cuda",
    cache_dir="/tmp/sure_eval_cache",
)
print(report.score)
```

## Output

- `report.json` — `score` (primary semantic metric), plus `sim`, MOS scores, and per-sample details.
- `pipeline_description.json` — selected route and node versions.

## Environment Notes

- Semantic metrics need ASR transcription nodes (`transcription/paraformer_zh` or `transcription/whisper_large_v3`).
- Speaker similarity and MOS require their respective scoring nodes.
- Use `export SURE_EVAL_CACHE_DIR=/path/to/large/disk` for model checkpoints and caches.
