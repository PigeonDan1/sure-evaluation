# TTS — Text-to-Speech

Evaluate synthesized speech for intelligibility, speaker similarity, and quality.

## Metrics

### Semantic error rate

| Canonical metric | Execution selector | Language | Pipeline ID | Nodes |
|:-----------------|:-------------------|:---------|:------------|:------|
| `cer` | `cer` or `tts_cer` | `zh` / `cmn` / `yue` | `tts.zh.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1.punctuation_strip_norm_v1.wenet_cer_v1` | `frontend/funasr_loader_16k_mono` → `transcription/paraformer_zh` → `normalization/punctuation_strip_norm` → `scoring/wenet_cer` |
| `wer` | `wer` or `tts_wer` | `en` | `tts.en.wer.whisper_large_v3_v1.whisper_norm_english_v1.wenet_wer_v1` | `transcription/whisper_large_v3` → `normalization/whisper_norm` → `scoring/wenet_wer` |

Mandarin semantic CER strips punctuation only before WeNet CER. It does not
use `normalization/aispeech_norm` by default, so numbers, case, and
non-punctuation text are preserved.

### Speaker similarity

| Canonical metric | Method selector | Scoring node | Pipeline ID |
|:-----------------|:----------------|:-------------|:------------|
| `spk_sim` | `spk_sim` or `sim/wavlm-large` | `scoring/wavlm_large_sim` | `tts.<language>.spk_sim.wavlm_large_sim_v1` |
| `spk_sim` | `sim/ecapa-tdnn` | `scoring/ecapa_tdnn_sim` | `tts.<language>.spk_sim.ecapa_tdnn_sim_v1` |
| `spk_sim` | `sim/eres2net` | `scoring/eres2net_sim` | `tts.<language>.spk_sim.eres2net_sim_v1` |

Compatibility reports from `TTSMetricPipeline` still expose legacy `sim/...`
keys, but `EvaluationReport.metric`, catalog rows, and pipeline IDs use
`spk_sim`.

### MOS (Mean Opinion Score)

| Canonical metric | Execution selector | Scoring node |
|:-----------------|:-------------------|:-------------|
| `dnsmos` | `dnsmos` | `scoring/dnsmos` |
| `wv_mos` | `wv_mos` or `wv-mos` | `scoring/wv_mos` |
| `utmos` | `utmos` | `scoring/utmos` |

Multi-metric selections are bundle IDs built from atomic member tails, such as
`tts.zh.multi.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1.punctuation_strip_norm_v1.wenet_cer_v1__spk_sim.wavlm_large_sim_v1__dnsmos.dnsmos_v1`; the selected atomic IDs are listed in `member_pipeline_ids`.

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
sure-eval env setup --task tts --language zh --metrics cer,dnsmos --dry-run

# Set up environments
sure-eval env setup --task tts --language zh --metrics cer,dnsmos

# Describe and run
sure-eval metric describe tts --language zh --metrics cer,dnsmos \
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
    metrics=("cer", "dnsmos"),
    output_dir="/tmp/tts_eval",
    device="cuda",
    cache_dir="/tmp/sure_eval_cache",
)
print(report.score)
```

## Output

- `report.json` — `score` for the first selected metric, canonical `details.results` keys, and per-sample details.
- `pipeline_description.json` — canonical `metric`, selected `pipeline_id`,
  `pipeline_kind`, `member_pipeline_ids`, `execution_metrics`,
  `computation_node_ids`, relative `task_config_path` / `route_config_path`,
  `script_entrypoint`, `executor`, and node versions.

## Environment Notes

- Semantic metrics need ASR transcription nodes (`transcription/paraformer_zh` or `transcription/whisper_large_v3`).
- Speaker similarity and MOS require their respective scoring nodes.
- Use `export SURE_EVAL_CACHE_DIR=/path/to/large/disk` for model checkpoints and caches.
