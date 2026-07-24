# VC — Voice Conversion

Evaluate converted speech for content preservation, speaker similarity, and quality.

## Metrics

### Semantic error rate

| Canonical metric | Execution selector | Language | Reference mode | Pipeline ID | Nodes |
|:-----------------|:-------------------|:---------|:---------------|:------------|:------|
| `cer` | `cer` or `vc_cer` | `zh` | text | `vc.zh.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1.punctuation_strip_norm_v1.wenet_cer_v1` | `frontend/funasr_loader_16k_mono` → `transcription/paraformer_zh` → `normalization/punctuation_strip_norm` → `scoring/wenet_cer` |
| `cer` | `cer` or `vc_cer` | `zh` | audio | `vc.zh.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1.funasr_loader_16k_mono_v1.paraformer_zh_v1.punctuation_strip_norm_v1.wenet_cer_v1` | transcribes both converted and reference audio, then applies `normalization/punctuation_strip_norm` → `scoring/wenet_cer` |
| `wer` | `wer` or `vc_wer` | `en` | text | `vc.en.wer.whisper_large_v3_v1.whisper_norm_english_v1.wenet_wer_v1` | `transcription/whisper_large_v3` → `normalization/whisper_norm` → `scoring/wenet_wer` |
| `wer` | `wer` or `vc_wer` | `en` | audio | `vc.en.wer.whisper_large_v3_v1.whisper_large_v3_v1.whisper_norm_english_v1.wenet_wer_v1` | transcribes both converted and reference audio |

When `reference_text` is absent, the pipeline transcribes `reference_audio` and uses that as the reference.
The audio-reference pipeline ID repeats the transcription node because both
converted and reference audio are transcribed.

### Speaker similarity

| Canonical metric | Method selector | Scoring node |
|:-----------------|:----------------|:-------------|
| `spk_sim` | `spk_sim` or `sim/wavlm-large` | `scoring/wavlm_large_sim` |
| `spk_sim` | `sim/ecapa-tdnn` | `scoring/ecapa_tdnn_sim` |
| `spk_sim` | `sim/eres2net` | `scoring/eres2net_sim` |

### MOS

| Canonical metric | Execution selector | Scoring node |
|:-----------------|:-------------------|:-------------|
| `dnsmos` | `dnsmos` | `scoring/dnsmos` |
| `wv_mos` | `wv_mos` or `wv-mos` | `scoring/wv_mos` |
| `utmos` | `utmos` | `scoring/utmos` |

Multi-metric selections are bundle IDs built from atomic member tails, such as
`vc.zh.multi.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1.punctuation_strip_norm_v1.wenet_cer_v1__spk_sim.wavlm_large_sim_v1__dnsmos.dnsmos_v1`, with atomic members in `member_pipeline_ids`.

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
sure-eval env setup --task vc --language zh --metrics cer,spk_sim --dry-run

# Set up environments
sure-eval env setup --task vc --language zh --metrics cer,spk_sim

# Describe and run
sure-eval metric describe vc --language zh --metrics cer,spk_sim \
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
    metrics=("cer", "spk_sim"),
    output_dir="/tmp/vc_eval",
    device="cuda",
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

- Semantic metrics need ASR transcription nodes.
- Speaker similarity and MOS require their respective scoring nodes.
- Use `export SURE_EVAL_CACHE_DIR=/path/to/large/disk` for model checkpoints and caches.
