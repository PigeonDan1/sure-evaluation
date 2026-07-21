# TSE — Target Speaker Extraction

Evaluates target-speaker-extraction models that take mixed speech + speaker enrollment clue and output the extracted clean speech.

## Metrics

| Metric family | Route selector | Node chain | Language-sensitive | Higher is better |
|:--------------|:---------------|:-----------|:-------------------|:-----------------|
| Signal quality | `si_sdr` | `scoring/si_sdr` | No | Yes |
| Speaker similarity | `sim/wavlm-large` | `scoring/wavlm_large_sim` | No | Yes |
| Speaker similarity | `sim/ecapa-tdnn` | `scoring/ecapa_tdnn_sim` | No | Yes |
| Speaker similarity | `sim/eres2net` | `scoring/eres2net_sim` | No | Yes |
| MOS | `dnsmos` | `scoring/dnsmos` | No | Yes |
| MOS | `wv-mos` | `scoring/wv_mos` | No | Yes |
| MOS | `utmos` | `scoring/utmos` | No | Yes |
| Semantic error rate | `tse_cer` (zh) | `frontend/funasr_loader_16k_mono → transcription/paraformer_zh → normalization/punctuation_strip_norm → scoring/wenet_cer` | Yes (zh→CER) | No |
| Semantic error rate | `tse_wer` (en) | `transcription/whisper_large_v3 → normalization/whisper_norm → scoring/wenet_wer` | Yes (en→WER) | No |

> **SI-SDRi**: when a sample provides `mixed_audio`, the `si_sdr` route additionally reports
> `si_sdri` (`SI-SDR(prediction, clean) − SI-SDR(mixture, clean)`, higher is better) alongside
> `si_sdr` — no separate route is needed. Samples without `mixed_audio` report `si_sdr` only.

## Input Format

TSE uses a JSONL file (one JSON object per line):

| Field | Required | Description |
|:------|:---------|:------------|
| `sample_id` | Yes | Unique sample identifier |
| `prediction_audio` | Yes | Path to the extracted audio (model output) |
| `reference_audio` | Yes | Path to the clean reference audio |
| `language` | Yes | Language code (e.g. `zh`, `en`) |
| `mixed_audio` | No | Path to the mixed speech input |
| `enrollment_audio` | No | Path to the speaker enrollment/clue audio |
| `reference_text` | Required for `tse_cer`/`tse_wer` | Reference transcript for semantic evaluation |
| `metadata` | No | Arbitrary metadata object |

Paths may be absolute or relative to the JSONL file location.

Example JSONL:

```jsonl
{"sample_id": "utt1", "prediction_audio": "pred1.wav", "reference_audio": "ref1.wav", "mixed_audio": "mix1.wav", "enrollment_audio": "enroll1.wav", "language": "zh"}
{"sample_id": "utt2", "prediction_audio": "pred2.wav", "reference_audio": "ref2.wav", "language": "en", "reference_text": "hello world"}
```

## CLI

```bash
# Describe a single-metric pipeline
sure-eval metric describe tse --language zh --metric si_sdr

# Describe a multi-metric pipeline
sure-eval metric describe tse --language zh --metrics si_sdr,sim/wavlm-large,dnsmos

# Run a pipeline
sure-eval metric run --pipeline /tmp/tse.json \
  --samples-jsonl samples.jsonl --output-dir /tmp/tse_eval
```

## Python API

```python
from sure_eval.evaluation.scripts import describe_pipeline, run_task

# Inspect the route
desc = describe_pipeline("tse", language="zh", metrics="si_sdr")
print(desc.node_ids)
# ('scoring/si_sdr',)

# Run and get a report
report = run_task(
    "tse",
    samples=samples,
    output_dir="/tmp/tse_eval",
    metrics=("si_sdr",),
)
print(report.score)
```

## Output

Each run writes `report.json` and `pipeline_description.json` to the output directory.