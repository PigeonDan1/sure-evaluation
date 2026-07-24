# TSE — Target Speaker Extraction

Evaluates target-speaker-extraction models that take mixed speech + speaker enrollment clue and output the extracted clean speech.

## Metrics

| Canonical metric | Execution selector | Pipeline ID | Node chain | Higher is better |
|:--------------|:---------------|:------------|:-----------|:-----------------|
| `si_sdr` | `si_sdr` | `tse.<language>.si_sdr.si_sdr_v1` | `scoring/si_sdr` | Yes |
| `spk_sim` | `spk_sim` or `sim/wavlm-large` | `tse.<language>.spk_sim.wavlm_large_sim_v1` | `scoring/wavlm_large_sim` | Yes |
| `spk_sim` | `sim/ecapa-tdnn` | `tse.<language>.spk_sim.ecapa_tdnn_sim_v1` | `scoring/ecapa_tdnn_sim` | Yes |
| `spk_sim` | `sim/eres2net` | `tse.<language>.spk_sim.eres2net_sim_v1` | `scoring/eres2net_sim` | Yes |
| `dnsmos` | `dnsmos` | `tse.<language>.dnsmos.dnsmos_v1` | `scoring/dnsmos` | Yes |
| `wv_mos` | `wv_mos` or `wv-mos` | `tse.<language>.wv_mos.wv_mos_v1` | `scoring/wv_mos` | Yes |
| `utmos` | `utmos` | `tse.<language>.utmos.utmos_v1` | `scoring/utmos` | Yes |
| `cer` | `cer` or `tse_cer` (zh) | `tse.zh.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1.punctuation_strip_norm_v1.wenet_cer_v1` | `frontend/funasr_loader_16k_mono → transcription/paraformer_zh → normalization/punctuation_strip_norm → scoring/wenet_cer` | No |
| `wer` | `wer` or `tse_wer` (en) | `tse.en.wer.whisper_large_v3_v1.whisper_norm_english_v1.wenet_wer_v1` | `transcription/whisper_large_v3 → normalization/whisper_norm → scoring/wenet_wer` | No |

> **SI-SDRi**: when a sample provides `mixed_audio`, the `si_sdr` route additionally reports
> `si_sdri` (`SI-SDR(prediction, clean) − SI-SDR(mixture, clean)`, higher is better) alongside
> `si_sdr` — no separate route is needed. Samples without `mixed_audio` report `si_sdr` only.

Multi-metric selections use bundle IDs built from atomic member tails, such as
`tse.zh.multi.si_sdr.si_sdr_v1__spk_sim.wavlm_large_sim_v1__dnsmos.dnsmos_v1`.

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
| `reference_text` | Required for `cer`/`wer` semantic evaluation | Reference transcript |
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
sure-eval metric describe tse --language zh --metrics si_sdr,spk_sim,dnsmos

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

Each run writes `report.json` and `pipeline_description.json` to the output
directory.

- `report.json` — `score` for the first selected metric, canonical
  `details.results` keys, and per-sample details.
- `pipeline_description.json` — canonical `metric`, selected `pipeline_id`,
  `pipeline_kind`, `member_pipeline_ids`, `execution_metrics`,
  `computation_node_ids`, relative `task_config_path` / `route_config_path`,
  `script_entrypoint`, `executor`, and node versions.
