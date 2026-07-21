# SE — Speech Enhancement

SE evaluates enhanced speech generated from noisy input audio.

## Metrics

| Metric | Family | Pipeline ID | Nodes | Required roles |
| --- | --- | --- | --- | --- |
| `si-sdr` | full-reference quality | `se.si_sdr.si_sdr` | `scoring/si_sdr` | `enhanced_audio`, `reference_audio` |
| `stoi` | full-reference intelligibility | `se.stoi.stoi` | `scoring/stoi` | `enhanced_audio`, `reference_audio` |
| `pesq` | full-reference perceptual quality | `se.pesq.pesq` | `scoring/pesq` | `enhanced_audio`, `reference_audio` |
| `dnsmos` | no-reference quality | `se.dnsmos.dnsmos` | `scoring/dnsmos` | `enhanced_audio` |
| `wv-mos` | no-reference quality | `se.wv_mos.wv_mos` | `scoring/wv_mos` | `enhanced_audio` |
| `utmos` | no-reference quality | `se.utmos.utmos` | `scoring/utmos` | `enhanced_audio` |

All SE metrics are higher-is-better and aggregate by mean over samples.
`noisy_audio` is recorded when available and is optional for metric scoring.

## Samples JSONL

```json
{"sample_id":"se_001","enhanced_audio":"enhanced.wav","noisy_audio":"noisy.wav","reference_audio":"clean.wav"}
```

Relative paths are resolved against the JSONL file location.

## CLI

```bash
sure-eval metric describe se --metrics si-sdr,stoi,pesq,dnsmos \
  --output /tmp/se_pipeline.json

sure-eval metric run --pipeline /tmp/se_pipeline.json \
  --samples-jsonl /path/to/se_samples.jsonl \
  --output-dir /tmp/se_eval
```

For a one-sample smoke test:

```bash
python scripts/run_se_metric_pipeline.py \
  --enhanced-audio enhanced.wav \
  --noisy-audio noisy.wav \
  --reference-audio clean.wav \
  --metrics si-sdr,stoi,pesq,dnsmos \
  --stub
```

## Environment

`scoring/si_sdr` runs in process with NumPy. `scoring/stoi` and `scoring/pesq`
use optional pip packages:

```bash
python -m pip install ".[se]"
```

MOS nodes use the existing node-local environments:
`scoring/dnsmos`, `scoring/wv_mos`, and `scoring/utmos`.
