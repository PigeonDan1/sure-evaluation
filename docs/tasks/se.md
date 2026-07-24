# SE — Speech Enhancement

SE evaluates enhanced speech generated from noisy input audio.

## Metrics

| Canonical metric | Execution selector | Pipeline ID | Nodes | Required roles |
| --- | --- | --- | --- | --- |
| `si_sdr` | `si_sdr` or `si-sdr` | `se.any.si_sdr.si_sdr_v1` | `scoring/si_sdr` | `enhanced_audio`, `reference_audio` |
| `stoi` | `stoi` | `se.any.stoi.stoi_v1` | `scoring/stoi` | `enhanced_audio`, `reference_audio` |
| `pesq` | `pesq` | `se.any.pesq.pesq_v1` | `scoring/pesq` | `enhanced_audio`, `reference_audio` |
| `dnsmos` | `dnsmos` | `se.any.dnsmos.dnsmos_v1` | `scoring/dnsmos` | `enhanced_audio` |
| `wv_mos` | `wv_mos` or `wv-mos` | `se.any.wv_mos.wv_mos_v1` | `scoring/wv_mos` | `enhanced_audio` |
| `utmos` | `utmos` | `se.any.utmos.utmos_v1` | `scoring/utmos` | `enhanced_audio` |

All SE metrics are higher-is-better and aggregate by mean over samples.
`noisy_audio` is recorded when available and is optional for metric scoring.
Multi-metric selections use bundle IDs such as
`se.any.multi.si_sdr.si_sdr_v1__stoi.stoi_v1__pesq.pesq_v1__dnsmos.dnsmos_v1`.

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

## Output

- `report.json` — `score` for the first selected metric, canonical
  `details.results` keys, and per-sample details.
- `pipeline_description.json` — canonical `metric`, selected `pipeline_id`,
  `pipeline_kind`, `member_pipeline_ids`, `execution_metrics`,
  `computation_node_ids`, relative `task_config_path` / `route_config_path`,
  `script_entrypoint`, `executor`, and node versions.

## Environment

`scoring/si_sdr` runs in process with NumPy. `scoring/stoi` and `scoring/pesq`
use optional pip packages:

```bash
python -m pip install ".[se]"
```

MOS nodes use the existing node-local environments:
`scoring/dnsmos`, `scoring/wv_mos`, and `scoring/utmos`.
