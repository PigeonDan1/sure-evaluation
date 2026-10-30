# SE — Speech Enhancement

SE evaluates enhanced speech generated from noisy input audio.

```bash
sure-eval metric routes se --metric dnsmos
```

## Metrics

| Canonical metric | Execution selector | Pipeline ID | Nodes | Required roles |
| --- | --- | --- | --- | --- |
| `si_sdr` | `si_sdr` or `si-sdr` | `se.any.si_sdr.si_sdr_v1` | `scoring/si_sdr` | `enhanced_audio`, `reference_audio` |
| `stoi` | `stoi` | `se.any.stoi.stoi_v1` | `scoring/stoi` | `enhanced_audio`, `reference_audio` |
| `estoi` | `estoi` | `se.any.estoi.estoi_v1` | `scoring/estoi` | `enhanced_audio`, `reference_audio` |
| `lsd` | `lsd` | `se.any.lsd.lsd_v1` | `scoring/lsd` | `enhanced_audio`, `reference_audio` |
| `lps` | `lps` | `se.any.lps.lps_v1` | `scoring/lps` | `enhanced_audio`, `reference_audio` |
| `pesq` | `pesq` | `se.any.pesq.pesq_v1` | `scoring/pesq` | `enhanced_audio`, `reference_audio` |
| `dnsmos` | `dnsmos` | `se.any.dnsmos.dnsmos_v1` | `scoring/dnsmos` | `enhanced_audio` |
| `dnsmos_p808` | `dnsmos_p808` | `se.any.dnsmos_p808.dnsmos_p808_v1` | `scoring/dnsmos_p808` | `enhanced_audio` |
| `dnsmos_sig` | `dnsmos_sig` | `se.any.dnsmos_sig.dnsmos_sig_v1` | `scoring/dnsmos_sig` | `enhanced_audio` |
| `dnsmos_bak` | `dnsmos_bak` | `se.any.dnsmos_bak.dnsmos_bak_v1` | `scoring/dnsmos_bak` | `enhanced_audio` |
| `wv_mos` | `wv_mos` or `wv-mos` | `se.any.wv_mos.wv_mos_v1` | `scoring/wv_mos` | `enhanced_audio` |
| `utmos` | `utmos` | `se.any.utmos.utmos_v1` | `scoring/utmos` | `enhanced_audio` |
| `nsrmr` | `nsrmr` | `se.any.nsrmr.nsrmr_v1` | `scoring/nsrmr` | `enhanced_audio` |
| `squim_stoi` | `squim_stoi` | `se.any.squim_stoi.squim_stoi_v1` | `scoring/squim_stoi` | `enhanced_audio` |
| `squim_pesq` | `squim_pesq` | `se.any.squim_pesq.squim_pesq_v1` | `scoring/squim_pesq` | `enhanced_audio` |
| `squim_si_sdr` | `squim_si_sdr` | `se.any.squim_si_sdr.squim_si_sdr_v1` | `scoring/squim_si_sdr` | `enhanced_audio` |
| `nisqa_mos` | `nisqa_mos` | `se.any.nisqa_mos.nisqa_mos_v1` | `scoring/nisqa_mos` | `enhanced_audio` |
| `nisqa_noi` | `nisqa_noi` | `se.any.nisqa_noi.nisqa_noi_v1` | `scoring/nisqa_noi` | `enhanced_audio` |
| `nisqa_dis` | `nisqa_dis` | `se.any.nisqa_dis.nisqa_dis_v1` | `scoring/nisqa_dis` | `enhanced_audio` |
| `nisqa_col` | `nisqa_col` | `se.any.nisqa_col.nisqa_col_v1` | `scoring/nisqa_col` | `enhanced_audio` |
| `nisqa_loud` | `nisqa_loud` | `se.any.nisqa_loud.nisqa_loud_v1` | `scoring/nisqa_loud` | `enhanced_audio` |

The default metrics and default bundle remain unchanged. Every new metric is an
opt-in atomic route. All scores aggregate by arithmetic mean of per-sample
scores. `lsd` (dB) is **lower-is-better**; all other
metrics are higher-is-better. The top-level `report.score` is the first selected
metric. Bundle reports preserve every selected canonical key in
`details.results` and every member pipeline ID.

`estoi` uses extended pystoi at 16 kHz. `lsd` uses the local OpenBench
512-point, 128-hop STFT definition: root mean square of all time-frequency
`20 log10` magnitude differences in dB. It is one LSD convention, not a
universal standard. `nsrmr` uses `srmrpy` fast normalized mode at 16 kHz and
is a dimensionless modulation-energy ratio, not dB. `lps` uses the pinned URGENT
2025 wav2vec2-espeak phoneme model and scores `1 - edit_distance /
reference_phoneme_length`; it can be negative.

Existing `dnsmos` remains the calibrated P.835-style **OVRL** score. New
`dnsmos_sig`, `dnsmos_bak`, and `dnsmos_p808` select the provider's calibrated
SIG, BAK, and P808_MOS outputs; they do not reroute or change old `dnsmos`
scores. These are learned predictions, not P.835/P.808 human listening tests.
`squim_*` are TorchAudio SQUIM **no-reference predictions** of the named
reference metrics; they cannot be mixed with `stoi`, `pesq`, or `si_sdr` as if
they used the same measurement definition. `nisqa_*` are the five DIM outputs
from the same NISQA v2 checkpoint, with one corpus mean per output.

POLQA/P.863 is **not** registered: its white paper is not a licensed scoring
implementation. Do not report a proxy as POLQA.
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
DNSMOS component routes reuse `scoring/dnsmos`'s ONNX runtime assets. New
lightweight nodes require `pystoi` (`estoi`), `librosa` (`lsd`), or
`srmrpy==1.0` (`nsrmr`). Model-backed nodes require manually staged assets:

| Family | Runtime asset | Environment variable | Material restriction |
| --- | --- | --- | --- |
| `lps` | Pinned `facebook/wav2vec2-lv-60-espeak-cv-ft` directory | `SURE_SE_LPS_MODEL_DIR` | `transformers==4.24.0` legacy weight-norm compatibility |
| `squim_*` | TorchAudio `squim_objective_dns2020.pth` | `SURE_SE_SQUIM_OBJECTIVE_CKPT` | Objective weights CC BY 4.0 |
| `nisqa_*` | NISQA DIM `nisqa.tar` | `SURE_SE_NISQA_DIM_CKPT` | Model weights CC BY-NC-SA 4.0; verify permitted use |

No checkpoint, local virtual environment, or cached result belongs in the PR.
All model-backed routes can be described without assets; real `run` requires
their declared environment and checkpoint. Consult the node READMEs and run
`sure-eval env setup --pipeline /tmp/se_pipeline.json --dry-run` before scoring.
