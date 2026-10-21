# RIR Estimation and Spatial Audio

The RIR estimation task evaluates predicted room impulse responses against reference RIRs.
Rows use `prediction_rir` and `reference_rir` paths and align by `sample_id`.
The audio files may be WAV/FLAC or NumPy `.npy`/`.npz` arrays. NumPy arrays
must provide `sample_rate` in the row because they do not carry an embedded
rate. The files must have matching sample rates and channel counts; spatial
metrics require at least two channels. Each pair is truncated to the shorter
file before scoring. NPM searches one common lag within +/-256 samples;
acoustic metrics begin each channel near its own strongest impulse; spatial
metrics use the channel signals as provided. Rows must refer to the same
source, room, microphone positions, and channel order. Each metric averages
its per-sample scores with equal weight. Every `per_sample` entry records
reference-side and prediction-side values with the unit. Multi-channel
spatial entries include each channel-0-relative pair; ILD/IPD additionally
report the valid frequency-bin count and absolute-error sum used for the mean.

The evaluator has two channel branches. The single-channel branch accepts one
channel and computes the base NPM, RT30, RT60, DRR, C50, C80, and EDC metrics. The
multi-channel branch computes the same base metrics before adding spatial
metrics. IACC is restricted to an exact binaural pair. ITD, ILD, IPD, and DOA
accept two or more channels and report one pair for every `chN` relative to
`ch0`; the metric score is the mean over those pairs. Set `channel_layout` to
`mono`, `multi_channel`, or `binaural_lr` when the layout is known; `auto` is
used when it is omitted.

## Metrics

| Metric | Meaning | Direction |
|---|---|---|
| `rir_npm` | normalized projection misalignment in dB | lower is better |
| `rt30_mae` | measured 30 dB decay-time error in seconds | lower is better |
| `rt60_mae` | absolute RT60 error in seconds | lower is better |
| `rir_drr_mae` | direct-to-reverberant ratio error in dB | lower is better |
| `rir_c50_mae` | C50 error in dB | lower is better |
| `rir_c80_mae` | C80 error in dB | lower is better |
| `rir_edc_mae` | energy decay curve MAE above -40 dB in dB | lower is better |
| `rir_iacc_mae` | interaural cross-correlation coefficient error in a +/-1 ms window | lower is better |
| `rir_itd_mae` | interaural time difference error in ms in a +/-1 ms window | lower is better |
| `rir_ild_mae` | ch0-relative interchannel level error in dB | lower is better |
| `rir_ipd_mae` | ch0-relative interchannel phase error in radians | lower is better |
| `rir_doa_mae` | ch0-relative TDOA-derived DOA error in degrees | lower is better |

The default bundle is:

```text
rir.any.multi.rir_npm.rir_npm_v1__rt30_mae.rt30_mae_v1__rt60_mae.rt60_mae_v1__rir_drr_mae.rir_drr_mae_v1__rir_c50_mae.rir_c50_mae_v1__rir_c80_mae.rir_c80_mae_v1__rir_edc_mae.rir_edc_mae_v1
```

The bundle's top-level `score` is `rir_npm`; all metric scores are in
`details.results`. This is a reporting convention, not a combined score.

For cross-channel base evaluation, the curated bundle is:

```text
rir.any.multi.rir_npm.rir_npm_v1__rt30_mae.rt30_mae_v1__rt60_mae.rt60_mae_v1__rir_drr_mae.rir_drr_mae_v1__rir_c50_mae.rir_c50_mae_v1__rir_c80_mae.rir_c80_mae_v1__rir_edc_mae.rir_edc_mae_v1
```

It accepts the monaural and multi-channel branches because it contains only
the shared base metrics. The existing atomic routes remain available for
single-metric evaluation. Add the spatial bundle separately when the samples
meet the IACC two-channel contract.

For multi-channel spatial evaluation with more than two channels, use the
registered spatial bundle without IACC:

```text
rir.any.multi.rir_npm.rir_npm_v1__rt30_mae.rt30_mae_v1__rt60_mae.rt60_mae_v1__rir_drr_mae.rir_drr_mae_v1__rir_c50_mae.rir_c50_mae_v1__rir_c80_mae.rir_c80_mae_v1__rir_edc_mae.rir_edc_mae_v1__rir_itd_mae.rir_itd_mae_v1__rir_ild_mae.rir_ild_mae_v1__rir_ipd_mae.rir_ipd_mae_v1__rir_doa_mae.rir_doa_mae_v1
```

This bundle reports the remaining spatial metrics as channel-0-relative pairs;
IACC is intentionally absent because it is defined only for an exact binaural
pair.

For the multi-channel spatial chain, start with the same NPM, RT30, RT60, DRR,
C50, C80, and EDC base
metrics, then select the registered bundle containing
`rir_iacc_mae`, `rir_itd_mae`, `rir_ild_mae`, and `rir_ipd_mae` together with
the channel-wise metrics. Its top-level score is `rir_npm`; spatial
scores remain separate entries in `details.results`.

Example row:

```json
{"sample_id":"room001_utt001","prediction_rir":"pred.wav","reference_rir":"ref.wav","sample_rate":16000,"channel_layout":"mono"}
```

`rir_iacc_mae` requires exactly two channels. The other spatial metrics use
channel 0 as the reference channel and report `pair_count` plus per-channel
pair details. ITD uses a normalized cross-correlation search over +/-1 ms.
`rir_doa_mae` additionally requires `metadata.microphone_spacing_m`; for
channel `N` the default spacing is `N * microphone_spacing_m`, or an explicit
`channel_spacings_m` list can be supplied. Its optional `sound_speed_mps`
defaults to 343 m/s.

The scoring nodes run in-process with the package's `audio` extra. They do
not need model checkpoints or node-local environments. A model may generate
the predicted RIR in a separate environment such as `usddps-recrir`; scoring
only consumes the resulting audio files.

## CLI

```bash
sure-eval metric routes rir --metric npm --json
sure-eval metric routes rir --metric iacc --json
sure-eval metric describe rir --metric npm --output rir_pipeline.json --json
sure-eval metric run --pipeline rir_pipeline.json --samples-jsonl rir_samples.jsonl --output-dir /tmp/rir_eval --json
```
