# RIR Estimation and Spatial Audio

The RIR task evaluates predicted room impulse responses against reference RIRs.
Rows use `prediction_rir` and `reference_rir` paths and align by `sample_id`.
The audio files may be WAV/FLAC or NumPy `.npy`/`.npz` arrays. NumPy arrays
must provide `sample_rate` in the row because they do not carry an embedded
rate. The files must have matching sample rates and channel counts; spatial
metrics require at least two channels. Each pair is truncated to the shorter
file before scoring; no peak or time-shift alignment is applied. Rows must
already refer to the same source, room, microphone positions, channel order,
and time origin. Each metric averages its per-sample scores with equal weight.

The evaluator has two channel branches. The single-channel branch accepts one
channel and computes the base NPM, RT60, DRR, C50, and C80 metrics. The
multi-channel branch computes the same base metrics before adding spatial
metrics. Spatial metrics use only a strict binaural pair (exactly two
channels); a row labelled `multi_channel` with three or more channels is
rejected for those metrics. Set `channel_layout` to `mono`, `multi_channel`, or
`binaural_lr` when the layout is known; `auto` is used when it is omitted.

## Metrics

| Metric | Meaning | Direction |
|---|---|---|
| `rir_npm` | normalized projection misalignment in dB | lower is better |
| `rt60_mae` | absolute RT60 error in seconds | lower is better |
| `rir_drr_mae` | direct-to-reverberant ratio error in dB | lower is better |
| `rir_c50_mae` | C50 error in dB | lower is better |
| `rir_c80_mae` | C80 error in dB | lower is better |
| `rir_iacc_mae` | interaural cross-correlation coefficient error in a +/-1 ms window | lower is better |
| `rir_itd_mae` | interaural time difference error in seconds in a +/-1 ms window | lower is better |
| `rir_ild_mae` | binaural ILD error in dB | lower is better |
| `rir_ipd_mae` | binaural IPD error in radians | lower is better |
| `rir_doa_mae` | binaural TDOA-derived DOA error in degrees | lower is better |

The default bundle is:

```text
rir.any.multi.rir_npm.rir_npm_v1__rt60_mae.rt60_mae_v1
```

The bundle's top-level `score` is `rir_npm`; both metric scores are in
`details.results`. This is a reporting convention, not a combined score.

For the multi-channel binaural chain, start with the same NPM and RT60 base
metrics, then select the registered bundle containing
`rir_iacc_mae`, `rir_itd_mae`, `rir_ild_mae`, and `rir_ipd_mae` together with
the channel-wise metrics. Its top-level score is `rir_npm`; spatial
scores remain separate entries in `details.results`.

Example row:

```json
{"sample_id":"room001_utt001","prediction_rir":"pred.wav","reference_rir":"ref.wav","sample_rate":16000,"channel_layout":"mono"}
```

`rir_iacc_mae`, `rir_itd_mae`, `rir_ild_mae`, `rir_ipd_mae`, and `rir_doa_mae`
require exactly two channels and use them as left/right. IACC and ITD use a
normalized cross-correlation search over +/-1 ms. `rir_doa_mae` additionally
requires `metadata.microphone_spacing_m`; its optional `sound_speed_mps`
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
