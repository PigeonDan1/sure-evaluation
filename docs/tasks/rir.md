# RIR Estimation and Spatial Audio

The RIR estimation task evaluates predicted room impulse responses against reference RIRs.
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
metrics. IACC is restricted to an exact binaural pair. ITD, ILD, IPD, and DOA
accept two or more channels and report one pair for every `chN` relative to
`ch0`; the metric score is the mean over those pairs. Set `channel_layout` to
`mono`, `multi_channel`, or `binaural_lr` when the layout is known; `auto` is
used when it is omitted.

## Metrics

| Metric | Meaning | Direction |
|---|---|---|
| `rir_npm` | normalized projection misalignment in dB | lower is better |
| `rt60_mae` | absolute RT60 error in seconds | lower is better |
| `rir_drr_mae` | direct-to-reverberant ratio error in dB | lower is better |
| `rir_c50_mae` | C50 error in dB | lower is better |
| `rir_c80_mae` | C80 error in dB | lower is better |
| `rir_iacc_mae` | interaural cross-correlation coefficient error in a +/-1 ms window | lower is better |
| `rir_itd_mae` | interaural time difference error in ms in a +/-1 ms window | lower is better |
| `rir_ild_mae` | ch0-relative interchannel level error in dB | lower is better |
| `rir_ipd_mae` | ch0-relative interchannel phase error in radians | lower is better |
| `rir_doa_mae` | ch0-relative TDOA-derived DOA error in degrees | lower is better |

The default bundle is:

```text
rir.any.multi.rir_npm.rir_npm_v1__rt60_mae.rt60_mae_v1
```

The bundle's top-level `score` is `rir_npm`; both metric scores are in
`details.results`. This is a reporting convention, not a combined score.

For the multi-channel spatial chain, start with the same NPM and RT60 base
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
