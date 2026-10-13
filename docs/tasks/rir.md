# RIR Estimation and Spatial Audio

The RIR task evaluates predicted room impulse responses against reference RIRs.
Rows use `prediction_rir` and `reference_rir` paths and align by `sample_id`.
The audio files must have matching sample rates and channel counts; spatial
metrics require at least two channels. Each pair is truncated to the shorter
file before scoring; no peak or time-shift alignment is applied. Rows must
already refer to the same source, room, microphone positions, channel order,
and time origin. Each metric averages its per-sample scores with equal weight.

## Metrics

| Metric | Meaning | Direction |
|---|---|---|
| `rir_nmse` | normalized mean squared RIR error | lower is better |
| `rir_corr` | channel-wise waveform correlation | higher is better |
| `rt60_mae` | absolute RT60 error in seconds | lower is better |
| `rir_drr_mae` | direct-to-reverberant ratio error in dB | lower is better |
| `rir_c50_mae` | C50 error in dB | lower is better |
| `rir_c80_mae` | C80 error in dB | lower is better |
| `rir_ild_mae` | binaural ILD error in dB | lower is better |
| `rir_ipd_mae` | binaural IPD error in radians | lower is better |
| `rir_doa_mae` | binaural TDOA-derived DOA error in degrees | lower is better |

The default bundle is:

```text
rir.any.multi.rir_nmse.rir_nmse_v1__rir_corr.rir_corr_v1__rt60_mae.rt60_mae_v1
```

The bundle's top-level `score` is `rir_nmse`; all three metric scores are in
`details.results`. This is a reporting convention, not a combined score.

Example row:

```json
{"sample_id":"room001_utt001","prediction_rir":"pred.wav","reference_rir":"ref.wav","sample_rate":16000,"channel_layout":"mono"}
```

`rir_doa_mae` additionally requires `metadata.microphone_spacing_m`. Its
optional `sound_speed_mps` defaults to 343 m/s. ILD/IPD/DOA require at least
two channels and use the first two channels as left/right.

The scoring nodes run in-process with the package's `audio` extra. They do
not need model checkpoints or node-local environments. A model may generate
the predicted RIR in a separate environment such as `usddps-recrir`; scoring
only consumes the resulting audio files.

## CLI

```bash
sure-eval metric routes rir --metric rir_nmse --json
sure-eval metric describe rir --metric rir_nmse --output rir_pipeline.json --json
sure-eval metric run --pipeline rir_pipeline.json --samples-jsonl rir_samples.jsonl --output-dir /tmp/rir_eval --json
```
