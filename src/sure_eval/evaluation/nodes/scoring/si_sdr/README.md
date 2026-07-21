# SI-SDR Scoring Node

`scoring/si_sdr` computes Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) between predicted and reference audio.

## Metric

SI-SDR measures the quality of separated/extracted speech signals relative to clean references.
It is invariant to the scaling of the prediction, making it the standard metric for speech
separation and target speaker extraction tasks.

Formula:

```
s        <- s - mean(s),  s_hat <- s_hat - mean(s_hat)    # zero-mean both signals
s_target = <s, s_hat> * s / ||s||^2
e_noise  = s_hat - s_target
SI-SDR   = 10 * log10(||s_target||^2 / ||e_noise||^2)
```

Higher is better. Identical signals yield positive infinity.

### SI-SDRi (improvement over mixture)

When a `mixed_audio` (the mixed/noisy input fed to the extraction model) is supplied for a sample,
the node additionally reports **SI-SDRi** — the improvement of the extracted signal over the mixture:

```
SI-SDRi = SI-SDR(prediction, clean) − SI-SDR(mixture, clean)
```

Higher is better. SI-SDRi is emitted as a companion field (`si_sdri`) next to `si_sdr` for every
sample that has a `mixed_audio`; samples without one report `si_sdr` only. The aggregated result
adds a mean `si_sdri` across the samples that provided a mixture.

## Inputs

- `prediction_audio`: path to the extracted/predicted audio file
- `reference_audio`: path to the clean reference audio file
- `mixed_audio` (optional): path to the mixed speech input; when provided, an `si_sdri`
  improvement score is reported alongside `si_sdr`

All files are loaded as mono (multi-channel averaged to mono) and length-aligned by truncation
to the shorter signal. If sample rates differ, they are resampled to the lower rate using
`scipy.signal.resample_poly`.

## Environment

Pure CPU computation. No model weights or GPU required.

Dependencies: `numpy`, `soundfile`, `scipy`.

## Usage

```bash
python -m sure_eval.evaluation.nodes.scoring.si_sdr.node \
  --prediction-audio predicted.wav --reference-audio clean.wav
```

With a mixture input (also reports `si_sdri`):

```bash
python -m sure_eval.evaluation.nodes.scoring.si_sdr.node \
  --prediction-audio predicted.wav --reference-audio clean.wav --mixed-audio mixed.wav --json
```

Batch mode via JSONL (`{"key": "...", "prediction_audio": "...", "reference_audio": "...", "mixed_audio": "..."}`;
`mixed_audio` is optional per row):

```bash
python -m sure_eval.evaluation.nodes.scoring.si_sdr.node --input-jsonl rows.jsonl --json
```