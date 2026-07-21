# SI-SDR Scoring Node

`scoring/si_sdr` computes Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
between predicted/enhanced audio and clean reference audio.

## Inputs

- `prediction_audio` or `enhanced_audio`: extracted/enhanced model output
- `reference_audio`: clean reference audio
- `mixed_audio` (optional): mixed speech input for TSE SI-SDRi

SE routes call this node through a full-reference provider. TSE routes call the
native scorer so `mixed_audio` can additionally report `si_sdri`:

```text
SI-SDRi = SI-SDR(prediction, clean) - SI-SDR(mixture, clean)
```

Batch JSONL accepts either `prediction_audio` or `enhanced_audio`; `mixed_audio`
is optional per row.

## Usage

```bash
python -m sure_eval.evaluation.nodes.scoring.si_sdr.node \
  --prediction-audio predicted.wav --reference-audio clean.wav
```

With a mixture input:

```bash
python -m sure_eval.evaluation.nodes.scoring.si_sdr.node \
  --prediction-audio predicted.wav --reference-audio clean.wav --mixed-audio mixed.wav --json
```
