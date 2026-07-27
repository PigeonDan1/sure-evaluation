# SI-SDR Scoring

## Purpose

`scoring/si_sdr` computes Scale-Invariant Signal-to-Distortion Ratio between
predicted or enhanced audio and clean reference audio. It is a full-reference
signal metric.

## Task Scenarios

- TSE signal quality, including optional SI-SDR improvement.
- SE full-reference speech enhancement quality.

## Input

- Schema: `audio_signal_or_enhancement_pairs`.
- Required roles:
  - `prediction_audio` or `enhanced_audio`
  - `reference_audio`
- Optional role:
  - `mixed_audio` for TSE SI-SDRi.

Batch JSONL accepts either `prediction_audio` or `enhanced_audio`; `mixed_audio`
is optional per row.

## Output

- Schema: `si_sdr_score`.
- Metric: `si_sdr`; alias: `si-sdr`.
- TSE may also report:

```text
SI-SDRi = SI-SDR(prediction, clean) - SI-SDR(mixture, clean)
```

- Higher scores are better.

## Versioned Computation

- Node id: `scoring/si_sdr`.
- Version: `v1`.
- Backend: `numpy-si-sdr`.
- Internal stages:
  - `audio_load`
  - `length_align`
  - `si_sdr_compute`
  - `audio_pair_decode`
  - `metric_provider`
  - `score_normalization`
  - `mean_aggregation`

## Runtime and Assets

- Runtime: in-process for core numpy implementation; node-local `uv` metadata
  exists for isolated signal tests.
- No model checkpoint.

Example:

```bash
python -m sure_eval.evaluation.nodes.scoring.si_sdr.node \
  --prediction-audio predicted.wav --reference-audio clean.wav --json
```

## Source and References

- SI-SDR discussion and definition:
  https://arxiv.org/abs/1811.02508

## Limitations

- SI-SDR is scale-invariant and does not directly measure perceptual MOS.
- Input signals are length-aligned before scoring; alignment policy is part of
  this node version.
