# STOI Scoring

## Purpose

`scoring/stoi` computes short-time objective intelligibility for enhanced
speech against clean reference speech. It is a full-reference intelligibility
metric.

## Task Scenarios

- SE full-reference speech enhancement quality.
- Routes that select metric `stoi`.

## Input

- Schema: `speech_enhancement_audio_pairs`.
- Required roles:
  - `prediction_audio` or `enhanced_audio`
  - `reference_audio`

## Output

- Schema: `full_reference_audio_metric`.
- Metric: `stoi`.
- Aggregation: mean over evaluated pairs.
- Higher scores are better.

## Versioned Computation

- Node id: `scoring/stoi`.
- Version: `v1`.
- Backend: `pystoi`.
- Internal stages:
  - `audio_pair_decode`
  - `metric_provider`
  - `score_normalization`
  - `mean_aggregation`
- Default provider uses 16 kHz audio.

## Runtime and Assets

- Runtime: optional `pip` node.
- Package: `pystoi>=0.4.1`.
- Install with:

```bash
pip install -e ".[se]"
```

## Source and References

- pystoi: https://github.com/mpariente/pystoi

## Limitations

- STOI is an intelligibility proxy, not a speech quality MOS metric.
- It requires clean reference audio.
