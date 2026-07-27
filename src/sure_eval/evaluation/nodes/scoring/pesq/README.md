# PESQ Scoring

## Purpose

`scoring/pesq` computes perceptual evaluation of speech quality for enhanced
speech against clean reference speech. It is a full-reference audio metric.

## Task Scenarios

- SE full-reference speech enhancement quality.
- Routes that select metric `pesq`.

## Input

- Schema: `speech_enhancement_audio_pairs`.
- Required roles:
  - `prediction_audio` or `enhanced_audio`
  - `reference_audio`
- Audio is decoded by the full-reference provider.

## Output

- Schema: `full_reference_audio_metric`.
- Metric: `pesq`.
- Aggregation: mean over evaluated pairs.
- Higher scores are better.

## Versioned Computation

- Node id: `scoring/pesq`.
- Version: `v1`.
- Backend: `pesq`.
- Internal stages:
  - `audio_pair_decode`
  - `metric_provider`
  - `score_normalization`
  - `mean_aggregation`
- Default provider uses wide-band PESQ at 16 kHz.

## Runtime and Assets

- Runtime: optional `pip` node.
- Package: `pesq>=0.0.4`.
- Install with:

```bash
pip install -e ".[se]"
```

## Source and References

- ITU-T P.862 recommendation:
  https://www.itu.int/rec/T-REC-P.862
- Python PESQ package: https://github.com/ludlows/PESQ

## Limitations

- PESQ has sampling-rate and mode assumptions; this node defaults to 16 kHz
  wide-band provider behavior.
- It is not valid for no-reference audio quality scoring.
