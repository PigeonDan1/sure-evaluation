# DNSMOS Scoring Node

## Purpose

`scoring/dnsmos` scores generated audio with a DNSMOS provider and reports the
canonical atomic metric `dnsmos`. DNSMOS is not folded into a generic `mos`
metric because the metric identity and backend are semantically specific.

The node wraps SURE's shared MOS normalization and aggregation. Heavy ONNX
execution stays inside the node-local provider.

## Task Scenarios

- TTS no-reference audio quality.
- VC no-reference audio quality.
- SE/TSE no-reference audio quality when selected by route.

## Input

- Schema: `generated_audio_rows`.
- Required audio role: `prediction_audio` or task-converted generated audio.
- Each row should carry a stable sample id for per-row reporting.

## Output

- Schema: `provider_normalized_mos`.
- Metric: `dnsmos`.
- Output contains normalized provider fields and corpus mean `score`.
- Higher scores are better.

## Versioned Computation

- Node id: `scoring/dnsmos`.
- Version: `v1`.
- Backend: `dnsmos`.
- Internal stages:
  - `audio_score_provider`
  - `score_normalization`
  - `mean_aggregation`

## Runtime and Assets

- Runtime: optional node-local `uv` project.
- Python: `3.11`.
- Required model env var: `DNSMOS_CHECKPOINT`.
- Default model target: `checkpoints/DNSMOS/model_v8.onnx`.
- Verify imports: `librosa`, `onnxruntime`.

## Source and References

- Microsoft DNSMOS under DNS-Challenge:
  https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS

## Limitations

- The ONNX model is a runtime asset and must not be committed.
- DNSMOS is a no-reference proxy metric, not human subjective MOS.
