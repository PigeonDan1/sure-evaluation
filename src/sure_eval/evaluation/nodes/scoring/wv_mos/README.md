# WV-MOS Scoring Node

## Purpose

`scoring/wv_mos` scores generated audio with a Wav2Vec2MOS provider and
reports the canonical metric `wv_mos`. The execution selector `wv-mos` is
accepted for compatibility with upstream naming.

WV-MOS is a distinct atomic metric and should not be merged into a generic
`mos` metric.

## Task Scenarios

- TTS no-reference audio quality.
- VC no-reference audio quality.
- SE/TSE no-reference audio quality when selected by route.

## Input

- Schema: `generated_audio_rows`.
- Required audio role: `prediction_audio` or task-converted generated audio.

## Output

- Schema: `provider_normalized_mos`.
- Metric: `wv_mos`.
- Compatibility selector: `wv-mos`.
- Aggregation: mean score over rows.
- Higher scores are better.

## Versioned Computation

- Node id: `scoring/wv_mos`.
- Version: `v1`.
- Backend: `wv-mos`.
- Internal stages:
  - `audio_score_provider`
  - `score_normalization`
  - `mean_aggregation`

## Runtime and Assets

- Runtime: optional node-local `uv` project.
- Python: `3.11`.
- GPU: optional.
- Required checkpoint env var: `WV_MOS_CHECKPOINT`.
- Default checkpoint target: `checkpoints/wv-mos/wav2vec2.ckpt`.
- Verify imports: `torch`, `transformers`.

## Source and References

- WV-MOS implementation: https://github.com/AndreevP/wvmos
- WV-MOS paper: https://arxiv.org/abs/2203.13086

## Limitations

- WV-MOS is a no-reference MOS prediction model, not a human listening test.
- The checkpoint is a runtime asset and must not be committed.
