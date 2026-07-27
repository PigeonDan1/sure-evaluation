# UTMOS Scoring Node

## Purpose

`scoring/utmos` scores generated audio with a UTMOS provider and reports the
canonical atomic metric `utmos`. UTMOS is kept separate from DNSMOS and WV-MOS
because each metric has a distinct model and score meaning.

Heavy UTMOS execution stays inside the node-local provider.

## Task Scenarios

- TTS no-reference audio quality.
- VC no-reference audio quality.
- SE/TSE no-reference audio quality when selected by route.

## Input

- Schema: `generated_audio_rows`.
- Required audio role: `prediction_audio` or task-converted generated audio.

## Output

- Schema: `provider_normalized_mos`.
- Metric: `utmos`.
- Aggregation: mean score over rows.
- Higher scores are better.

## Versioned Computation

- Node id: `scoring/utmos`.
- Version: `v1`.
- Backend: `utmos`.
- Internal stages:
  - `audio_score_provider`
  - `score_normalization`
  - `mean_aggregation`

## Runtime and Assets

- Runtime: optional node-local `uv` project.
- Python: `3.8`.
- GPU: optional.
- Required checkpoint env var: `UTMOS_CHECKPOINT`.
- Default checkpoint target: `checkpoints/UTMOS-demo/epoch=3-step=7459.ckpt`.
- Verify imports: `torch`, `fairseq`.

## Source and References

- UTMOS public implementation: https://github.com/sarulab-speech/UTMOS22

## Limitations

- UTMOS is a no-reference MOS prediction model, not a human listening test.
- The checkpoint is a runtime asset and must not be committed.
