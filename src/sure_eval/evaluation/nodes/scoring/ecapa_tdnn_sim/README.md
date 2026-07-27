# ECAPA-TDNN Speaker Similarity Scoring Node

## Purpose

`scoring/ecapa_tdnn_sim` scores speaker similarity with an ECAPA-TDNN provider.
It reports the canonical metric `spk_sim`; `sim/ecapa-tdnn` is the
method-specific selector recorded in `execution_metrics`.

The node wraps shared speaker-similarity normalization and aggregation. Heavy
SpeechBrain model loading stays inside the node-local provider.

## Task Scenarios

- TTS speaker similarity.
- VC speaker similarity.
- Alternative `spk_sim` provider when exact pipeline id selects
  `ecapa_tdnn_sim_v1`.

## Input

- Schema: `speaker_audio_pairs`.
- Required roles:
  - generated/prediction audio
  - reference speaker audio
- Rows should be aligned by sample id.

## Output

- Schema: `provider_normalized_similarity`.
- Canonical metric: `spk_sim`.
- Method selector: `sim/ecapa-tdnn`.
- Aggregation: mean similarity.
- Higher scores indicate more similar speaker embeddings.

## Versioned Computation

- Node id: `scoring/ecapa_tdnn_sim`.
- Version: `v1`.
- Backend/method: `ecapa-tdnn`.
- Model id: `speechbrain/spkrec-ecapa-voxceleb`.
- Internal stages:
  - `embedding_or_score_provider`
  - `score_normalization`
  - `mean_aggregation`

## Runtime and Assets

- Runtime: optional node-local `uv` project.
- Python: `3.11`.
- GPU: optional.
- Model env var: `ECAPA_TDNN_SIM_CHECKPOINT`.
- Verify imports: `speechbrain`, `torch`.

## Source and References

- SpeechBrain ECAPA model card:
  https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- SpeechBrain: https://github.com/speechbrain/speechbrain

## Limitations

- Scores are method-specific and should not be merged with WavLM or ERes2Net
  `spk_sim` scores unless the pipeline id is identical.
- Checkpoint paths are runtime assets and must not be committed.
