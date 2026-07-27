# WavLM-Large Speaker Similarity Scoring Node

## Purpose

`scoring/wavlm_large_sim` scores speaker similarity with a WavLM-large based
provider. It reports the canonical metric `spk_sim`; `sim/wavlm-large` is the
method-specific selector recorded in `execution_metrics`.

The default provider follows the Seed-TTS-Eval style WavLM-large speaker
verification path: load `wavlm_large_finetune.pth`, aggregate WavLM hidden
states through the finetuned head, then score by cosine similarity.

## Task Scenarios

- TTS speaker similarity.
- VC speaker similarity.
- Default or selected `spk_sim` provider for routes whose exact pipeline id
  includes `wavlm_large_sim_v1`.

## Input

- Schema: `speaker_audio_pairs`.
- Required roles:
  - generated/prediction audio
  - reference speaker audio
- Rows should be aligned by sample id.

## Output

- Schema: `provider_normalized_similarity`.
- Canonical metric: `spk_sim`.
- Method selector: `sim/wavlm-large`.
- Aggregation: mean similarity.
- Higher scores indicate greater speaker similarity.

## Versioned Computation

- Node id: `scoring/wavlm_large_sim`.
- Version: `v1`.
- Backend/method: `wavlm-large`.
- Model id: `microsoft/wavlm-large`.
- Internal stages:
  - `embedding_or_score_provider`
  - `score_normalization`
  - `mean_aggregation`

## Runtime and Assets

- Runtime: optional node-local `uv` project.
- Python: `3.11`.
- GPU: optional.
- Required checkpoint env var: `WAVLM_LARGE_SIM_CHECKPOINT`.
- Default finetuned checkpoint target: `checkpoints/wavlm_large_finetune.pth`.
- Base config env var: `WAVLM_LARGE_BASE_CONFIG`.
- Verify imports: `torch`, `transformers`.

## Source and References

- WavLM model card: https://huggingface.co/microsoft/wavlm-large
- WavLM repository: https://github.com/microsoft/unilm/tree/master/wavlm
- Seed-TTS-Eval speaker similarity implementation:
  https://github.com/BytedanceSpeech/seed-tts-eval

## Limitations

- The finetuned checkpoint is a runtime asset and must not be committed.
- Scores are backend-specific; compare WavLM similarity only against the same
  pipeline id and checkpoint.
