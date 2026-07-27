# ERes2Net Speaker Similarity Scoring Node

## Purpose

`scoring/eres2net_sim` scores speaker similarity with an ERes2Net provider. It
reports the canonical metric `spk_sim`; `sim/eres2net` is the method-specific
selector recorded in `execution_metrics`.

Heavy ModelScope or 3D-Speaker model loading stays inside the node-local
provider.

## Task Scenarios

- TTS speaker similarity.
- VC speaker similarity.
- Alternative `spk_sim` provider when exact pipeline id selects
  `eres2net_sim_v1`.

## Input

- Schema: `speaker_audio_pairs`.
- Required roles:
  - generated/prediction audio
  - reference speaker audio
- Rows should be aligned by sample id.

## Output

- Schema: `provider_normalized_similarity`.
- Canonical metric: `spk_sim`.
- Method selector: `sim/eres2net`.
- Aggregation: mean similarity.
- Higher scores indicate more similar speaker embeddings.

## Versioned Computation

- Node id: `scoring/eres2net_sim`.
- Version: `v1`.
- Backend/method: `eres2net`.
- Model id: `iic/speech_eres2net_sv_zh-cn_16k-common`.
- Internal stages:
  - `embedding_or_score_provider`
  - `score_normalization`
  - `mean_aggregation`

## Runtime and Assets

- Runtime: optional node-local `uv` project.
- Python: `3.11`.
- GPU: optional.
- Model env var: `ERES2NET_SIM_CHECKPOINT`.
- Verify imports: `modelscope`, `torch`.

## Source and References

- ModelScope model:
  https://modelscope.cn/models/iic/speech_eres2net_sv_zh-cn_16k-common
- 3D-Speaker toolkit: https://github.com/modelscope/3D-Speaker

## Limitations

- Scores are method-specific and should be compared only under identical
  pipeline ids.
- Checkpoint paths are runtime assets and must not be committed.
