# Cosine Trial Scores

## Purpose

`scoring/cosine_trial_scores` reads model-produced speaker embeddings, validates
an evaluator-owned testset trial manifest, and computes one cosine score for
every target or nontarget trial. It does not calculate a final verification
metric or expose trial labels to the model.

## Task Scenarios

- SV EER: `sv.any.eer.cosine_trial_scores_v1.det_eer_v1`.
- SV minDCF: `sv.any.min_dcf.cosine_trial_scores_v1.min_dcf_p005_v1`.
- It is the default shared scoring stage for both SV routes.

## Input

- Schema: `embedding_jsonl_plus_sv_trial_manifest`.
- Required roles: `sample_output`, `trial_manifest`.
- Alignment key: `sample_id`, represented by `key` or `sample_id` in the model
  output JSONL and by `enroll_key`/`test_key` in the trial TSV.
- Model rows contain `result.embedding`; all embeddings must be finite,
  non-empty, unique by key, and have the same dimension.
- The manifest schema is `sure.sv.trial_manifest.v1`. It identifies the
  OpenBench dataset, testset protocol, source dataset, trial counts, trial TSV,
  and frozen TSV SHA-256.
- Trial TSV columns are `enroll_key`, `test_key`, `label`, and `condition`.
  Labels are `target` or `nontarget`; `condition` is metadata-only.

## Output

- Schema: `sv_trial_score_memmaps`.
- Produces float32 cosine-score and uint8 binary-label NumPy memmaps aligned in
  trial-file order.
- Trace details include dataset, testset and source-dataset ids, trial and
  embedding counts, embedding dimension, extra embedding count, and trial-file
  SHA-256.
- Higher cosine scores indicate greater speaker similarity. Final metric
  direction is defined by the downstream EER or minDCF node.

## Versioned Computation

- Node id: `scoring/cosine_trial_scores`.
- Version: `v1`.
- Embeddings are L2-normalized before scoring; cosine similarity is then the
  dot product of the normalized enrollment and test vectors.
- Duplicate embedding keys, zero-norm vectors, missing trial keys, malformed
  labels, inconsistent dimensions, count mismatches, and trial fingerprint
  mismatches raise errors.
- Internal stages:
  - `embedding_validation`
  - `l2_normalization`
  - `trial_alignment`
  - `cosine_scoring`

## Runtime and Assets

- Runtime: `in_process`.
- Dependency: NumPy from the base package.
- No model checkpoint, external binary, network access, or node-local setup is
  required.
- Temporary memmaps are written inside the pipeline work directory.

## Source and References

- Local implementation:
  `src/sure_eval/evaluation/nodes/scoring/cosine_trial_scores/node.py`.
- Metric helpers:
  `src/sure_eval/evaluation/nodes/scoring/common/sv_metrics.py`.

## Limitations

- Only one-dimensional real-valued embeddings are supported.
- Enrollment is represented by one embedding key per trial; enrollment-side
  multi-utterance pooling must be performed before this node.
- Score normalization, calibration, cohort scoring, and condition-level metric
  aggregation are not implemented.
- The full embedding key index is kept in memory, while embeddings and trial
  outputs use disk-backed memmaps.
