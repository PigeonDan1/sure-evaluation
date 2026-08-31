# WeNet WER/CER/MER Scoring

## Purpose

`scoring/wenet_wer` wraps the vendored WeNet `compute_wer` script as the
default ASR edit-distance scoring backend. The wrapper exposes WER, CER, and
code-switch MER variants while keeping upstream scoring semantics visible in
pipeline traces.

## Task Scenarios

- ASR English WER after `normalization/whisper_norm` or legacy
  `normalization/site_norm`.
- ASR Chinese CER after `normalization/wetext_norm` or legacy
  `normalization/site_norm`.
- ASR code-switch MER after `normalization/site_norm`.
- TTS/VC/TSE semantic WER/CER after transcription and task-specific
  normalization.

## Input

- Schema: `normalized_key_text_files`.
- Row format: `<key><TAB><normalized text>`.
- Required roles: `ref`, `hyp`.

## Output

- Schema: `edit_distance_counts`.
- Aliased node ids:
  - `scoring/wenet_wer`
  - `scoring/wenet_cer`
  - `scoring/wenet_mer`
- Report fields include `all`, `cor`, `sub`, `ins`, `del`,
  missing/extra hypothesis counts, metric percentage, and `score`.
- Lower scores are better.

## Versioned Computation

- Node id: `scoring/wenet_wer`.
- Version: `v1`.
- Internal stages:
  - `tokenization`
  - `case_normalization`
  - `edit_distance`
- The wrapped script includes tokenization, optional character splitting, case
  normalization, tag stripping, and edit-distance counting.

The only intentional local deviation is performance-only: the DP matrix reset
touches only the submatrix used by the current utterance. Scores are unchanged
and covered by regression tests.

## Runtime and Assets

- Runtime: `in_process`.
- No optional packages, checkpoints, or external binaries.

## Source and References

- WeNet compute-wer source:
  https://github.com/wenet-e2e/wenet/blob/main/tools/compute-wer.py
- Vendored implementation:
  `src/sure_eval/evaluation/nodes/scoring/wenet_wer/wenet_compute_cer.py`

## Limitations

- Normalization is upstream; this node should receive normalized key-text
  files.
- Reference utterances missing from the hypothesis are scored as empty
  hypotheses instead of being silently skipped.
- Fully empty/malformed coverage raises instead of reporting a perfect score.
