# G-STAR SA-ASR Normalization

## Purpose

`normalization/gstar_norm` preserves the text normalization rule used by the
legacy SA-ASR evaluator before MeetEval cpWER/DER scoring. It operates on
key-text files produced by conversion utilities and keeps STM parsing outside
the node.

The node normalizes text only. It does not run MeetEval and does not calculate
cpWER or DER.

## Task Scenarios

- SA-ASR Mandarin cpWER route:
  `sa_asr.zh.cpwer.conversion_sa_asr_cpwer_v1.gstar_norm_v1.meeteval_v1`.
- Used after SA-ASR STM/key-text conversion and before `scoring/meeteval`.

## Input

- Schema: `key_text_files`.
- Row format: `<key><TAB><text>`.
- Required roles after conversion: `ref`, `hyp`.

Task-specific formats such as STM are converted by
`src/sure_eval/evaluation/conversion/` before this node runs.

## Output

- Schema: `key_text_files`.
- Output files preserve keys and contain normalized text for MeetEval input
  reconstruction.
- Trace records `case_sensitive=False` and `remove_tag=True` behavior.

## Versioned Computation

- Node id: `normalization/gstar_norm`.
- Version: `v1`.
- Internal stages:
  - `key_text_parse`
  - `gstar_text_normalize`
- Reference compatibility: `SUREEvaluator._eval_sa_asr`.

## Runtime and Assets

- Runtime: `in_process`.
- No optional packages, checkpoints, or external binaries.

## Source and References

- G-STAR paper: G-STAR: End-to-End Global Speaker-Tracking Attributed
  Recognition, https://arxiv.org/pdf/2603.10468v2
- Source: local SURE legacy compatibility implementation.
- This node documents the project-local G-STAR compatibility behavior preserved
  from the existing evaluator.

## Limitations

- The node does not parse STM, RTTM, CTM, or other diarization annotation
  formats.
- Any change to the text rule should bump the node version because cpWER is
  sensitive to normalization.
