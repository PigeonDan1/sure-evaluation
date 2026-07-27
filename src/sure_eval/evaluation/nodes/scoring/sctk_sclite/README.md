# NIST SCTK sclite Scoring

## Purpose

`scoring/sctk_sclite` wraps the NIST SCTK `sclite` binary as an optional ASR
scoring backend. It does not install SCTK into the main Python environment and
is not selected by default ASR routes.

## Task Scenarios

- Optional ASR WER/CER backend for users who need SCTK-compatible reports.
- Default ASR scoring remains `scoring/wenet_wer`, `scoring/wenet_cer`, or
  `scoring/wenet_mer`.

## Input

- Schema: `normalized_key_text_files`.
- Row format: `<key><TAB><normalized text>`.
- Required roles: `ref`, `hyp`.
- Intermediate schema: TRN.
- Reference and hypothesis key sets must match exactly.

## Output

- Schema: `sctk_sclite_metric_report`.
- Metrics: `wer`, `cer`.
- Lower scores are better.
- Parsed report fields come from `sclite` output.

## Versioned Computation

- Node id: `scoring/sctk_sclite`.
- Version: `v1`.
- Toolkit: NIST SCTK `sclite`.
- Pinned commit: `9688a26882a688132a5e414cadcb4c19b6fffaba`.
- Internal stages:
  - `key_text_parse`
  - `trn_materialize`
  - `sclite`
  - `sclite_report_parse`

## Runtime and Assets

- Runtime: optional external binary.
- Build script:

```bash
bash src/sure_eval/evaluation/nodes/scoring/sctk_sclite/build_sctk.sh
```

Binary resolution order:

1. Explicit `sclite_bin=` argument.
2. `SURE_EVAL_SCLITE_BIN`.
3. `PATH`.
4. `SURE_EVAL_SCTK_ROOT/<pinned_commit>/bin/sclite`.
5. `${SURE_EVAL_CACHE_DIR:-~/.cache/sure-eval}/tools/sctk/<pinned_commit>/bin/sclite`.
6. This node's `.local/sctk/<pinned_commit>/bin/sclite`.

## Source and References

- NIST SCTK: https://github.com/usnistgov/SCTK

## Limitations

- This implementation currently supports key-text to TRN scoring.
- STM/CTM timed scoring is intentionally left for a later extension.
- Normalization must happen upstream.
