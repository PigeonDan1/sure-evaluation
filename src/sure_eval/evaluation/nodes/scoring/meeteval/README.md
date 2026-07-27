# MeetEval Scoring

## Purpose

`scoring/meeteval` wraps the MeetEval toolkit for diarization and
speaker-attributed ASR metrics. The node delegates annotation parsing to
`meeteval.io.load` instead of maintaining a local parser for every supported
annotation format.

## Task Scenarios

- SD DER route: `sd.any.der.meeteval_v1`.
- SA-ASR cpWER route:
  `sa_asr.en.cpwer.conversion_sa_asr_cpwer_v1.gstar_norm_v1.meeteval_v1`.

## Input

- Schema: `meeteval_annotation_files`.
- Supported input families, depending on selected metric:
  - STM
  - CTM
  - SegLST
  - RTTM for DER
- Required roles: `ref`, `hyp`.

## Output

- Schema: `meeteval_metric_report`.
- Metrics:
  - `der`: diarization error rate via `meeteval.der.dscore`.
  - `cpwer`: concatenated minimum-permutation WER via
    `meeteval.wer.cpwer`.
- Report details include selected metric, collar, companion metrics, loader,
  aggregation rule, and per-session results when available.
- Lower scores are better for DER and cpWER.

## Versioned Computation

- Node id: `scoring/meeteval`.
- Version: `v1`.
- Internal stages:
  - `meeteval_load`
  - `metric_scoring`
  - `result_aggregation`
- Reference compatibility:
  - `SUREEvaluator._eval_sd`
  - `SUREEvaluator._eval_sa_asr`

## Runtime and Assets

- Runtime: optional node-local `uv` project.
- Python: `3.11`.
- Package: `meeteval`.
- DER uses `md-eval-22.pl` through MeetEval. Keep a local copy at one of:

```text
src/sure_eval/evaluation/nodes/scoring/meeteval/md-eval-22.pl
src/sure_eval/evaluation/nodes/scoring/meeteval/.cache/md-eval-22.pl
```

Setup:

```bash
sure-eval env setup --node scoring/meeteval
```

## Source and References

- MeetEval: https://github.com/fgnt/meeteval
- NIST md-eval is used by MeetEval for DER-compatible scoring.

## Limitations

- Route-level parameters such as collar must be recorded and kept identical
  for fair comparison.
- Do not keep persistent MeetEval environments in `/tmp`; use the node-local
  environment declared by `node_env.yaml`.
