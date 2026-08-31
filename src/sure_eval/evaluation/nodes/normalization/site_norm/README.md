# Site ASR Normalization

## Purpose

`normalization/site_norm` preserves the legacy Site-style ASR text
normalization that used to live inside `SUREEvaluator._eval_asr` and
`SUREEvaluator._eval_asr_codeswitch`. It is now a versioned pipeline node so
reports expose the exact normalization backend used before edit-distance
scoring.

The node normalizes text only. It does not compute WER, CER, or MER.

## Task Scenarios

- ASR Chinese CER legacy-compatible route:
  `asr.zh.cer.site_norm_zh_v1.wenet_cer_v1`.
- ASR English WER legacy-compatible route:
  `asr.en.wer.site_norm_en_v1.wenet_wer_v1`.
- ASR code-switch MER default route:
  `asr.cs.mer.site_norm_cs_v1.wenet_mer_v1`.

## Input

- Schema: `key_text_files`.
- Row format: `<key><TAB><text>`.
- Required roles: `ref`, `hyp`.

Rows with an empty text field are preserved so downstream scoring counts them
as deletions. Non-empty lines without a tab are dropped and counted in
`row_stats`; a file with no parseable rows raises.

## Output

- Schema: `key_text_files`.
- Output files keep the same keys and contain normalized text.
- Trace fields include selected profile, row statistics, and temporary output
  file paths.

## Versioned Computation

- Node id: `normalization/site_norm`.
- Version: `v1`.
- Profiles:
  - `zh`: number text normalization plus punctuation stripping.
  - `en`: number text normalization plus punctuation stripping.
  - `cs`: code-switch tokenization, language split, and number text
    normalization.
- Internal stages:
  - `number_text_normalization`
  - `punctuation_stripping`
  - `codeswitch_tokenization` and `language_split` for `cs`.

Normalization map tables are loaded once per node call and shared across rows.
This is a performance change only; scoring output is intended to remain
legacy-compatible.

## Runtime and Assets

- Runtime: `in_process`.
- No optional node-local environment.
- Uses vendored/local normalization implementation under
  `normalization/site_norm/normalization_impl/`.

## Source and References

- Source: local SURE legacy compatibility implementation.
- External source not identified; this node documents the behavior preserved
  from the existing evaluator rather than a separately versioned public
  toolkit.

## Limitations

- This node intentionally keeps legacy behavior even where a cleaner
  normalization policy might be preferable.
- For new canonical ASR routes, prefer `normalization/canonical_itn` plus the
  token-level scorers.
