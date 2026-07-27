# Canonical Written-Form ITN Normalization

## Purpose

`normalization/canonical_itn` converts ASR reference and hypothesis text into a
canonical written form before token-level CER, WER, or MER scoring. The core
idea is that inverse text normalization is many-to-one: `2024`, `二零二四`,
and `两千零二十四` can collapse to the same written representation while real
recognition errors still remain visible to the scorer.

The node normalizes text only. It does not compute edit distance.

## Task Scenarios

- ASR Chinese canonical CER:
  `asr.zh.cer.canonical_itn_zh_v1.token_cer_v1`.
- ASR English canonical WER:
  `asr.en.wer.canonical_itn_en_v1.token_mer_v1`.
- ASR code-switch canonical MER:
  `asr.cs.mer.canonical_itn_cs_v1.token_mer_v1`.

## Input

- Schema: `key_text_files`.
- Row format: `<key><TAB><text>`.
- Required roles: `ref`, `hyp`.

Empty-text rows are preserved and files with no parseable key-text rows raise.

## Output

- Schema: `key_text_files`.
- Output text is canonicalized written-form text.
- Trace records cn2an version, fallback counts, row statistics, and generated
  temporary files.

## Versioned Computation

- Node id: `normalization/canonical_itn`.
- Version: `v1`.
- Internal stages:
  - `nfkc_lowercase`
  - `idiom_unit_masking`
  - `cn2an_itn`
  - `cjk_numeral_span_pass`
  - `percent_rewrite`
  - `mixed_number_expansion`
  - `punctuation_spacing`

Punctuation is replaced by spaces rather than deleted. `%`, `$`, `¥`, `°`, and
digit-context `.`, `/`, `-` are preserved as semantic symbols. English and
code-switch paths additionally use the vendored Whisper English normalizer for
Latin spans before the shared token scorer.

## Runtime and Assets

- Runtime: optional `pip` node.
- Package: `cn2an>=0.5.24,<0.6`.
- Paired scoring package: `rapidfuzz>=3.0,<4` through `scoring/token_cer` or
  `scoring/token_mer`.
- Install with:

```bash
pip install -e ".[canonical]"
```

## Source and References

- cn2an: https://github.com/Ailln/cn2an
- RapidFuzz: https://github.com/rapidfuzz/RapidFuzz
- Local chain implementation:
  `src/sure_eval/evaluation/nodes/normalization/canonical_itn/chain.py`

## Limitations

- Clock times such as `两点半` versus `2:30` are not fully unified.
- Unit lexemes such as `千克` versus `kg` are not unified.
- Approximate numerals such as `七八个` may merge with exact written numbers.
- Per-string cn2an failures deterministically fall back to the original string
  and are counted in the trace.
