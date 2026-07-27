# Token-Level CER Scoring

## Purpose

`scoring/token_cer` scores canonical written-form key-text files with
token-level edit distance. It is designed to pair with
`normalization/canonical_itn`.

## Task Scenarios

- ASR Chinese canonical CER:
  `asr.zh.cer.canonical_itn_zh_v1.token_cer_v1`.

## Input

- Schema: `key_text_files`.
- Row format: `<key><TAB><canonical text>`.
- Required roles: `ref`, `hyp`.
- Text should already be normalized by `normalization/canonical_itn`.

## Output

- Schema: `metric_result`.
- Report fields include total reference tokens, `sub`, `del`, `ins`,
  missing/extra hypothesis counts, `spacing_repairs`, and `score`.
- Score is corpus micro-average:

```text
(sub + del + ins) / total_reference_tokens
```

- Lower scores are better.

## Versioned Computation

- Node id: `scoring/token_cer`.
- Version: `v1`.
- Tokenization:
  - CJK: one token per character.
  - Latin letter runs: one token per word.
  - Digits: one token per character.
  - Surviving semantic symbols: one token each.
- Distance: RapidFuzz Levenshtein with unit costs.
- Internal stages:
  - `canonical_tokenization`
  - `token_edit_distance`
  - `sdi_decomposition`

## Runtime and Assets

- Runtime: optional `pip` node.
- Package: `rapidfuzz>=3.0,<4`.
- Install with:

```bash
pip install -e ".[canonical]"
```

## Source and References

- RapidFuzz: https://github.com/rapidfuzz/RapidFuzz
- Local tokenizer and coverage policy:
  `src/sure_eval/evaluation/nodes/scoring/token_cer/node.py`

## Limitations

- It assumes canonical written-form input; using raw ASR text will change the
  metric meaning.
- Missing hypothesis utterances are scored as empty hypotheses.
- Zero covered reference tokens raise instead of reporting 0.0.
