# Mixed and English Token Error Rate

## Purpose

`scoring/token_mer` scores canonical written-form key-text files with a mixed
token edit-distance policy: CJK at character level, Latin text at word level,
digits per character, and symbols per character. It is used for English-heavy
and code-switch canonical ASR routes where pure character CER is not the right
comparison unit.

## Task Scenarios

- ASR English canonical WER:
  `asr.en.wer.canonical_itn_en_v1.token_mer_v1`.
- ASR code-switch canonical MER:
  `asr.cs.mer.canonical_itn_cs_v1.token_mer_v1`.

## Input

- Schema: `key_text_files`.
- Row format: `<key><TAB><canonical text>`.
- Required roles: `ref`, `hyp`.
- Text should already be normalized by `normalization/canonical_itn`.

## Output

- Schema: `metric_result`.
- Report fields include total reference tokens, `sub`, `del`, `ins`,
  missing/extra hypothesis counts, `spacing_repairs`, and `score`.
- Lower scores are better.

## Versioned Computation

- Node id: `scoring/token_mer`.
- Version: `v1`.
- It uses the same scorer and tokenizer as `scoring/token_cer`.
- For `en` and `cs`, Latin spans are Whisper-normalized before the shared token
  chain.
- Internal stages:
  - `canonical_tokenization`
  - `token_edit_distance`
  - `sdi_decomposition`

Degeneration guarantees are locked by tests:

- Text with no Latin letters scores identically to the canonical CER path.
- Text with no CJK scores identically to the canonical WER path.

## Runtime and Assets

- Runtime: optional `pip` node.
- Package: `rapidfuzz>=3.0,<4`.
- Install with:

```bash
pip install -e ".[canonical]"
```

## Source and References

- RapidFuzz: https://github.com/rapidfuzz/RapidFuzz
- Whisper normalizer source for Latin spans:
  https://github.com/openai/whisper/tree/main/whisper/normalizers

## Limitations

- Apostrophe handling intentionally collapses ambiguous `'s` forms instead of
  expanding every case.
- The node assumes canonical normalized input and should not be used directly
  on raw ASR text.
