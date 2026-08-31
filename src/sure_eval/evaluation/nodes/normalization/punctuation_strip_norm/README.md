# Punctuation Strip Normalization

## Purpose

`normalization/punctuation_strip_norm` removes punctuation from key-text files
without applying TN, ITN, number normalization, case folding, tokenization, or
whitespace compaction.

The node is intentionally narrower than `normalization/site_norm`.

## Task Scenarios

- Mandarin TTS semantic CER before `scoring/wenet_cer`.
- Mandarin VC/TSE semantic CER routes that need punctuation removal without
  rewriting digits, Latin text, or Chinese text.

## Input

- Schema: `key_text_files`.
- Row format: `<key><TAB><text>`.
- Required roles: usually `ref`, `hyp` after transcription.

## Output

- Schema: `key_text_files`.
- Output rows preserve keys and remove punctuation from text.

## Versioned Computation

- Node id: `normalization/punctuation_strip_norm`.
- Version: `v1`.
- Profile: `unicode_category_p_or_ascii`.
- Removes:
  - Unicode categories whose name starts with `P`
  - ASCII `string.punctuation` characters
- Preserves:
  - numbers
  - case
  - whitespace
  - non-punctuation symbols outside ASCII punctuation

## Runtime and Assets

- Runtime: `in_process`.
- No optional packages, checkpoints, or external binaries.

## Source and References

- Source: local SURE normalization implementation based on Python standard
  Unicode category handling.

## Limitations

- It does not normalize full-width and half-width variants.
- It does not canonicalize spoken and written numbers.
