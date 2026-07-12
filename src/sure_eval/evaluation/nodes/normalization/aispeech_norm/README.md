# AISpeech ASR Normalization

This node wraps the ASR text normalization behavior that was previously embedded
inside `SUREEvaluator._eval_asr` and `SUREEvaluator._eval_asr_codeswitch`.

It intentionally does not change the existing normalization implementation. The
node records the normalization version in pipeline reports so ASR scores can be
compared with their text-normalization backend visible.

Profiles:

- `en`: AISpeech number text normalization plus punctuation stripping.
- `zh`: AISpeech number text normalization plus punctuation stripping.
- `cs`: AISpeech code-switch tokenization, language split, and number text
  normalization.

Input handling:

- Rows are `<key>\t<text>`. A row whose text is empty is a valid recognition
  result and is preserved, so downstream scoring counts it as deletions instead
  of silently dropping the utterance.
- Non-empty lines without a tab separator are dropped and counted in the node
  trace under `row_stats`. If a file yields no parseable rows at all, the node
  raises instead of producing an empty (and trivially perfect) evaluation.
- Normalization map tables are loaded once per node call and shared across
  rows (with a shared number-conversion cache), instead of re-globbing and
  re-reading map files for every line/token. Output is identical; the
  per-call fallback inside `asr_num2words` remains for legacy callers.
