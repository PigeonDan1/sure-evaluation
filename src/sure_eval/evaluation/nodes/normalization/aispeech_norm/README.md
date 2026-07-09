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
