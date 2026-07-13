# Punctuation Strip Normalization

`normalization/punctuation_strip_norm` removes punctuation only from key-tab-text
files. It does not run TN/ITN, number normalization, case folding, tokenization,
or whitespace compaction.

The node is the default Mandarin TTS semantic normalizer before `scoring/wenet_cer`.
It is intentionally separate from `normalization/aispeech_norm`, whose legacy ASR
behavior combines number text normalization with punctuation stripping.
