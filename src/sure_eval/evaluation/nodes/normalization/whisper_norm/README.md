# Whisper Normalization

`normalization/whisper_norm` normalizes key-tab-text ASR files with the
OpenAI Whisper text normalizers before WeNet edit-distance scoring.

The vendored implementation comes from `openai-whisper==20250625`,
`whisper/normalizers`, licensed under MIT. The node intentionally vendors only
the normalizer files instead of depending on the full Whisper package. The only
local change is replacing `more_itertools.windowed` with an equivalent private
triple-window helper so this in-process node does not add a runtime dependency.

Profiles:

- `english`: Whisper `EnglishTextNormalizer`; default for English WER.
- `basic`: Whisper `BasicTextNormalizer`; available for explicit future routes,
  but not a default for multilingual evaluation.

ASR Chinese CER and code-switch MER continue to use `normalization/aispeech_norm`
by default. TTS Mandarin CER uses `normalization/punctuation_strip_norm`.
