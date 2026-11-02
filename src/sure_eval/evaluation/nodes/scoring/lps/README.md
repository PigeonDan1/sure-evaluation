# Levenshtein Phoneme Similarity Node

## Purpose

Apply URGENT 2025 phoneme similarity to enhanced and clean speech. This is an
ASR-based intelligibility proxy, not a listening score.

## Task Scenarios

- Optional SE metric `lps`, outside the default bundle.
- Exact pipeline: `se.any.lps.lps_v1`.

## Input

- Schema `speech_enhancement_audio_pairs`; required `enhanced_audio` and
  `reference_audio`, optional `noisy_audio`; aligned by `sample_id`.
- Audio is downmixed/resampled to 16 kHz; waveform lengths need not match.

## Output

- Schema `full_reference_audio_metric`, per-sample `lps`, phoneme strings,
  and arithmetic corpus mean. Unitless, potentially negative; higher is better.

## Versioned Computation

- Node `scoring/lps` v1 decodes both phoneme strings using the pinned
  `facebook/wav2vec2-lv-60-espeak-cv-ft` model, removes spaces, and scores
  `1 - Levenshtein(enhanced, reference) / len(reference)`.
- Denominator: the predicted clean-reference phoneme-string length.

## Runtime and Assets

- Optional `pip` runtime with `transformers==4.24.0`, PyTorch, soundfile and
  soxr. This version loads the published legacy weight-norm checkpoint.
- Stage model revision `ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4`
  outside git and set `SURE_SE_LPS_MODEL_DIR` to its directory.

## Source and References

- [URGENT 2025 phoneme similarity](https://github.com/urgent-challenge/urgent2025_challenge/blob/main/evaluation_metrics/calculate_phoneme_similarity.py).

## Limitations

- Empty reference phoneme strings are rejected. Scores depend on the pinned
  recognizer and decoding implementation.
