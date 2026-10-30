# eSTOI Scoring Node

## Purpose

Compute extended STOI for enhanced speech against clean speech. It is distinct
from the existing standard STOI route.

## Task Scenarios

- Optional SE intelligibility metric `estoi`, outside the default bundle.
- Exact pipeline: `se.any.estoi.estoi_v1`.

## Input

- Schema `speech_enhancement_audio_pairs`; required `enhanced_audio` and
  `reference_audio`, optional `noisy_audio`, aligned by `sample_id`.
- Mono 16 kHz audio; pairs are truncated to the shorter length.

## Output

- Schema `full_reference_audio_metric`, per-sample `estoi` and corpus mean.
- Unitless; higher is better.

## Versioned Computation

- Node `scoring/estoi` v1 calls `pystoi.stoi(ref, enhanced, 16000,
  extended=True)` and averages sample scores.

## Runtime and Assets

- Optional `pip` runtime with `pystoi>=0.4.1`; no checkpoint.
- Install with `pip install -e '.[se]'` in the evaluation runtime.

## Source and References

- [pystoi](https://github.com/mpariente/pystoi).

## Limitations

- Requires clean reference audio. Short or silent speech may be undefined.
- An objective intelligibility proxy is not a human listening score.
