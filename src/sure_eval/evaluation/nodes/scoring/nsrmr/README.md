# Normalized SRMR Scoring Node

## Purpose

Score enhanced speech using normalized SRMR without a clean reference.

## Task Scenarios

- Optional SE reverberation-quality metric `nsrmr`, outside the default bundle.
- Exact pipeline: `se.any.nsrmr.nsrmr_v1`.

## Input

- Schema `enhanced_audio_rows`; required `enhanced_audio`, optional
  `noisy_audio` and `reference_audio`; aligned by `sample_id`.
- Audio is downmixed and resampled to 16 kHz.

## Output

- Schema `provider_normalized_mos`, per-sample `nsrmr` and corpus mean.
- Dimensionless modulation-energy ratio, not dB; higher is better.

## Versioned Computation

- Node `scoring/nsrmr` v1 calls
  `srmrpy.srmr(audio, 16000, fast=True, norm=True)[0]`.
- `srmrpy` returns an energy ratio without logarithmic dB conversion.

## Runtime and Assets

- Optional `pip` runtime with `srmrpy==1.0`; no checkpoint.

## Source and References

- [srmrpy](https://pypi.org/project/srmrpy/), as used by the local
  `build_additional_metric_table.py` OpenBench leaderboard script.

## Limitations

- Other SRMR implementations and non-normalized mode give different scores.
- Short or silent inputs may not yield finite scores.
