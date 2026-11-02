# DNSMOS P.835 BAK Prediction Node

## Purpose

Expose the existing DNSMOS provider's calibrated background-quality dimension.

## Task Scenarios

- Optional SE metric `dnsmos_bak`, outside the default bundle.
- Exact pipeline: `se.any.dnsmos_bak.dnsmos_bak_v1`.

## Input

- Schema `enhanced_audio_rows`; required `enhanced_audio`, optional
  `noisy_audio`/`reference_audio`; aligned by `sample_id`.

## Output

- Schema `provider_normalized_mos`; per-sample `BAK` and arithmetic mean.
- Higher is better.

## Versioned Computation

- Node `scoring/dnsmos_bak` v1 selects calibrated `BAK` from the unchanged
  `scoring/dnsmos` v1 provider; missing/non-finite values are errors.

## Runtime and Assets

- In-process selector over the existing `scoring/dnsmos` ONNX runtime.
- Prepare its sig_bak_ovr and model_v8 assets outside git.

## Source and References

- [Microsoft DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS).

## Limitations

- This predicts a P.835-style dimension; it is not a human P.835 test.
- The underlying DNSMOS v1 runtime must be prepared separately.
