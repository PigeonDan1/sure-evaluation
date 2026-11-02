# DNSMOS P.808 Prediction Node

## Purpose

Expose the existing DNSMOS provider's `P808_MOS` score as an independent metric.

## Task Scenarios

- Optional SE metric `dnsmos_p808`, outside the default bundle.
- Exact pipeline: `se.any.dnsmos_p808.dnsmos_p808_v1`.

## Input

- Schema `enhanced_audio_rows`; required `enhanced_audio`, optional
  `noisy_audio`/`reference_audio`; rows aligned by `sample_id`.

## Output

- Schema `provider_normalized_mos`; per-sample predicted P.808 MOS and mean.
- Higher is better.

## Versioned Computation

- Node `scoring/dnsmos_p808` v1 selects `P808_MOS` from the unchanged
  `scoring/dnsmos` v1 provider. Missing/non-finite fields are errors.

## Runtime and Assets

- In-process selector over the existing `scoring/dnsmos` ONNX runtime.
- Prepare that node's model_v8 and sig_bak_ovr assets; no weights are bundled.

## Source and References

- [Microsoft DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS).

## Limitations

- This is a learned prediction, not a crowdsourced P.808 listening test.
- The underlying DNSMOS v1 runtime must be prepared separately.
