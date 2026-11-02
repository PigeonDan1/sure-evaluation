# NISQA DIM MOS Node

## Purpose

Predict overall speech quality from enhanced audio without a reference.

## Task Scenarios

- Optional SE metric `nisqa_mos`, outside the default bundle.
- Exact pipeline: `se.any.nisqa_mos.nisqa_mos_v1`.

## Input

- Schema `enhanced_audio_rows`; required `enhanced_audio`, aligned by
  `sample_id`. NISQA's frontend handles audio loading/resampling.

## Output

- Schema `provider_normalized_mos`; `mos_pred` per sample and arithmetic mean.
- MOS-like prediction, higher is better.

## Versioned Computation

- Node `scoring/nisqa_mos` v1 selects the overall output from NISQA v2 DIM.
- The five DIM selectors share the same model instance within a bundle.

## Runtime and Assets

- Optional `pip` runtime: `nisqa==2.0.post2`, PyTorch, pandas.
- Stage `nisqa.tar` outside git via `SURE_SE_NISQA_DIM_CKPT`.
- Code is MIT; model **weights are CC BY-NC-SA 4.0**. Verify permitted use.

## Source and References

- [NISQA](https://github.com/gabrielmittag/NISQA).

## Limitations

- A learned prediction is not a human MOS or an ITU score.
