# NISQA DIM COL Node

## Purpose

Predict the NISQA coloration-quality dimension without a clean reference.

## Task Scenarios

- Optional SE metric `nisqa_col`; pipeline `se.any.nisqa_col.nisqa_col_v1`.

## Input

- Schema `enhanced_audio_rows`; required `enhanced_audio`, `sample_id` alignment.

## Output

- Schema `provider_normalized_mos`; `col_pred` per sample and arithmetic mean;
  MOS-like scale, higher is better.

## Versioned Computation

- Node `scoring/nisqa_col` v1 selects NISQA v2 DIM `col_pred`.

## Runtime and Assets

- Optional `pip`: `nisqa==2.0.post2`, PyTorch, pandas. Set
  `SURE_SE_NISQA_DIM_CKPT` to staged `nisqa.tar`; no weights are committed.
- Model weights are **CC BY-NC-SA 4.0**; verify permitted use.

## Source and References

- [NISQA](https://github.com/gabrielmittag/NISQA).

## Limitations

- Learned prediction, not a human listening score; not in the default bundle.
