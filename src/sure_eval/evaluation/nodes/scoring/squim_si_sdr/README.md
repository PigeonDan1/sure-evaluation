# SQUIM SI-SDR Prediction Node

## Purpose

Predict SI-SDR without clean reference audio; this is not measured SI-SDR.

## Task Scenarios

- Optional SE metric `squim_si_sdr`, outside the default bundle.
- Exact pipeline: `se.any.squim_si_sdr.squim_si_sdr_v1`.

## Input

- Schema `enhanced_audio_rows`; required `enhanced_audio`; aligned by
  `sample_id`. Audio is downmixed/resampled to 16 kHz.

## Output

- Schema `provider_normalized_mos`; predicted SI-SDR in dB per sample/mean.
- Higher is better.

## Versioned Computation

- Node `scoring/squim_si_sdr` v1 selects the **third** TorchAudio 2.2.0
  SQUIM_OBJECTIVE DNS2020 output; corpus aggregation is a sample mean.

## Runtime and Assets

- Optional `pip` runtime: `torch==2.2.0`, `torchaudio==2.2.0`.
- Stage official `squim_objective_dns2020.pth` outside git and set
  `SURE_SE_SQUIM_OBJECTIVE_CKPT`. Objective weights are CC BY 4.0.

## Source and References

- [TorchAudio SQUIM objective bundle](https://pytorch.org/audio/2.2.0/pipelines.html).

## Limitations

- Its predictions are not interchangeable with full-reference `si_sdr`.
