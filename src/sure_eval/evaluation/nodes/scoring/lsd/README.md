# Log-Spectral Distance Scoring Node

## Purpose

Measure the local OpenBench speech-enhancement benchmark's full-reference LSD
definition. This is one versioned convention, not a universal LSD standard.

## Task Scenarios

- Optional SE spectral distortion metric `lsd`, outside the default bundle.
- Exact pipeline: `se.any.lsd.lsd_v1`.

## Input

- Schema `speech_enhancement_audio_pairs`; required `enhanced_audio` and
  `reference_audio`, optional `noisy_audio`; aligned by `sample_id`.
- Downmix to mono, resample with librosa to 16 kHz, truncate to shorter audio.

## Output

- Schema `full_reference_audio_metric`, per-sample `lsd` and corpus mean.
- Units: dB. Lower is better.

## Versioned Computation

- Node `scoring/lsd` v1: centered librosa STFT, 512-sample FFT/window,
  128-sample hop, magnitude floor `1e-7`, then `20 log10` amplitudes.
- `sqrt(mean((enhanced_db - reference_db)^2))` reduces all time-frequency bins
  per utterance; utterance scores are averaged.

## Runtime and Assets

- Optional `pip` runtime with `librosa>=0.10.0` (`audio` extra); no checkpoint.

## Source and References

- Local benchmark definition: `build_additional_metric_table.py` in the
  AutoResearch evaluation workspace; [librosa STFT](https://librosa.org/doc/main/generated/librosa.stft.html).

## Limitations

- Other LSD implementations average frame RMS values or use different STFTs;
  their scores are not directly comparable to this route.
- Requires clean reference audio.
