# KOMOS Scoring Node

## Purpose

`scoring/komos` predicts the overall quality of Korean generated speech and
reports the canonical atomic metric `komos` on the calibrated 1--5 MOS scale.

## Route and Input

- Pipeline: `tts.ko.komos.komos_v1`.
- Schema: `generated_audio_rows`.
- Required audio role: `prediction_audio`.
- Aggregation: arithmetic mean over utterance scores; higher is better.

KOMOS loads each audio path, converts it to mono, and resamples it to 16 kHz.
Its default `score()` interface evaluates up to the first 10 seconds.

## Runtime and Assets

- Runtime: optional node-local `uv` project, prepared with Python 3.11.
- Package: `komos==0.0.2`.
- Model: `upskyy/komos` pinned to revision
  `c558983b9777d4f01aa61003b151dea37db0cc2d`.
- GPU: optional; select CPU with `--device cpu`.

After preparing the node environment, `sure-eval env download --node
scoring/komos` stores the pinned model under `checkpoints/komos`. The provider
detects that directory automatically; `KOMOS_MODEL` can be set manually to use
another local model directory. Checkpoints and caches must not be committed.

## Source and Limitations

- Implementation and documentation: https://github.com/upskyy/komos
- Model card: https://huggingface.co/upskyy/komos
- Paper: https://arxiv.org/abs/2610.15353

KOMOS is a no-reference predictor trained on Korean ratings, not a replacement
for a human listening test. This version is intentionally routed only for
Korean TTS audio.
