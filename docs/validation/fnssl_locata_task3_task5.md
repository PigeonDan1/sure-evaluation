# FN-SSL Validation on LOCATA Task 3 and Task 5

## Objective

Validate the model-independent SURE DOA task with real predictions from a real
localization model, while keeping model inference and export conversion outside
the formal evaluator. The shared comparison targets are MAE and ACC@30 only.

## Dataset

The validation uses the DICIT array recordings below.

| LOCATA subset | Recording | Evaluation frames | Active reference frames |
|:--|:--|--:|--:|
| Task 3 | `recording1/dicit` | 121 | 66 |
| Task 5 | `recording1/dicit` | 125 | 90 |
| Total | - | 246 | 156 |

## Model

Predictions were produced by the FN-SSL DOA classification path using the
`lightning_doa.ckpt` checkpoint. Model inference was completed before this
validation record was written and is not part of the SURE pipeline.

## Checkpoint Configuration

| Setting | Value |
|:--|:--|
| Sampling rate | 16000 Hz |
| Window length | 512 samples |
| FFT size | 512 |
| Window shift ratio | 0.5 |
| Channel mode | `MM` |
| Maximum sources | 1 |
| Classification outputs | 180 |

## FN-SSL Segmentation

The FN-SSL run produced 121 evaluation frames for Task 3 and 125 for Task 5.
With a window shift ratio of 0.5 and a 512-sample window, the configured hop is
256 samples. Recording boundaries were preserved: per-recording metrics were
computed independently before the two recordings were averaged for the macro
result.

## Frozen Inference Outputs

The inference predictions, ground-truth angles, and ground-truth activity used
for this comparison were frozen after the successful FN-SSL run. They are
validation inputs, not repository runtime assets, and no checkpoint, LOCATA
audio, NumPy array, or generated report is committed with the DOA task.

## Standard JSONL Conversion

The frozen reference and prediction data were converted in an external
experiment step to the canonical DOA JSONL contract. Each aligned row records
the recording id, frame index, timestamp, channel metadata,
`array_local_spherical` coordinate system, degree angle unit, and source list.
This conversion is not a SURE Node or Route.

## Evaluation Protocol

- Dimension: 2D circular azimuth.
- Assignment: frame-local minimum-total-angular-cost one-to-one matching.
- Gating: assignment precedes gating and the comparison is strict
  `error_deg < threshold_deg`.
- SURE prediction activity policy: `presence`.
- Shared FN-SSL comparison metrics: MAE and ACC@30.
- SURE results are reported with both `micro` and `macro_recording`
  aggregation.

## Per-Recording Results

| Recording | Active references | FN-SSL MAE (deg) | ACC@30 |
|:--|--:|--:|--:|
| Task 3 `recording1/dicit` | 66 | 4.433853842995384 | 1.0 |
| Task 5 `recording1/dicit` | 90 | 5.613924577501085 | 1.0 |

The equal-weight macro-recording FN-SSL MAE is
`5.023889210248234` degrees. The original classification evaluator displayed
this as approximately `5.0239`.

## FN-SSL and SURE Comparison

Under the aligned protocol, SURE reproduces the FN-SSL MAE and ACC@30 results
on the real LOCATA Task 3 and Task 5 predictions.

| Shared metric | FN-SSL | SURE | Aggregation |
|:--|--:|--:|:--|
| MAE (deg) | 5.023889210248234 | 5.023889210248234 | macro recording |
| ACC@30 | 1.0 | 1.0 | macro recording |

This comparison does not claim that every SURE metric has an FN-SSL baseline.

## Additional SURE Metrics

| Metric | Macro recording | Micro |
|:--|--:|--:|
| MAE | 5.023889210248234 | 5.114663882133288 |
| RMSE | 5.824752639383391 | 5.968532318927137 |
| ACC@10 | 0.9515151515151515 | 0.9487179487179487 |
| ACC@15 | 0.9757575757575758 | 0.9743589743589743 |
| ACC@30 | 1.0 | 1.0 |
| Localization Recall | 0.9515151515151515 | 0.9487179487179487 |
| Localization Precision | 0.6004628099173555 | 0.6016260162601627 |
| Localization F1 | 0.7329436637234176 | 0.736318407960199 |

The micro threshold counts are 148 of 156 active references within 10 degrees,
152 of 156 within 15 degrees, and 156 of 156 within 30 degrees.

## Micro and Macro Recording

Micro aggregation pools assigned errors and TP/FP/FN counts across both
recordings before computing each metric. Macro-recording aggregation computes
the metric separately for Task 3 and Task 5 and then gives the two defined
recording scores equal weight. Their different weighting explains the expected
difference between the micro and macro values.

## Precision and Recall Interpretation

The FN-SSL classification path emitted one DOA prediction for every evaluation
frame: 246 predictions in total, while only 156 frames contained an active
reference source. SURE's `prediction_activity_policy=presence` therefore counts
predictions on inactive-reference frames as false positives. This produces high
micro recall (approximately 0.949), lower micro precision (approximately
0.602), and micro F1 of approximately 0.736. This is expected protocol
behavior, not a scoring error.

## Known Protocol Difference

The FN-SSL reference evaluator masks predictions with ground-truth VAD, so it
does not expose false positives on inactive-reference frames in the same way.
MAE and ACC@30 can be compared directly after aligning the protocol. Precision,
recall, F1, RMSE, ACC@10, and ACC@15 are additional SURE results without a
corresponding FN-SSL baseline in this validation and must not be described as
independently cross-validated shared metrics.

## Validation Conclusion

**PASS — real-data, real-model validation completed successfully for shared MAE and ACC@30 metrics.**
