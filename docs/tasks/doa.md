# DOA — Direction of Arrival

DOA is a model-independent evaluator for already generated microphone-array
localization results. It does not read multichannel audio or run a model.

## Pipeline

All registered metrics use:

`normalization/doa_jsonl` -> `scoring/doa_localization` -> `report.json`

The public route is intentionally compact. Input validation, alignment,
activity handling, assignment, gating, and aggregation are internal stages of
the two SURE Nodes and are recorded in the pipeline trace.

## Standard Input

The required roles are `reference_jsonl` and `sample_output`. Each file has one
frame per JSONL row:

```json
{"recording_id":"rec-1","frame_index":0,"timestamp":0.0,"num_channels":4,"coordinate_system":"array_local_spherical","angle_unit":"degree","sources":[{"source_id":"s0","azimuth":30.0,"activity":true},{"source_id":"s1","azimuth":100.0,"activity":true}]}
```

Required row fields are `recording_id`, `frame_index`, `timestamp`,
`num_channels >= 2`, `coordinate_system`, `angle_unit`, and `sources`.
`azimuth` is required for each source. `elevation` is required only with
`--dimension 3d`; a 2D row must not provide elevation. Both files must use the
same coordinate convention, and the evaluator currently supports only
`array_local_spherical` in degrees. Coordinate conversion is not implemented.

Source `activity` defaults to true when omitted. `presence` and `explicit`
prediction activity policies are supported; there is no hard-coded VAD
threshold. `timestamp` is checked after exact frame-index alignment, with
`--timestamp-tolerance-sec` controlling the allowed difference.

`source_id` is optional trace metadata and is not used for identity matching.
`confidence` is also optional metadata and is not consumed by the scorer. Under
the default `presence` policy every listed prediction is active even if it says
`"activity": false`; `explicit` filters predictions by that boolean. Reference
sources always use their activity value. Empty `sources` means no active source.

Rows are sorted by `(recording_id, frame_index)` before alignment, so file order
does not affect results. The two files must have exactly the same key set;
missing or extra prediction frames are rejected. `frame_index` is primary and
`timestamp` is a tolerance-checked consistency field.

The `array_local_spherical` convention is right-handed: +x is array-forward,
+y is left, +z is up, azimuth is `atan2(y, x)`, and elevation is the angle from
the horizontal plane toward +z. Dataset/model exporters must perform any needed
coordinate conversion before evaluation.

## Metrics

The default aggregation is matched-pair `micro`. `macro_recording` can be
selected explicitly.

For a matched error set `E`:

```text
MAE  = sum(E) / len(E)
RMSE = sqrt(sum(e^2 for e in E) / len(E))
```

After assignment, a pair is TP exactly when `error_deg < threshold_deg`:

```text
Recall    = TP / active_reference_sources
Precision = TP / active_prediction_sources
F1        = 2 * Precision * Recall / (Precision + Recall)
```

`acc_10`, `acc_15`, and `acc_30` use the corresponding fixed threshold and
the active-reference denominator. Unmatched references and predictions are
reported as misses and false predictions. Assignment is minimum-total-angular
cost and uses a pure-Python Hungarian algorithm with cubic time complexity in
the number of sources. It is intentionally performed before gating and does not
maximize gated TP count. A paired error outside the gate stays in MAE/RMSE and
counts as both one miss and one false prediction for thresholded metrics.

`acc_*` and localization recall use the same `TP / active_reference_sources`
formula. Micro MAE/RMSE pool assigned pairs, while micro thresholded metrics
pool counts. Macro recording is the arithmetic mean of defined per-recording
scores, including the arithmetic mean of per-recording RMSE values rather than
a pooled RMSE. The report states how many recordings were undefined and
excluded. Zero denominators and no-pair MAE/RMSE are represented as JSON `null`,
not zero or NaN. When precision and recall are both defined but both are zero,
F1 is `0.0` rather than `null`.

The pipeline ID identifies the metric and Node versions. Runtime configuration
is recorded separately under `details.evaluation_config`, the scoring trace
records both configured and effective thresholds, and `details.input_sha256`
records the exact input bytes.

## Run

```powershell
sure-eval metric describe doa --metric mae --output doa_pipeline.json
sure-eval metric run --pipeline doa_pipeline.json `
  --reference-jsonl examples/readme/doa_reference.jsonl `
  --sample-output examples/readme/doa_prediction.jsonl `
  --dimension 2d --threshold-deg 10 --aggregation micro `
  --output-dir doa_eval
```

Model-specific outputs must be converted outside SURE into this standard
JSONL. Export conversion is an experiment or data-preparation step, not a
formal DOA Node or Route.
