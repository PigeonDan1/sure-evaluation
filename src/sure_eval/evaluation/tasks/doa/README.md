# DOA Task

This is a model-independent evaluator for already generated direction-of-arrival
predictions. It does not read waveforms, run a DOA model, or parse a
model-specific export format.

## Route

All registered metrics use the same SURE node chain:

`normalization/doa_jsonl` -> `scoring/doa_localization` -> `report.json`

Registered metrics are `mae`, `rmse`, `acc_10`, `acc_15`, `acc_30`,
`localization_recall`, `localization_precision`, and `localization_f1`.
The default aggregation is `micro`; `macro_recording` is available as an
explicit run option.

## Standard JSONL Contract

Each reference and prediction file contains one JSON object per line. Every row
must contain:

- `recording_id`: non-empty recording identifier;
- `frame_index`: non-negative integer frame key;
- `timestamp`: non-negative seconds, checked against the matching row;
- `num_channels`: integer >= 2, recording array metadata;
- `coordinate_system`: currently exactly `array_local_spherical`;
- `angle_unit`: currently exactly `degree`;
- `sources`: list of zero or more source objects.

Each source requires `azimuth` in degrees. `source_id` defaults to its position
inside the row, `activity` defaults to `true`, and `confidence` is optional
metadata that does not affect scoring. `source_id` is used only for traceability;
matching is frame-local and does not match identities. In 3D mode, `elevation`
is required and must be in `[-90, 90]`.
In 2D mode, an elevation must not be supplied. Reference and prediction rows
must have the same `(recording_id, frame_index)` key set, channel metadata, and
coordinate system. JSONL row order is normalized and does not affect the score;
missing or extra frame keys are rejected rather than treated as misses or false
predictions. `frame_index` is the primary alignment key and `timestamp` is a
consistency check within the configured tolerance.

`array_local_spherical` is a right-handed array-local convention: +x is the
array forward direction, +y is left, +z is up, azimuth is `atan2(y, x)`, and
elevation is measured from the horizontal plane toward +z. The evaluator does
not perform coordinate-system conversion; the producer must export both files
in this convention.

`num_channels` is required to identify a multi-channel evaluation input, but it
is not proof that a model actually used every microphone. The evaluator never
opens the WAV file.

## Scoring Protocol

2D uses circular azimuth distance. 3D uses spherical great-circle distance.
For every frame, active reference and prediction sources are matched by a
minimum-total-angular-cost one-to-one assignment. The assignment is computed
before gating and is independent of the configured threshold. A pair is a true
positive only when `error_deg < threshold_deg`. This objective minimizes raw
angular error; it does not maximize the number of pairs inside the gate. An
assigned pair outside the gate remains in MAE/RMSE and contributes one miss and
one false prediction to thresholded counts.

- `mae`: mean of all assigned angular errors;
- `rmse`: square root of the mean squared assigned angular errors;
- `acc_10`, `acc_15`, `acc_30`: TP divided by active reference count at the
  fixed threshold;
- `localization_recall`: TP / active reference count;
- `localization_precision`: TP / active prediction count;
- `localization_f1`: harmonic mean of precision and recall.

`acc_10`, `acc_15`, and `acc_30` are threshold-fixed metrics. `acc_*` and
`localization_recall` use the same formula; the different names distinguish a
fixed-threshold route from configurable-threshold localization recall.

Unmatched references are misses and unmatched predictions are false
predictions. MAE/RMSE describe assigned localization quality and therefore do
not silently turn an unmatched source into an angular error; recall, precision,
F1, and the report counters expose that detection failure.

For `micro`, MAE/RMSE pool assigned pairs and thresholded metrics pool counts
over the dataset. For `macro_recording`, the requested per-recording metric is
averaged over recordings where it is defined; excluded counts are reported.
MAE/RMSE are `null` without assigned pairs, recall/accuracy are `null` when the
active-reference denominator is zero, and precision is `null` when the active-
prediction denominator is zero. F1 is `null` if either component is undefined.
When both components are defined and both are zero, F1 is `0.0`.

The pipeline ID records the metric and Node versions. Score-affecting runtime
settings and the effective threshold are recorded under `evaluation_config` and
the scoring trace, and input file SHA-256 values are recorded in `report.json`.

## CLI

```powershell
sure-eval metric describe doa --metric mae --output doa_pipeline.json
sure-eval metric run --pipeline doa_pipeline.json `
  --reference-jsonl examples/readme/doa_reference.jsonl `
  --sample-output examples/readme/doa_prediction.jsonl `
  --dimension 2d --threshold-deg 10 --aggregation micro `
  --output-dir doa_eval
```

Model-specific outputs must be converted outside the Task into this schema.
That conversion is an experiment/data-preparation step, not a formal DOA Node
or Route.
