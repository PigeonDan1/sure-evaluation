# Canonical DOA JSONL Normalization

## Purpose

`normalization/doa_jsonl` parses, validates, normalizes, and exactly aligns a
pair of canonical frame-wise DOA JSONL files. It does not read audio, perform
coordinate conversion, run a localization model, or parse model-specific
exports.

## Task Scenarios

- All `DOA/any/*` routes use this node as their normalization stage.
- It is the default and only normalization node for the DOA v1 task.

## Input

- Schema: `doa_jsonl_pair`.
- Required roles: `reference_jsonl` and `sample_output`.
- Each row contains `recording_id`, `frame_index`, `timestamp`,
  `num_channels`, `coordinate_system`, `angle_unit`, and `sources`.
- Rows align exactly on `(recording_id, frame_index)`; timestamps must agree
  within the configured tolerance.

## Output

- Schema: `doa_normalized_bundle`.
- Produces normalized reference and prediction frames plus input summaries,
  dimension, alignment, and prediction-activity policy trace fields.

## Versioned Computation

- Node id: `normalization/doa_jsonl`.
- Version: `v1`.
- Supports 2D azimuth and 3D azimuth/elevation rows in degrees using
  `array_local_spherical` coordinates.
- Wraps finite azimuths to `[-180, 180)`, validates source metadata, sorts rows
  by the alignment key, and rejects missing, extra, or duplicate frame keys.

## Runtime and Assets

- Runtime: `in_process`.
- Runs in the base package with no optional dependencies, checkpoints, or
  external assets.

## Source and References

- Source: local SURE canonical-input implementation.

## Limitations

- Only degree-valued `array_local_spherical` coordinates are supported.
- `num_channels >= 2` is required metadata; it does not establish which
  channels a prediction producer used.
- Coordinate and model-output conversion must happen before this node.
