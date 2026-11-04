# Full-duplex timing task

This task evaluates saved user/system speech timelines. It does not invoke a
speech model, synthesize audio, detect voice activity, or use an LLM judge.
Model-specific recording and export happen before scoring.

The four atomic metrics `ird` (default), `fsed`, `ert`, and `eit` share:

`normalization/duplex_timeline` -> `scoring/fdbench_timing` -> `report.json`

Each route selects one primary `report.score` and one `execution_metrics`
selector. Every run computes all four results once and retains them under
`details.results`, with millisecond samples, counts, and empty-event status.

Route discovery marks only IRD as `default: true`. With
`sure-eval metric routes full_duplex --metric fsed --json`, the FSED route has
`default: false` and `execution_metrics: ["fsed"]`. The response preserves the
IRD task default in `default_pipeline_id` and `default_pipeline_ids`, including
when the metric filter excludes IRD from the returned routes.

## Input contract

Both inputs are JSONL with one complete conversation per row and identical key
sets. Both require `schema_version: 1`, a nonempty `key`, `time_unit: "seconds"`,
and `time_origin: "conversation_start"`. The reference supplies positive
`duration` and nonempty `user_segments`; the prediction supplies
`system_segments`, which may be empty for a silent system.

```json
{"schema_version":1,"key":"case-A","time_unit":"seconds","time_origin":"conversation_start","duration":10,"user_segments":[{"id":"u1","start":1,"end":3},{"id":"u2","start":6,"end":8}]}
```

```json
{"schema_version":1,"key":"case-A","time_unit":"seconds","time_origin":"conversation_start","system_segments":[{"id":"a1","start":2.75,"end":6.25},{"id":"a2","start":8.4,"end":9}]}
```

Segments need unique nonempty IDs within each channel, strictly increasing
starts, positive lengths, and bounds inside the reference duration. Same-channel
overlap is rejected; cross-channel overlap is preserved. Inputs must already
share the physical conversation clock. Optional `metadata` records provenance
without affecting scores. Endpoints must map exactly to the compatibility
protocol's 16,000 ticks per second; the scorer rejects off-grid values.

Optional `metadata.measurement_protocol` records VAD, playback, reference,
preamble, capture, and notes with checked field types. Partial or missing
objects remain valid v1 input and trigger a summary warning; invalid supplied
types are rejected. Reference and prediction provenance are kept separately.
The report marks coverage as `recorded`, `partial`, or `missing`; recorded
settings are not independently verified. Scores without recorded measurement
settings must not be compared directly. See the
[public task guide](../../../../../docs/tasks/full_duplex.md#optional-measurement-provenance)
for the schema, completeness rule, and an example.

## Protocol and interpretation

`fdbench_8a4b7df_compat` follows the active FD-Bench timing procedure at commit
`8a4b7df1b4dcb0fc50a7a5660247a2ffe8d394eb`, including its strict boundary
conditions, 500 ms split, stateful slot selection, and special IRD branches.
It does not silently replace those branches with intuitive onset/offset
differences. Model-specific preamble removal must be explicit in the exporter;
metadata such as model name never causes automatic segment removal.

All positive raw selected deltas are pooled across conversations. Each delta is
truncated to integer milliseconds before the corpus median is computed. Empty
pools retain the official numeric score `0.0` while reporting `sample_count: 0`
and `status: "no_events"`; this is not evidence of perfect behavior. A positive
sub-millisecond event can also yield numeric zero with a nonempty sample pool.
ERT and EIT are descriptive timing quantities, not a universal lower-is-better
quality score. There is no composite score, SRR, SIR, or WER in this task.

Score-affecting protocol changes require a new node/route version. Existing
pipeline IDs do not automatically hash protocol/configuration contents.

## Report layout

`details.results` contains all four pooled results; `details.per_conversation`
contains per-conversation raw slots and statistics, and `details.events` is
the candidate ledger. `details.evaluation_config` and `details.input_sha256`
identify the rules and input bytes. Both node traces are compact summaries,
with no `details.scoring` or `details.normalization` duplicate. Measurement objects
live in `details.measurement_protocol`; other metadata is retained once per
conversation. The CLI summary distinguishes an empty compatibility zero from a
measured zero using `sample_count`, `status`, and `score_is_measured`.

## CLI

Run from the repository root after installing SURE:

```powershell
sure-eval metric routes full_duplex --json
sure-eval metric describe full_duplex --metric ird --output full_duplex_pipeline.json
sure-eval env setup --pipeline full_duplex_pipeline.json --dry-run
sure-eval env check --pipeline full_duplex_pipeline.json
sure-eval metric run --pipeline full_duplex_pipeline.json `
  --reference-jsonl tests/fixtures/full_duplex/reference.jsonl `
  --sample-output tests/fixtures/full_duplex/sample_output.jsonl `
  --output-dir .sure-eval-demo/full-duplex --validate-env --json
```

Use `--metric fsed`, `ert`, or `eit` when describing the corresponding primary
route. Pipeline JSON records an atomic identity; it is not a four-metric bundle.
Input SHA-256 values and protocol configuration are recorded in the report.
