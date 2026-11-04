# Duplex timeline normalization v1

## Purpose

`normalize_duplex_timeline(reference_jsonl, sample_output)` validates two JSONL
files and returns a shared `DuplexTimelineBundle` plus a normalization trace.
It is independent of any model, language, audio backend, or timing metric.

## Task Scenarios

All four default `full_duplex/any/{ird,fsed,ert,eit}` atomic routes share this
node. Other timestamp scorers can consume the same normalized bundle.

## Input

Schema: `duplex_timeline_jsonl_pair`, aligned by exact conversation `key`.

Reference rows require `schema_version: 1`, a nonempty `key`, `time_unit: seconds`,
`time_origin: conversation_start`, a positive `duration`, and nonempty
`user_segments`. Prediction rows require the same schema/clock/key fields and
`system_segments`; an empty array is valid silence. Each segment has exactly
`id`, `start`, and `end`. Optional row `metadata` must be an object. Unknown
fields, duplicate keys/IDs, booleans as numbers, nonfinite numbers, negative or
zero-length segments, and segments outside the reference duration are rejected.

Each channel must already be in strictly increasing start order without internal
overlap (adjacent segments are allowed). Cross-channel overlap is essential and
is preserved. Conversation rows are aligned by exact key and sorted by key;
segments are never reordered, merged, clipped, or removed. A missing prediction
row is an error, not silence.

Invalid interval bounds, zero/reversed intervals, start ordering, and overlap
errors identify the conversation key, channel, segment index, and `segment_id`.
Ordering/overlap errors also identify the previous segment. Validation rejects
these inputs without repairing the timeline; exception types remain unchanged.

## Output

Schema: `duplex_timeline_bundle`. Shared immutable records contain exact seconds,
user/system segments, and provenance metadata. The normalization trace records
input counts and clock conventions; it does not produce a score.

## Versioned Computation

Node `normalization/duplex_timeline`, version `v1`. Internal stages are exact
decimal JSONL parsing, schema/clock validation, channel validation, and exact
conversation key alignment.

JSON decimal literals are parsed as `Decimal`; traces serialize exact seconds
as strings. This node does not quantize to an audio sample grid. A downstream
scorer may enforce its own grid. Metadata has no scoring effect.

Optional `metadata.measurement_protocol` uses a validated, descriptive schema
for VAD, playback, reference, preamble, capture, and notes. Missing fields stay
valid; supplied fields use strict types and checked finite/range values.
VAD numeric parameters must fit the finite binary64 export range without
nonzero underflow; exact integer values stay integers. Timestamp precision
and the treatment of other metadata are unchanged.
Unknown fields inside this optional object are rejected; surrounding metadata
remains extensible. A separate summary records per-role values and
`recorded`/`partial`/`missing` coverage without changing segments or scores.
See the [measurement schema](../../../../../../docs/tasks/full_duplex.md#optional-measurement-provenance).

## Runtime and Assets

`in_process`, required, CPU only. Python standard library and SURE base package;
no API keys, optional packages, model checkpoints, or downloaded assets.

## Source and References

Local canonical timeline contract. See the [task contract](../../../tasks/full_duplex/README.md)
and [FD-Bench](https://github.com/pengyizhou/FD-Bench) for the downstream scenario.

## Limitations

This node accepts saved timestamps only. It does not establish that two clocks
were physically synchronized or that a VAD detector measured true speech.
It does not infer units, repair recordings, run VAD, or remove model preambles.

See [task contract](../../../tasks/full_duplex/README.md) for an input example.
