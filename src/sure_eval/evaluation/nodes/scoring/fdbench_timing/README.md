# FD-Bench Timing

## Purpose

Compute IRD, FSED, ERT and EIT from validated conversation timelines using the
fixed `fdbench_8a4b7df_compat` operational protocol. A single traversal produces
all four metrics and an auditable event ledger.

## Task Scenarios

- All four `full_duplex/any/{ird,fsed,ert,eit}` atomic routes select this node.
- The route's metric selects the primary score; `results` always contains all four.

## Input

`duplex_timeline_bundle` from `normalization/duplex_timeline`: ordered user and
system segments with Decimal endpoints in seconds relative to conversation start.
Endpoints must lie exactly on the 16000 tick/s grid; values are never silently
rounded. Cross-channel overlaps are essential input evidence.

## Output

`fdbench_timing_report` in `PipelineNodeResult.details` contains the selected
`score`, four `results`, `evaluation_config`, `per_conversation` raw slots and
`events`. Each metric reports ms samples, median, count, zero-ms count and status.
An empty pool yields `0.0` with `sample_count=0` and `status=no_events`; it is not
a measured zero latency. A positive sub-ms raw delta can legitimately produce
`0ms`, `sample_count=1`, `status=ok`.

The full-duplex task stores this detailed output once under its top-level
`details.results`, `details.per_conversation`, and `details.events`, while the
task's scoring trace is a compact summary. The standalone scoring node still
returns the complete detailed result. Measurement provenance is descriptive
task metadata and never changes this node's computation.

IRD and FSED use the paper's lower-is-better interpretation together with event
counts. ERT/EIT have no universally declared optimization direction. No composite
score or shared success-rate denominator is introduced.

## Versioned Computation

- Node `scoring/fdbench_timing`, version `v1`.
- Internal stages: exact tick conversion, protocol traversal, slot selection,
  positive filtering, integer ms conversion, pooled median, event diagnostics.
- EIT requires `user.start < system.start < user.end - 8000`; exactly 500ms early
  belongs to ERT. Equality with the user start/end is excluded from start branches.
- Ordinary early responses overwrite their per-user ERT slot; ordinary late
  responses retain their first FSED candidate. EIT retains multiple candidates.
- FSED pools ordinary and interrupted response-delay arrays; ERT pools ordinary
  and interrupted lead-time arrays. Positive tick deltas are individually
  truncated to integer ms before the dataset median. Conversation medians are
  never averaged.
- Source-compatible IRD can use the early system segment's duration, the old
  speech end minus user start, or a new reply's end minus user start. Final-user
  old-speech handling differs from earlier users. These rules intentionally
  preserve the fixed implementation rather than replace it with a new matcher.
- No system segment is globally consumed. The source's unreachable non-final
  old-speech branch remains unreachable. Counter-only source diagnostics and the
  noise-count pass do not affect these four metrics and are omitted.
- Replaced candidates remain in the event ledger with `selected=false` and an
  exclusion reason; selected candidates identify the actual difference endpoints.

## Runtime and Assets

`in_process`, base Python runtime, no optional packages, model weights, GPU or
API key. Environment setup does not create a virtual environment for this node.

## Source and References

Independent local compatibility implementation of the operational behavior in
[FD-Bench's fixed source](https://github.com/pengyizhou/FD-Bench/blob/8a4b7df1b4dcb0fc50a7a5660247a2ffe8d394eb/benchmark/benchmarking.py),
method `analyze_VAD_interruption_new2` and timing aggregation in `analyze_VAD_marks`.
The source commit and LF-normalized SHA256 are recorded in every scoring trace.
No upstream source or benchmark recordings are bundled. See also the
[task documentation](../../../../../../docs/tasks/full_duplex.md).

## Limitations

This node scores supplied observable speech timestamps, not network packet
arrival or cancellation-event latency. Timeline generation and VAD settings must
be controlled externally. Silence produces no timing events and must not be
ranked as ideal latency. Observation duration adds diagnostics but does not
introduce censoring or missing-response penalties. No implicit Moshi preamble
removal is applied: reproductions must supply the same effective preprocessing
explicitly. Source-compatible timing is not a new semantic definition of each
metric; changes to the matching rules require a new protocol/version.
The ordered traversal costs `O(U*A)` per conversation for `U` user and `A` system
segments. The detailed candidate ledger can also contain `O(U*A)` entries;
large conversations therefore produce correspondingly large reports.
