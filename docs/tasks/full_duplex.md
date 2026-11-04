# FULL_DUPLEX — Full-Duplex Dialogue Timing

Evaluate saved user/system speech intervals using four FD-Bench timing
metrics. The task consumes timestamps; it does not call a model, transcribe
speech, run VAD, or judge response content. A model API is an optional producer
of test recordings, outside this scoring route.

```bash
sure-eval metric routes full_duplex --metric ird
```

## Metrics and routes

| Canonical metric | Pipeline ID | Interpretation |
|:--|:--|:--|
| `ird` | `full_duplex.any.ird.duplex_timeline_v1.fdbench_timing_v1` | interrupt-response delay, ms |
| `fsed` | `full_duplex.any.fsed.duplex_timeline_v1.fdbench_timing_v1` | first-speech-emit delay, ms |
| `ert` | `full_duplex.any.ert.duplex_timeline_v1.fdbench_timing_v1` | early-reply time, ms |
| `eit` | `full_duplex.any.eit.duplex_timeline_v1.fdbench_timing_v1` | early-interrupt time, ms |

The default metric is `ird`. Each route is atomic and has a singleton
`execution_metrics`, such as `["ird"]`. All routes reuse
`normalization/duplex_timeline` → `scoring/fdbench_timing`. The scorer calculates
the four metrics together; the selected metric supplies the report's primary
`score`, and all four remain in `details.results`. This is not a bundle or a
composite quality score.

Route discovery marks only the IRD route as `default: true`. Filtering with
`sure-eval metric routes full_duplex --metric fsed --json` returns the FSED
route with `default: false` and `execution_metrics: ["fsed"]`.
The response still records the IRD task default in `default_pipeline_id` and
`default_pipeline_ids`, even though that route is absent from the filtered list.

The report language is `n/a`, and pipeline identity uses `any`: these numerical
timing rules do not depend on the spoken language. IRD and FSED are ordinarily
interpreted as delays, but must be considered alongside event counts and
nonresponse. ERT and EIT have no universal optimization direction. A smaller
number alone does not establish better dialogue behavior.

## Input contract

Both existing CLI input roles are required:

| Role | Contents |
|:--|:--|
| `reference_jsonl` | one complete conversation per line, with user intervals and observation duration |
| `sample_output` | one corresponding conversation per line, with system intervals |

Reference:

```json
{"schema_version":1,"key":"case-A","time_unit":"seconds","time_origin":"conversation_start","duration":10.0,"user_segments":[{"id":"u1","start":1.0,"end":3.0},{"id":"u2","start":6.0,"end":8.0}]}
```

Prediction:

```json
{"schema_version":1,"key":"case-A","time_unit":"seconds","time_origin":"conversation_start","system_segments":[{"id":"a1","start":2.75,"end":6.25},{"id":"a2","start":8.4,"end":9.0}]}
```

Strict rules:

- `schema_version` must be integer `1`; time unit and origin must be exactly
  `seconds` and `conversation_start` on both sides. The two channels must refer
  to the same clock and observation window.
- Keys must be unique nonempty strings and align exactly. Row order may differ;
  missing predictions are errors, not silence.
- `duration` is a positive finite number on the reference side. Each segment
  has exactly `id`, `start`, and `end`, with `0 <= start < end <= duration`.
  Segment IDs are unique nonempty strings within each channel.
- Segments must already be in increasing start order and must not overlap
  within one channel. Adjacent intervals are allowed. Overlap between the user
  and system channels is essential signal and is preserved.
- User intervals must be nonempty. An explicit `system_segments: []` is valid
  silence. The parser does not sort, merge, clip, or silently drop segments.
- Numeric strings, booleans, null, NaN, and infinity are rejected. Decimal
  literals are retained without first converting them to binary floats.
- Unknown fields are rejected, except an optional row-level `metadata` object
  for provenance. Optional `metadata.measurement_protocol` has the validated
  structure described below; other metadata stays extensible. Metadata does
  not change scoring.
- This compatibility scorer requires every speech endpoint to be an exact
  multiple of `1/16000` second. Integer milliseconds satisfy this requirement.
  Off-grid endpoints fail explicitly; no hidden rounding is applied.

`duration` validates the observation window and supports diagnostics. It does
not add a timeout penalty or change the published implementation's final-turn
logic. An interval ending exactly at the recording boundary is not proof that
a model successfully stopped speaking.

## Optional measurement provenance

Timestamp scores depend on how speech boundaries and playback were measured.
Either input row may optionally include `metadata.measurement_protocol`:

```json
{
  "metadata": {
    "measurement_protocol": {
      "vad": {"name":"example-detector","version":"1","threshold":0.5,"min_silence_ms":1500,"padding_ms":30},
      "playback": {"queue_policy":"retain_all_received_audio","clock":"shared_monotonic","speaker_output_observed":false},
      "reference": {"source":"upstream_machine_schedule"},
      "preamble": {"removed":false,"reason":"no preamble exclusion"},
      "capture": {"model_id":"example-model","endpoint":"https://example.invalid/realtime","sample_rates":{"user":16000,"system":24000}},
      "notes":"Illustrative configuration, not a model benchmark."
    }
  }
}
```

The object and all its sections are optional. Supplied fields are checked,
including strict boolean/numeric types, finite numeric values, a VAD threshold
in `[0,1]`, nonnegative silence/padding, and positive integer sample rates.
VAD numeric parameters must fit the finite binary64 export range, and nonzero
values must not underflow to zero. Integers in that range remain exact integers
in the report; this restriction does not change exact Decimal timestamps.
Unknown fields inside this object or its named sections are rejected to catch
misspelled measurement settings. Other metadata keys are not restricted.
This is descriptive provenance: it never runs VAD, changes playback, deletes
preambles, or supplies scoring parameters.

`playback.queue_policy` accepts `retain_all_received_audio`,
`flush_on_server_speech_started`, or `other`. `reference.source` accepts
`human_annotation`, `upstream_machine_schedule`, or `other`. Known text fields
must be nonempty strings; `notes` is a string. Reference and prediction objects
are retained separately, without silently merging conflicting settings.

The report's `details.measurement_protocol.status` is:

- `recorded`: every conversation has at least one object containing all five
  VAD fields, all three playback fields, `reference.source`, and both preamble
  fields shown above. Capture and notes are optional.
- `missing`: no conversation supplies a measurement object.
- `partial`: incomplete objects or a mixture of covered and uncovered
  conversations. Missing fields remain allowed in v1.

Missing/partial provenance produces a CLI warning, not an input rejection.
`recorded` means settings were supplied, not independently verified; compare
the actual values and reference origins before comparing scores. Scores with
unrecorded measurement protocols must not be compared directly. In particular,
server cancellation, queued playback, and observed speaker silence are distinct.

## Fixed scoring protocol

The protocol is `fdbench_8a4b7df_compat`, anchored to FD-Bench commit
`8a4b7df1b4dcb0fc50a7a5660247a2ffe8d394eb`. The operational reference is
[`analyze_VAD_interruption_new2`, followed by `analyze_VAD_marks`](https://github.com/pengyizhou/FD-Bench/blob/8a4b7df1b4dcb0fc50a7a5660247a2ffe8d394eb/benchmark/benchmarking.py).
The implementation is independently organized around validated timelines,
explicit per-user slots, and an auditable event ledger; no upstream model or
dataset code is required.

The paper gives the physical meanings: how long old system speech continues
after a user interruption, how long a new response waits after user speech,
and how far a system start leads the user's endpoint. Exact sample selection
also depends on the fixed source implementation. Important compatibility
behaviors include:

- Every user interval visits system intervals in order with per-user state.
  This is not a one-to-one nearest-neighbor turn matcher.
- The early-interruption branch uses a strict threshold of 8000 ticks
  (500 ms). A start exactly 500 ms early follows the
  `elif user.start < system.start < user.end` / ERT branch.
  Starts exactly at the user start or end are excluded by strict comparisons.
- Some IRD branches measure the duration of an early system interval, or the
  new system endpoint minus the user start. They need not equal the intuitive
  old-system-stop minus user-start formula.
- Ordinary early-start ERT candidates can overwrite earlier candidates within
  a user slot; ordinary late-start FSED candidates retain their first candidate.
  The final-user branch differs from nonfinal-user handling.
- No implicit model-brand preamble removal is performed. Any deliberate
  preamble exclusion belongs in an explicit input export step with provenance.

After selecting the final slots, IRD uses interruption-delay slots, FSED pools
ordinary and post-interruption response delays, ERT pools the corresponding
lead-time slots, and EIT uses early-interruption lead times. For each metric:

1. Flatten selected slots over all conversations and retain **positive raw
   tick values**.
2. Convert each retained sample separately to integer milliseconds by
   truncating `ticks / 16`.
3. Take the median of the pooled millisecond samples. An even sample count
   averages its two middle values; do not average conversation medians.

An empty pool reports `score: 0.0`, `sample_count: 0`, `status: "no_events"`
for compatibility. This zero is not an observed instant response. A positive
one-tick delay instead becomes 0 ms with count 1 and status `ok`. Never rank
silence as best latency merely because its compatibility score is zero.

Score-affecting changes to the protocol require a new node/route version and
updated expectations. A `pipeline_id` does not automatically hash the protocol
string or its configuration; changing a literal under an existing identity
does not create a new computation version. The current input and score
semantics remain v1.

## CLI usage

The committed synthetic fixtures exercise threshold boundaries, branch
fallbacks, overwritten candidates, silence, and submillisecond truncation:

```bash
sure-eval metric describe full_duplex \
  --pipeline-id full_duplex.any.ird.duplex_timeline_v1.fdbench_timing_v1 \
  --output .sure-eval-demo/full-duplex.json

sure-eval env setup --pipeline .sure-eval-demo/full-duplex.json --dry-run
sure-eval env check --pipeline .sure-eval-demo/full-duplex.json

sure-eval metric run --pipeline .sure-eval-demo/full-duplex.json \
  --reference-jsonl tests/fixtures/full_duplex/reference.jsonl \
  --sample-output tests/fixtures/full_duplex/sample_output.jsonl \
  --output-dir .sure-eval-demo/full-duplex --validate-env
```

No optional model environment or API key is needed. To change the primary
metric, describe its exact route into a new pipeline JSON and run that file.
All four result details are present even when the primary metric is IRD.
The full-duplex run summary also includes `sample_count`, `status`, and
`score_is_measured`. Empty pools use `status: "no_events"`,
`score_is_measured: false`, and an explicit warning; a positive one-tick event
has `status: "ok"` and `score_is_measured: true` even though its score is 0 ms.
`measurement_protocol_status` and provenance warnings describe input coverage.

## Python API

```python
from sure_eval.evaluation.scripts import run_task

report = run_task(
    "full_duplex",
    reference_jsonl="tests/fixtures/full_duplex/reference.jsonl",
    sample_output="tests/fixtures/full_duplex/sample_output.jsonl",
    metric="ird",
    output_dir=".sure-eval-demo/full-duplex-python",
)
print(report.metric, report.score)
print(report.details["results"])
```

For this fixture corpus, IRD is 1700 ms (4 samples), FSED 1200 ms (4), ERT
250 ms (3), and EIT 750 ms (2). These hand-calculated values are synthetic
regression expectations, not model benchmark results. See the
[fixture rationale](../../tests/fixtures/full_duplex/README.md).

## Reports and verification

- `report.json` records the selected primary metric, all four result objects,
  input hashes, fixed configuration, per-conversation slots, and candidate
  events including final selection and overwrites.
- Full arrays and the ledger live at `details.per_conversation` and
  `details.events`. `details.results` holds the four pooled results;
  `details.evaluation_config` and `details.input_sha256` identify computation
  and input bytes. The trace contains compact node summaries, not another
  copy of the per-conversation arrays and ledger. There is no `details.scoring`
  duplicate or `details.normalization` input copy; the normalization count/clock
  summary is in `pipeline_trace[0].details.input_summary`.
- `details.measurement_protocol` records optional per-conversation, per-role
  measurement objects and coverage status. Other input metadata is retained
  once in `details.per_conversation[].metadata`; provenance has no score effect.
- `pipeline_description.json` records the exact route, two computation nodes,
  versions, singleton execution selector, input contract, and implementation
  entrypoints.

Run the focused regression suite from the repository root:

```bash
python -m pytest tests/evaluation/nodes/test_duplex_timeline.py \
  tests/evaluation/nodes/test_fdbench_timing.py \
  tests/evaluation/tasks/test_full_duplex_task.py tests/test_full_duplex_cli.py
```

Scoring the same saved timeline is deterministic. Repeating an external
realtime model session introduces model, network, scheduling, playback, and
VAD variation; it is not a test of deterministic scoring. When preparing
real recordings, retain a shared monotonic clock, actual user/system audio
and silence gaps, VAD/export settings, and any client cancellation policy.
Packet arrival time and `response.done` are not speech-boundary ground truth.
Do not concatenate received chunks without recording their actual presentation
positions. Offline VAD timestamps must be mapped back onto the shared timeline.
