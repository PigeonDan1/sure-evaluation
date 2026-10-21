# KWS — Keyword Spotting

KWS v2 reports per-keyword recall and macro recall at a fixed system-wide
false-alarm-per-hour budget, plus trigger latency at the same operating point.
The default budget is **0.1 FA/hour**. ROC curves and AUC are optional.

## Metric definitions

- `FA/hour = false trigger events on negative recordings / negative recording hours`.
  A negative recording contains no annotated target keyword. Each recording's
  duration is counted once, shared across all keywords. Multiple false triggers
  in one negative recording count separately. Unmatched triggers on positive
  recordings are reported as diagnostics and do not enter this background-only
  FA/hour definition.
- `recall_k = correctly matched events for keyword k / annotated events for k`.
  A wrong keyword is a miss. Keywords are normalized by removing whitespace and
  uppercasing. Only keywords represented by positive references enter the average.
- `macro_recall = mean(recall_k)`, with equal weight for each keyword.
- All keywords share one threshold. The evaluator sweeps distinct score boundaries
  and selects maximum macro recall subject to `achieved FA/hour <= target FA/hour`.
  Ties prefer fewer false alarms, then a higher threshold. Tied scores are accepted
  together using `score >= threshold`. An all-rejected point is always included.
  The report records the threshold, actual FA/hour, false-alarm count, and exposure.
  This is a curve operating point selected on the supplied evaluation data;
  deployment thresholds should be calibrated on separate data.
- `latency_ms = (trigger_time_sec - keyword_end_sec) * 1000` for correctly matched
  events at the selected threshold. Negative latency is preserved. The report
  includes overall and per-keyword mean, P50, P95, valid count, matched count,
  and timestamp coverage. Missing timestamps produce `null`, not zero latency.

Accuracy, precision, standalone pooled recall, F1, FRR, FAR, and DET are no longer
KWS metrics. FA/hour and event counts remain operating-point details.

## Exact pipeline workflow

```bash
sure-eval metric routes kws --metric macro_recall --json
sure-eval metric describe kws \
  --pipeline-id kws.any.macro_recall.conversion_kws_sure_json_to_samples_v2.wekws_det_v2 \
  --output /tmp/kws.json
sure-eval env setup --pipeline /tmp/kws.json --dry-run
sure-eval env setup --pipeline /tmp/kws.json
sure-eval env check --pipeline /tmp/kws.json
sure-eval metric run --pipeline /tmp/kws.json \
  --reference-jsonl ref.jsonl --sample-output pred.jsonl \
  --output-dir /tmp/kws_eval
```

`macro_recall` is the default primary metric; `macro-recall` is an accepted alias.
The described pipeline's `run_args` records `target_fa_per_hour: 0.1`,
`include_roc: false`, and `match_tolerance_sec: 1.0`. Edit these values or override
with `--target-fa-per-hour`, `--include-roc` / `--no-include-roc`, and
`--match-tolerance-sec` when running. Omitting an override preserves JSON settings.
The historical `--macro-recall-false-alarms` count option has been removed.

Three input routes are available, all with conversion v2 and scoring v2:

| Input mode | Required roles | Pipeline suffix after `kws.any.macro_recall.` |
|:--|:--|:--|
| `sure_json` | `reference_jsonl`, `sample_output` | `conversion_kws_sure_json_to_samples_v2.wekws_det_v2` |
| `wekws_score_ctc` | `wekws_label_file`, `wekws_score_file`, `keyword` | `conversion_kws_wekws_score_ctc_to_samples_v2.wekws_det_v2` |
| `wekws_frame_score` | `wekws_label_file`, `wekws_frame_score_file`, `keyword` | `conversion_kws_wekws_frame_score_to_samples_v2.wekws_det_v2` |

Old v1 pipeline JSON files must be regenerated. Their scoring semantics differ.

## Scored clips

Reference JSONL, with times and duration in seconds:

```jsonl
{"key":"pos","expected":true,"expected_keyword":"hello","keyword_start_sec":0.3,"keyword_end_sec":1.0,"duration":2.0}
{"key":"neg","expected":false,"duration":36000.0}
```

Predictions accept JSONL, a JSON array, or an object containing `rows`. Each row
may contain a `result` object or put its fields directly alongside `key`:

```jsonl
{"key":"pos","result":{"keyword":"hello","score":0.9,"trigger_time_sec":1.15}}
{"key":"neg","result":{"keyword":"hello","score":0.2,"trigger_time_sec":12.0}}
```

Provide raw candidate scores on every clip, including clips below the model's
original threshold. A `detected` flag is accepted as historical metadata but does
not gate the evaluator's threshold sweep. Do not use threshold-censored outputs.
A scalar-score clip represents at most one trigger. Missing, extra, or duplicate
keys are errors. Negative durations must be finite and positive; at least one
negative recording and one positive keyword are required.

## Multiple events in recordings

Use explicit lists when recordings can contain multiple wake events:

```jsonl
{"key":"speech","duration":8.0,"events":[{"keyword":"hello","start_sec":0.5,"end_sec":1.0},{"keyword":"hello","start_sec":4.0,"end_sec":4.7}]}
{"key":"background","duration":36000.0,"events":[]}
```

```jsonl
{"key":"speech","detections":[{"keyword":"hello","score":0.9,"trigger_time_sec":1.1},{"keyword":"hello","score":0.8,"trigger_time_sec":4.8}]}
{"key":"background","detections":[{"keyword":"hello","score":0.6,"trigger_time_sec":10.0},{"keyword":"hello","score":0.5,"trigger_time_sec":20.0}]}
```

`detections` must contain distinct candidate trigger events before score filtering,
with the detector's event grouping/debouncing applied. The evaluator does not
invent a cooldown or infer distinct events from arbitrary frame peaks. An empty
list explicitly means no candidates. Do not combine a detection list with scalar
or frame scores.

Matching is one-to-one and keyword-specific. Candidates are processed in time
order and matched to the eligible reference with the earliest end. The matching
window is `[start_sec, end_sec + match_tolerance_sec]`; without a reference start,
it is `[max(0, end_sec - match_tolerance_sec), end_sec + match_tolerance_sec]`.
Multiple reference events require reference end times and candidate trigger times.
For a single clip with unavailable timestamps, matching uses keyword identity and
latency is unavailable. Extra triggers cannot increase recall.

## WeKWS input adapters

Labels are JSONL with `key`, `txt` (or `text`), and negative `duration` in seconds.
Optional timing fields use the same flat names as scored clips.

The CTC adapter requires an uncensored score on **every** row:

```text
pos detected hello 0.9 1.15
neg rejected hello 0.2
```

The final trigger time is optional. Original `key rejected` rows contain no raw
score and are rejected with an explanation; they cannot recover the requested
threshold sweep.

Frame-score rows retain the format `key keyword score score ...`. Each clip
represents at most one detection. For latency, provide `frame_shift_sec` and
optional `frame_offset_sec` (default zero) in the labels. Frame index `i` has
availability time `frame_offset_sec + (i + 1) * frame_shift_sec`. The first frame
meeting the selected threshold determines the trigger time, not the peak frame.
No frame interval is guessed when metadata is missing.

## Optional ROC and AUC

Use `--include-roc` for standard one-vs-rest clip ROC, with false positive rate
on the x-axis and true positive rate on the y-axis. AUC uses trapezoidal integration;
the aggregate is the unweighted mean of keyword AUCs. For multiple keywords,
provide `keyword_scores` for every evaluated keyword on every clip, including
other-keyword clips and negatives:

```json
{"key":"pos","keyword":"hello","score":0.9,"keyword_scores":{"hello":0.9,"computer":0.1}}
```

Single-keyword clips can use their scalar or maximum frame score. Event-list
inputs and incomplete multi-keyword score tables report ROC/AUC as unavailable,
with a reason and `null` score. FA/hour-versus-recall points are always available
as `operating_curve`; their area is not reported as standard ROC AUC.

## Python API and reports

```python
from sure_eval.evaluation.tasks.kws import KWSMetricPipeline, KWSSample

report = KWSMetricPipeline(target_fa_per_hour=0.1).evaluate([
    KWSSample(key="p", expected_detected=True, expected_keyword="hello",
              predicted_keyword="hello", score=0.9,
              keyword_end_sec=1.0, trigger_time_sec=1.15),
    KWSSample(key="n", expected_detected=False, duration=36000.0, score=0.2),
])
print(report.results["macro_recall"].score)
```

`report.json` contains primary `score`, `results.macro_recall`,
`results.per_keyword_recall`, `results.trigger_latency`, the operating-point
summary, reference-event rows, and `operating_curve` under `details`.
`results.roc_curve` and `results.auc` appear only when requested.
`pipeline_description.json` records the exact route and node versions.
The convenience script `scripts/run_kws_metric_pipeline.py` also requires
`--pipeline` and `--output-dir` and executes the same pipeline contract.
