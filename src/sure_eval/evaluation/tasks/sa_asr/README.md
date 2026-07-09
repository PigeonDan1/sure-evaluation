# SA-ASR Evaluation

Speaker-attributed ASR uses a conversion bridge around
`normalization/gstar_norm`, followed by the generic `scoring/meeteval` node.
The default route reports cpWER as the main score and DER as a companion
metric.

The default route is `sa_asr.cpwer.gstar_norm.meeteval` with `collar=0.5`
for the DER companion metric. It expects STM six-field rows so the normalization
node can match the G-STAR-compatible SA-ASR text behavior used by
`SUREEvaluator._eval_sa_asr`.

The task layer uses:

```text
src/sure_eval/evaluation/conversion/sa_asr__cpwer/
```

to convert STM to key-text before normalization and normalized key-text back to
STM before `meeteval.io.load`. This conversion is not a metric node, but it is
recorded in `conversion_trace` because it can affect the final score.
