# Speaker Verification

The `SV` task evaluates target and nontarget trials from model embeddings at the
testset level. An OpenBench dataset may expose multiple testset protocols. Each
evaluation run consumes one testset trial manifest and never concatenates trials
from different testsets. Models receive audio samples and return one embedding
per unique sample; labels and pair definitions remain evaluator-side.

## Inputs

- `sample_output`: JSONL model output with `key` or `sample_id` and
  `result.embedding`.
- `trial_manifest`: JSON manifest using schema
  `sure.sv.trial_manifest.v1`, pointing to a TSV trial file.

The canonical trial TSV columns are:

```text
enroll_key\ttest_key\tlabel\tcondition
```

`condition` is metadata-only. The generic task computes one score for the full
testset protocol and does not implement benchmark-specific diagnostics.

## Metrics

- `eer`: percentage, using linear interpolation at the first DET crossing.
- `min_dcf`: normalized minimum DCF with `P_target=0.05`, `C_miss=1`,
  `C_fa=1`.
