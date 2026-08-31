# SV — Speaker Verification

`SV` is a language-independent speaker-verification task evaluated at the
testset level. An OpenBench dataset may contain multiple testset protocols, and
each evaluation run consumes one testset manifest. A model emits one embedding
per unique audio sample, while the dataset provides an evaluator-only trial
manifest containing target and nontarget pairs.

Canonical pipelines:

- `sv.any.eer.cosine_trial_scores_v1.det_eer_v1`
- `sv.any.min_dcf.cosine_trial_scores_v1.min_dcf_p005_v1`

The default bundle evaluates both metrics. EER is reported in percent using
linear DET-crossing interpolation. minDCF is normalized with `P_target=0.05`,
`C_miss=1`, and `C_fa=1`.

```bash
sure-eval metric describe sv --metrics eer,min_dcf --output sv.json
sure-eval metric run \
  --pipeline sv.json \
  --sample-output embeddings.jsonl \
  --trial-manifest trial_manifest.json \
  --output-dir sv-report
```
