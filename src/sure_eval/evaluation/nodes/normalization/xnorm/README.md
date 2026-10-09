# XNorm normalization node

This node adapts the bundled XNorm 1.5.0 multilingual normalizer to SURE
key-text files. The XNorm source is kept unchanged under `xnorm_vendor/`.

Supported profiles are `ar`, `de`, `en`, `es`, `fr`, `he`, `hi`, `hu`, `hy`,
`id`, `it`, `ja`, `ko`, `mr`, `pt`, `ru`, `sv`, `th`, `vi`, and `zh`.

Use it explicitly for ASR with `normalizer="xnorm:<language>"`, for example:

```python
evaluate_asr_files(ref, hyp, language="id", metric="wer", normalizer="xnorm:id")
```

The node runs in its optional node-local environment. Run
`sure-eval env setup --node normalization/xnorm` before scoring. Existing ASR
default routes are unchanged; XNorm is selected only when requested.
