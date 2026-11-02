# ETC canonicalization

This optional node adapts the model canonicalizer interface released with
[ETC-Bench](https://github.com/zh-qian2004/etc-bench). It independently maps
reference and hypothesis transcripts to a deterministic shared written form.

Prepare its node-local environment with:

```bash
sure-eval env setup --node normalization/etc_canonicalization
```

Download one of the published checkpoints separately, then set
`SURE_ETC_CANONICALIZATION_MODEL` to its local directory. The default model ID
is `zhiheng-qian/etc-qwen3.5-0.8b`; it works only when that checkpoint is
already present in the local Hugging Face cache. The runner uses
`local_files_only=True`, so evaluation never downloads weights implicitly.

The node environment and model checkpoint are intentionally not included in
SURE-Evaluation or this change. After cloning, install the optional node
environment and configure the checkpoint in your own runtime.
