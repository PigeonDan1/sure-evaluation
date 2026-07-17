# WeText Normalization

`normalization/wetext_norm` is an optional wrapper around
`WeTextProcessing==1.2.0`.

It is a node-local `uv` project because WeTextProcessing depends on Pynini and
must remain version-managed by the node rather than by the root SURE-EVAL
environment. Mandarin ASR CER selects this node by default, while TTS/VC
Mandarin semantic routes default to `punctuation_strip_norm`.

Prepare it only when a selected route needs it:

```bash
sure-eval env setup --node normalization/wetext_norm
```

Supported profiles:

- `zh_tn`: Chinese text normalization.
- `zh_itn`: Chinese inverse text normalization.
- `en_tn`: English text normalization.
- `en_itn`: English inverse text normalization.
- `ja_tn`: Japanese text normalization.
- `ja_itn`: Japanese inverse text normalization.

Use temporary cache directories for demos and tests so generated FST artifacts do
not land in this source tree.
