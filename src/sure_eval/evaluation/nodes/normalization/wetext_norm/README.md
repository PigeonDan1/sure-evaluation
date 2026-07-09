# WeText Normalization

`normalization/wetext_norm` is an optional wrapper around
`WeTextProcessing==1.2.0`.

It is intentionally isolated as a node-local project because WeTextProcessing
depends on Pynini. The SURE default ASR/TTS/VC routes do not use this node unless
a future task configuration explicitly selects it.

Supported profiles:

- `zh_tn`: Chinese text normalization.
- `zh_itn`: Chinese inverse text normalization.
- `en_tn`: English text normalization.
- `en_itn`: English inverse text normalization.
- `ja_tn`: Japanese text normalization.
- `ja_itn`: Japanese inverse text normalization.

Use temporary cache directories for demos and tests so generated FST artifacts do
not land in this source tree.
