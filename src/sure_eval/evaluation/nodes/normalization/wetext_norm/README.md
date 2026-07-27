# WeText Normalization

## Purpose

`normalization/wetext_norm` wraps `WeTextProcessing==1.2.0` as a versioned
optional node. It provides TN and ITN profiles for Chinese, English, and
Japanese while keeping the Pynini dependency isolated from the root
SURE-EVAL environment.

The node normalizes text only. It does not calculate WER, CER, or MER.

## Task Scenarios

- Default Mandarin ASR CER route:
  `asr.zh.cer.wetext_norm_zh_itn_v1.wenet_cer_v1`.
- Other routes may select this node explicitly through exact pipeline ids or
  future normalizer arguments.

Mandarin TTS/VC semantic CER defaults to `punctuation_strip_norm`, not WeText.

## Input

- Schema: `text_or_key_text_files`.
- Supports single text strings and aligned `<key><TAB><text>` files.
- Profiles:
  - `zh_tn`, `zh_itn`
  - `en_tn`, `en_itn`
  - `ja_tn`, `ja_itn`

## Output

- Schema: `text_or_key_text_files`.
- Output preserves the input shape: string input returns text; key-text input
  returns key-text files.
- Trace records profile, package version, cache behavior, and generated output
  files.

## Versioned Computation

- Node id: `normalization/wetext_norm`.
- Version: `v1`.
- Package: `WeTextProcessing==1.2.0`.
- Important dependency: `pynini>=2.1.6,<2.2`.
- Direction:
  - TN: written text to spoken-style normalized text.
  - ITN: spoken-style text to written form.

## Runtime and Assets

- Runtime: optional node-local `uv` project.
- Python: `3.11`.
- Setup:

```bash
sure-eval env setup --node normalization/wetext_norm
```

Use temporary cache directories for demos and tests so generated FST artifacts
do not land in the source tree.

## Source and References

- WeTextProcessing: https://github.com/wenet-e2e/WeTextProcessing
- Pynini: https://www.opengrm.org/twiki/bin/view/GRM/Pynini

## Limitations

- The node requires its node-local environment; base install alone is not
  enough to run selected WeText routes.
- FST cache artifacts are runtime state and must not be committed.
