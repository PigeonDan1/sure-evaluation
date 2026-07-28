# FunASR Inverse Text Normalization

## Purpose

`normalization/funasr_itn` wraps the FunASR `fun_text_processing` package as a
versioned optional node. It provides Inverse Text Normalization (ITN) —
converting spoken-form text to written form — for 12 languages using pynini
WFST grammars. The `fun_text_processing` source is vendored under this node
directory, keeping it self-contained.

The node normalizes text only. It does not calculate WER, CER, or MER.

## Task Scenarios

This node is the **default normalizer** for 10 languages (ja, ko, es, fr, de, ru,
pt, vi, id, tl) — calling `evaluate_asr_files(language="ja", metric="cer")`
without a `normalizer` argument automatically selects `funasr:ja`:

```python
from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

# Default (funasr automatically selected for ja/ko/es/fr/de/ru/pt/vi/id/tl)
evaluate_asr_files(ref_file="ref.txt", hyp_file="hyp.txt", language="ja", metric="cer")

# Explicit override (e.g., for zh/en which default to wetext/whisper)
evaluate_asr_files(
    ref_file="ref.txt", hyp_file="hyp.txt",
    language="zh", metric="cer",
    normalizer="funasr:zh",
)
```

Supported languages: zh, en, ja, es, fr, de, ko, ru, pt, vi, id, tl.

## Input

- Schema: `key_text_files`.
- Each file is a `<key><TAB><text>` file with one row per utterance.
- Profiles:

| Profile | Language | Direction |
|---------|----------|-----------|
| zh      | Chinese  | ITN       |
| en      | English  | ITN       |
| ja      | Japanese | ITN       |
| es      | Spanish  | ITN       |
| fr      | French   | ITN       |
| de      | German   | ITN       |
| ko      | Korean   | ITN       |
| ru      | Russian  | ITN       |
| pt      | Portuguese | ITN     |
| vi      | Vietnamese | ITN     |
| id      | Indonesian | ITN     |
| tl      | Tagalog  | ITN       |

## Output

- Schema: `key_text_files`.
- Output preserves key alignment and row count from the input.
- Trace records: `profile`, `language`, `direction`, `normalizer_class`,
  `num_rows`.

## Versioned Computation

- Node id: `normalization/funasr_itn`.
- Version: `v1`.
- Package: `fun_text_processing` (vendored, no version pin).
- Important dependency: `pynini>=2.1.6`.
- Direction: ITN (spoken-form text to written form).
- Internal stages: tagging (WFST-based) → verbalization.

## Runtime and Assets

- Runtime: optional node-local `uv` project.
- Python: `3.11`.
- No model weights, checkpoints, or API keys.
- Setup:

```bash
sure-eval env setup --node normalization/funasr_itn
```

The `fun_text_processing` package is vendored under this node directory and
loaded via `PYTHONPATH`. No external path configuration is required.

## Source and References

- FunASR repository: https://github.com/modelscope/FunASR
  (Apache 2.0 license)
- Pynini: https://www.opengrm.org/twiki/bin/view/GRM/Pynini

## Limitations

- The node requires its node-local environment; base install alone is not
  enough to run normalization.
- FST cache artifacts are runtime state and must not be committed.
- Only ITN direction is supported; TN (written-to-spoken) is not exposed
  through this node.
- Some languages (ko, ru, pt, vi, id, tl) have limited upstream coverage
  compared to zh/en.
