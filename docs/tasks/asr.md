# ASR — Automatic Speech Recognition

Evaluate text transcripts against references.

## Metric Families And Routes

The reported ASR metric families are still CER, WER, and MER. Selectors such
as `cer_canonical`, `wer_canonical`, and `mer_canonical` are opt-in route
selectors for canonical normalization variants; they are not new evaluation
metric families.

| Reported metric | Route selector | Language | Pipeline ID | Nodes |
|:----------------|:---------------|:---------|:------------|:------|
| CER | `cer` | `zh` | `asr.zh.cer.aispeech_norm.wenet_cer` | `normalization/aispeech_norm` → `scoring/wenet_cer` |
| CER | `cer_canonical` (opt-in, needs `[canonical]` extra) | `zh` | `asr.zh.cer_canonical.canonical_itn.token_cer` | `normalization/canonical_itn` → `scoring/token_cer` |
| WER | `wer` | `en` (Whisper norm) | `asr.en.wer.whisper_norm.wenet_wer` | `normalization/whisper_norm` → `scoring/wenet_wer` |
| WER | `wer` + `normalizer="aispeech"` | `en` (legacy AISpeech norm) | `asr.en.wer.aispeech_norm.wenet_wer` | `normalization/aispeech_norm` → `scoring/wenet_wer` |
| WER | `wer_canonical` (opt-in, needs `[canonical]` extra) | `en` | `asr.en.wer_canonical.canonical_itn.token_mer` | `normalization/canonical_itn` → `scoring/token_mer` |
| MER | `mer` | `cs` (code-switching) | `asr.cs.mer.aispeech_norm.wenet_mer` | `normalization/aispeech_norm` → `scoring/wenet_mer` |
| MER | `mer_canonical` (opt-in, needs `[canonical]` extra) | `cs` | `asr.cs.mer_canonical.canonical_itn.token_mer` | `normalization/canonical_itn` → `scoring/token_mer` |

## Input Format

Tab-separated key-text files:

```text
utt_001<TAB>你好世界
utt_002<TAB>今天天气不错
```

Both `--ref-file` and `--hyp-file` use the same format, aligned by key.

## CLI Usage

```bash
# Describe the route
sure-eval metric describe asr --language zh --metric cer --output /tmp/asr.json

# Run
sure-eval metric run --pipeline /tmp/asr.json \
  --ref-file ref.txt --hyp-file hyp.txt --output-dir /tmp/asr_eval
```

## Python API

```python
from sure_eval.evaluation.scripts import run_task

report = run_task(
    "asr",
    ref_file="ref.txt",
    hyp_file="hyp.txt",
    language="zh",
    metric="cer",
    output_dir="/tmp/asr_eval",
)
print(report.score)  # CER
```

## Output

- `report.json` — `score`, `cer` or `wer`, edit counts (`all`, `cor`, `sub`, `ins`, `del`).
- `pipeline_description.json` — selected route and node versions.

## Optional Tools

- `normalization/wetext_norm` — select with `normalizer="wetext:zh_tn"` or `normalizer="wetext:en_itn"`.
- `scoring/sctk_sclite` — select with `scorer="sctk_sclite"`.

## Canonical Normalization Routes

Canonical routes keep the same metric families (CER for zh, WER for en, MER
for code-switch ASR) but change the normalization/scoring route. Text is
canonicalized via ITN (spoken → written, many-to-one — 2024 ≡ 二零二四 ≡
两千零二十四, 百分之五十 ≡ 50%), punctuation becomes spaces, and scoring is
token-level (CJK per char, latin words, digits per char) with S/D/I from
minimal edit operations. These routes coexist with the default WeNet-compatible
routes and never replace them.

```bash
pip install "sure-evaluation[canonical]"
sure-eval metric describe asr --language zh --metric cer_canonical --output /tmp/asr_canon.json
sure-eval metric run --pipeline /tmp/asr_canon.json \
  --ref-file ref.txt --hyp-file hyp.txt --output-dir /tmp/asr_canon_eval
```

Determinism: identical scores require an identical `cn2an` version; the
engine version is recorded in each run's node trace. See
`src/sure_eval/evaluation/nodes/normalization/canonical_itn/README.md` for
the full chain and known limitations.

The canonical route selectors share one normalization chain, one tokenizer,
and one scorer:

- `cer_canonical` selects canonical-normalized CER for zh — CJK per char,
  digits per char;
- `wer_canonical` selects canonical-normalized WER for en — latin words; the
  normalization stage additionally
  whisper-normalizes latin spans (contraction expansion `don't` →
  `do not`, spoken numbers `fifty percent` → `50%`, spoken-filler removal,
  British→American spelling fold) using the vendored Whisper English
  normalizer, with two robustness passes: unambiguous bare contractions are
  restored first (`dont` → `don't`, so apostrophe-dropping outputs expand
  identically), and `'s` is collapsed instead of expanded (`it's` ≡ `its`,
  `john's` ≡ `johns`; `'s` is possessive/is/has-ambiguous);
- `mer_canonical` selects canonical-normalized MER for code-switch ASR — both
  of the above in one token stream.

Scoring includes a deterministic word-spacing repair: a latin word equal to
the concatenation of 2–4 consecutive words on the other side is split
(`tenthe` ≡ `ten the`), so pure spacing artifacts never score as errors
while any letter difference stays fully scored.

Degeneration holds by construction and is locked by tests: text without
latin letters scores identically under `mer_canonical` and `cer_canonical`
(the whisper stage only touches spans containing latin letters); text
without CJK scores identically under `mer_canonical` and `wer_canonical`.

These are not part of the default routes and must be requested explicitly.
