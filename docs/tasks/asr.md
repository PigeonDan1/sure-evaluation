# ASR - Automatic Speech Recognition

Evaluate text transcripts against references.

## Metrics And Pipelines

ASR reports only canonical metrics: `cer`, `wer`, and `mer`. Each metric can
have multiple concrete pipeline IDs when the normalization or scoring nodes
differ. Use `--metric` for the default pipeline and `--pipeline-id` for a
specific non-default pipeline.

### `cer`

| Pipeline ID | Language | Nodes | Notes |
|:------------|:---------|:------|:------|
| `asr.zh.cer.wetext_norm_zh_itn_v1.wenet_cer_v1` | `zh` | `normalization/wetext_norm` (`zh_itn`) -> `scoring/wenet_cer` | Default Mandarin CER |
| `asr.zh.cer.aispeech_norm_zh_v1.wenet_cer_v1` | `zh` | `normalization/aispeech_norm` -> `scoring/wenet_cer` | Legacy AISpeech-normalized CER |
| `asr.zh.cer.canonical_itn_zh_v1.token_cer_v1` | `zh` | `normalization/canonical_itn` -> `scoring/token_cer` | Canonical ITN CER; requires `[canonical]` |

### `wer`

| Pipeline ID | Language | Nodes | Notes |
|:------------|:---------|:------|:------|
| `asr.en.wer.whisper_norm_english_v1.wenet_wer_v1` | `en` | `normalization/whisper_norm` -> `scoring/wenet_wer` | Default English WER |
| `asr.en.wer.aispeech_norm_en_v1.wenet_wer_v1` | `en` | `normalization/aispeech_norm` -> `scoring/wenet_wer` | Legacy AISpeech-normalized WER |
| `asr.en.wer.canonical_itn_en_v1.token_mer_v1` | `en` | `normalization/canonical_itn` -> `scoring/token_mer` | Canonical ITN WER; requires `[canonical]` |

### `mer`

| Pipeline ID | Language | Nodes | Notes |
|:------------|:---------|:------|:------|
| `asr.cs.mer.aispeech_norm_cs_v1.wenet_mer_v1` | `cs` | `normalization/aispeech_norm` -> `scoring/wenet_mer` | Default code-switch MER |
| `asr.cs.mer.canonical_itn_cs_v1.token_mer_v1` | `cs` | `normalization/canonical_itn` -> `scoring/token_mer` | Canonical ITN MER; requires `[canonical]` |

`pipeline_id` uses the canonical metric plus versioned computation nodes.
Canonical ITN pipelines are selected by exact `pipeline_id`, not by a separate
metric name.

## Input Format

Tab-separated key-text files:

```text
utt_001<TAB>你好世界
utt_002<TAB>今天天气不错
```

Both `--ref-file` and `--hyp-file` use the same format, aligned by key.

## CLI Usage

```bash
# Default Mandarin CER pipeline
sure-eval metric describe asr --language zh --metric cer --output /tmp/asr.json

# Specific canonical Mandarin CER pipeline
sure-eval metric describe asr \
  --pipeline-id asr.zh.cer.canonical_itn_zh_v1.token_cer_v1 \
  --output /tmp/asr_canonical.json

sure-eval metric run --pipeline /tmp/asr_canonical.json \
  --ref-file ref.txt --hyp-file hyp.txt --output-dir /tmp/asr_eval
```

## Python API

```python
from sure_eval.evaluation.scripts import run_task

report = run_task(
    "asr",
    ref_file="ref.txt",
    hyp_file="hyp.txt",
    pipeline_id="asr.zh.cer.canonical_itn_zh_v1.token_cer_v1",
    output_dir="/tmp/asr_eval",
)
print(report.metric, report.score)  # cer, score
```

## Output And Traceability

- `report.json` contains canonical `metric`, `score`, edit counts, and the
  actual `pipeline_id`.
- `pipeline_description.json` contains canonical `metric`, `pipeline_id`,
  `computation_node_ids`, `nodes`, relative `task_config_path`,
  `route_config_path`, `describe_entrypoint`, `script_entrypoint`, and
  `executor`.

The route config is
`src/sure_eval/evaluation/tasks/asr/routes.yaml`. The script entrypoint is
`sure_eval.evaluation.scripts.asr.run`, and the task executor is
`sure_eval.evaluation.tasks.asr.pipeline.evaluate_asr_files`.

## Additional Tools

- `normalization/wetext_norm` - Mandarin CER defaults to `wetext:zh_itn`;
  other profiles can be selected with `normalizer="wetext:zh_tn"` or
  `normalizer="wetext:en_itn"` in the lower-level task API.
- `scoring/sctk_sclite` - optional binary-backed scorer wrapping NIST SCTK
  `sclite`; default ASR pipelines continue to use WeNet-compatible scorers.

## Canonical ITN Pipelines

Canonical ITN pipelines keep the same reported metrics (`cer`, `wer`, `mer`)
but change the normalization and scorer chain. Text is canonicalized via ITN
(spoken to written, many-to-one: `2024` equals `二零二四`; `50%` equals
`百分之五十`), punctuation becomes spaces, and scoring is token-level with
CJK characters, latin words, digits, and symbols as explicit tokens.

English and code-switch canonical pipelines additionally normalize latin spans
with the vendored Whisper English normalizer. The code-switch pipeline uses the
same token scorer as English WER while preserving CJK character tokens.

Scoring includes deterministic word-spacing repair: a latin word equal to the
concatenation of 2-4 consecutive words on the other side is split
(`tenthe` equals `ten the`), so pure spacing artifacts do not count as errors.

Determinism requires an identical `cn2an` version; the engine version is
recorded in the node trace.
