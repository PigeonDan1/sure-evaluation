# ASR - Automatic Speech Recognition

Evaluate text transcripts against references.

Discover exact routes before selecting one:

```bash
sure-eval metric routes asr --language zh --metric cer
```

## Metrics And Pipelines

ASR reports only the public metrics `cer`, `wer`, and `mer`. Each metric can
have multiple concrete pipeline IDs when the normalization or scoring nodes
differ. Use `--metric` for the default pipeline and `--pipeline-id` for a
specific non-default pipeline.

### `cer`

| Pipeline ID | Language | Nodes | Notes |
|:------------|:---------|:------|:------|
| `asr.zh.cer.wetext_norm_zh_itn_v1.wenet_cer_v1` | `zh` | `normalization/wetext_norm` (`zh_itn`) -> `scoring/wenet_cer` | Default Mandarin CER |
| `asr.zh.cer.aispeech_norm_zh_v1.wenet_cer_v1` | `zh` | `normalization/aispeech_norm` -> `scoring/wenet_cer` | Legacy AISpeech-normalized CER |
| `asr.zh.cer.canonical_itn_zh_v2.token_cer_v1` | `zh` | `normalization/canonical_itn` -> `scoring/token_cer` | Canonical ITN CER; requires `[canonical]` |
| `asr.ja.cer.funasr_itn_ja_v1.wenet_cer_v1` | `ja` | `normalization/funasr_itn` (`ja`) -> `scoring/wenet_cer` | Default Japanese CER; optional node setup required |
| `asr.ko.cer.funasr_itn_ko_v1.wenet_cer_v1` | `ko` | `normalization/funasr_itn` (`ko`) -> `scoring/wenet_cer` | Default Korean CER; optional node setup required |
| `asr.zh.cer.giga_norm_chn_v1.wenet_cer_v1` | `zh` | `normalization/giga_norm` (`CHN`) -> `scoring/wenet_cer` | GigaSpeechBench alternative; requires `[giga]` |
| `asr.ja.cer.giga_norm_jpn_v1.wenet_cer_v1` | `ja` | `normalization/giga_norm` (`JPN`) -> `scoring/wenet_cer` | GigaSpeechBench alternative; requires `[giga]` |
| `asr.ko.cer.giga_norm_kor_v1.wenet_cer_v1` | `ko` | `normalization/giga_norm` (`KOR`) -> `scoring/wenet_cer` | GigaSpeechBench alternative; requires `[giga]` |
| `asr.th.cer.giga_norm_tha_v1.wenet_cer_v1` | `th` | `normalization/giga_norm` (`THA`) -> `scoring/wenet_cer` | GigaSpeechBench route; requires `[giga]` |
| `asr.ar.cer.nemo_norm_ar_tn_v1.wenet_cer_v1` | `ar` | `normalization/nemo_norm` (`ar_tn`) -> `scoring/wenet_cer` | Default Arabic CER; optional node setup required |

### `wer`

| Pipeline ID | Language | Nodes | Notes |
|:------------|:---------|:------|:------|
| `asr.en.wer.whisper_norm_english_v1.wenet_wer_v1` | `en` | `normalization/whisper_norm` -> `scoring/wenet_wer` | Default English WER |
| `asr.en.wer.aispeech_norm_en_v1.wenet_wer_v1` | `en` | `normalization/aispeech_norm` -> `scoring/wenet_wer` | Legacy AISpeech-normalized WER |
| `asr.en.wer.canonical_itn_en_v2.token_mer_v1` | `en` | `normalization/canonical_itn` -> `scoring/token_mer` | Canonical ITN WER; requires `[canonical]` |
| `asr.es.wer.funasr_itn_es_v1.wenet_wer_v1` | `es` | `normalization/funasr_itn` (`es`) -> `scoring/wenet_wer` | Default Spanish WER; optional node setup required |
| `asr.fr.wer.funasr_itn_fr_v1.wenet_wer_v1` | `fr` | `normalization/funasr_itn` (`fr`) -> `scoring/wenet_wer` | Default French WER; optional node setup required |
| `asr.de.wer.funasr_itn_de_v1.wenet_wer_v1` | `de` | `normalization/funasr_itn` (`de`) -> `scoring/wenet_wer` | Default German WER; optional node setup required |
| `asr.ru.wer.funasr_itn_ru_v1.wenet_wer_v1` | `ru` | `normalization/funasr_itn` (`ru`) -> `scoring/wenet_wer` | Default Russian WER; optional node setup required |
| `asr.pt.wer.funasr_itn_pt_v1.wenet_wer_v1` | `pt` | `normalization/funasr_itn` (`pt`) -> `scoring/wenet_wer` | Default Portuguese WER; optional node setup required |
| `asr.vi.wer.funasr_itn_vi_v1.wenet_wer_v1` | `vi` | `normalization/funasr_itn` (`vi`) -> `scoring/wenet_wer` | Default Vietnamese WER; optional node setup required |
| `asr.id.wer.funasr_itn_id_v1.wenet_wer_v1` | `id` | `normalization/funasr_itn` (`id`) -> `scoring/wenet_wer` | Default Indonesian WER; optional node setup required |
| `asr.tl.wer.funasr_itn_tl_v1.wenet_wer_v1` | `tl` | `normalization/funasr_itn` (`tl`) -> `scoring/wenet_wer` | Default Tagalog WER; optional node setup required |
| `asr.en.wer.giga_norm_usa_v1.wenet_wer_v1` | `en` | `normalization/giga_norm` (`USA`) -> `scoring/wenet_wer` | GigaSpeechBench alternative; requires `[giga]` |
| `asr.id.wer.giga_norm_idn_v1.wenet_wer_v1` | `id` | `normalization/giga_norm` (`IDN`) -> `scoring/wenet_wer` | GigaSpeechBench alternative; requires `[giga]` |
| `asr.ms.wer.giga_norm_mys_v1.wenet_wer_v1` | `ms` | `normalization/giga_norm` (`MYS`) -> `scoring/wenet_wer` | GigaSpeechBench route; requires `[giga]` |
| `asr.fil.wer.giga_norm_phl_v1.wenet_wer_v1` | `fil` | `normalization/giga_norm` (`PHL`) -> `scoring/wenet_wer` | GigaSpeechBench route; requires `[giga]` |
| `asr.vi.wer.giga_norm_vnm_v1.wenet_wer_v1` | `vi` | `normalization/giga_norm` (`VNM`) -> `scoring/wenet_wer` | GigaSpeechBench alternative; requires `[giga]` |
| `asr.ar.wer.giga_norm_ar_v1.wenet_wer_v1` | `ar` | `normalization/giga_norm` (`AR`) -> `scoring/wenet_wer` | GigaSpeechBench route; requires `[giga]` |
| `asr.ar_ae.wer.giga_norm_are_v1.wenet_wer_v1` | `ar-ae` | `normalization/giga_norm` (`ARE`) -> `scoring/wenet_wer` | GigaSpeechBench route; requires `[giga]` |
| `asr.ar_dz.wer.giga_norm_dza_v1.wenet_wer_v1` | `ar-dz` | `normalization/giga_norm` (`DZA`) -> `scoring/wenet_wer` | GigaSpeechBench route; requires `[giga]` |
| `asr.ar_eg.wer.giga_norm_egy_v1.wenet_wer_v1` | `ar-eg` | `normalization/giga_norm` (`EGY`) -> `scoring/wenet_wer` | GigaSpeechBench route; requires `[giga]` |
| `asr.ar_iq.wer.giga_norm_irq_v1.wenet_wer_v1` | `ar-iq` | `normalization/giga_norm` (`IRQ`) -> `scoring/wenet_wer` | GigaSpeechBench route; requires `[giga]` |
| `asr.ar_ma.wer.giga_norm_mar_v1.wenet_wer_v1` | `ar-ma` | `normalization/giga_norm` (`MAR`) -> `scoring/wenet_wer` | GigaSpeechBench route; requires `[giga]` |
| `asr.ar_sa.wer.giga_norm_sau_v1.wenet_wer_v1` | `ar-sa` | `normalization/giga_norm` (`SAU`) -> `scoring/wenet_wer` | GigaSpeechBench route; requires `[giga]` |
| `asr.ar_sy.wer.giga_norm_syr_v1.wenet_wer_v1` | `ar-sy` | `normalization/giga_norm` (`SYR`) -> `scoring/wenet_wer` | GigaSpeechBench route; requires `[giga]` |

### `mer`

| Pipeline ID | Language | Nodes | Notes |
|:------------|:---------|:------|:------|
| `asr.cs.mer.aispeech_norm_cs_v1.wenet_mer_v1` | `cs` | `normalization/aispeech_norm` -> `scoring/wenet_mer` | Default code-switch MER |
| `asr.cs.mer.canonical_itn_cs_v2.token_mer_v1` | `cs` | `normalization/canonical_itn` -> `scoring/token_mer` | Canonical ITN MER; requires `[canonical]` |

`pipeline_id` uses the canonical metric plus versioned computation nodes.
Canonical ITN pipelines are selected by exact `pipeline_id`, not by a separate
metric name.
`canonical_itn` v2 collapses apostrophe-s case-insensitively in English and
code-switch text. WER/MER can therefore differ from v1 on all-caps input;
Chinese normalization is unchanged, but its route ID also advances because
the profiles share one versioned node. The default ASR routes are unchanged.

## Input Format

Tab-separated key-text files:

```text
utt_001<TAB>你好世界
utt_002<TAB>今天天气不错
```

Both `--ref-file` and `--hyp-file` use the same format, aligned by key.
Keys must be nonempty and unique within each file. Nonblank rows without a tab
are rejected with their input path and line number; malformed rows are not
silently dropped.

## CLI Usage

```bash
# Default Mandarin CER pipeline
sure-eval metric describe asr --language zh --metric cer --output /tmp/asr.json

# Default Arabic CER pipeline (written-form tokens are TN-normalized)
sure-eval agent plan asr --language ar --metric cer --json
sure-eval env setup --node normalization/nemo_norm
sure-eval metric describe asr --language ar --metric cer --output /tmp/asr_ar.json
sure-eval metric run --pipeline /tmp/asr_ar.json \
  --ref-file ref.txt --hyp-file hyp.txt --output-dir /tmp/asr_ar_eval

# Specific canonical Mandarin CER pipeline
sure-eval metric describe asr \
  --pipeline-id asr.zh.cer.canonical_itn_zh_v2.token_cer_v1 \
  --output /tmp/asr_canonical.json

sure-eval metric run --pipeline /tmp/asr_canonical.json \
  --ref-file ref.txt --hyp-file hyp.txt --output-dir /tmp/asr_eval

# Prepare and run the default Spanish FunASR ITN route
sure-eval agent plan asr --language es --metric wer --json
sure-eval env setup --node normalization/funasr_itn
sure-eval metric describe asr --language es --metric wer --output /tmp/asr_es.json
sure-eval metric run --pipeline /tmp/asr_es.json \
  --ref-file ref.txt --hyp-file hyp.txt --output-dir /tmp/asr_es_eval
```

## Python API

```python
from sure_eval.evaluation.scripts import run_task

report = run_task(
    "asr",
    ref_file="ref.txt",
    hyp_file="hyp.txt",
    pipeline_id="asr.zh.cer.canonical_itn_zh_v2.token_cer_v1",
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
- `normalization/funasr_itn` - default ITN for `ja`, `ko`, `es`, `fr`, `de`,
  `ru`, `pt`, `vi`, `id`, and `tl`. Its node-local runtime must be prepared
  once before scoring; see the [node README](../../src/sure_eval/evaluation/nodes/normalization/funasr_itn/README.md).
- `normalization/nemo_norm` - default Arabic TN (`ar_tn`) before CER. Its
  frozen node-local runtime must be prepared once before scoring; see the
  [node README](../../src/sure_eval/evaluation/nodes/normalization/nemo_norm/README.md).
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

## GigaSpeechBench Normalization

`normalization/giga_norm` vendors the text normalization published with
[GigaSpeechBench](https://github.com/SpeechColab/GigaSpeechBench), pinned to
upstream commit `ca782bf`. It is an explicit alternative for languages that
already have a SURE-EVAL route, and the default route for the additional
GigaSpeechBench language profiles. Install it with `pip install -e ".[giga]"`.

The node README records the paper, upstream license, vendored revision, and
reproduction notes: [giga_norm README](../../src/sure_eval/evaluation/nodes/normalization/giga_norm/README.md).
