<div align="center">

# 🎯 SURE-EVAL

**A reproducible, version-managed evaluation framework for speech & audio tasks.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/PigeonDan1/sure-evaluation.svg?style=social&label=Stars)](https://github.com/PigeonDan1/sure-evaluation/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/PigeonDan1/sure-evaluation.svg?style=social&label=Forks)](https://github.com/PigeonDan1/sure-evaluation/network/members)

🌐 [English](./README.md) · [中文](./README_ZH.md)

</div>

---

## ✨ What is SURE-EVAL?

SURE-EVAL is not a collection of ad-hoc metric scripts. It is a **deterministic evaluation system** built around:

- 🧩 **Pipeline routes** — every metric is a declared chain of versioned nodes.
- 📊 **Reproducible reports** — every run writes `report.json` + `pipeline_description.json`.
- ⚖️ **Fair comparison** — the same inputs always produce the same score, with full traceability.

Use it as a **CLI**, a **Python library**, or a module inside larger agent workflows.

---

## 🚀 30-Second Quick Start

```bash
# 1. Install
pip install -e .

# 2. Check the base package
sure-eval doctor

# 3. Describe the metric route
sure-eval metric describe asr --language zh --metric cer --output /tmp/asr.json

# 4. Run
sure-eval metric run --pipeline /tmp/asr.json \
  --ref-file ref.txt --hyp-file hyp.txt --output-dir /tmp/asr_eval

# 5. Get the score
cat /tmp/asr_eval/report.json | grep score
```

Input files are tab-separated: `<key>\t<text>`.

---

## 📋 Supported Tasks

Click a task to see its metrics and quick example.

<details>
<summary>🎙️ <b>ASR</b> — Automatic Speech Recognition</summary>

Metrics: `WER` (en) · `CER` (zh) · `MER` (code-switching)

```bash
sure-eval metric describe asr --language zh --metric cer --output /tmp/asr.json
sure-eval metric run --pipeline /tmp/asr.json \
  --ref-file ref.txt --hyp-file hyp.txt --output-dir /tmp/asr_eval
```
</details>

<details>
<summary>🌐 <b>S2TT</b> — Speech-to-Text Translation</summary>

Metrics: `BLEU` · `chrF2` · `XCOMET-XL` · `BLEURT-20`

```bash
sure-eval metric describe s2tt --language zh --metric bleu --output /tmp/s2tt.json
sure-eval metric run --pipeline /tmp/s2tt.json \
  --ref-file ref.txt --hyp-file hyp.txt --src-file src.txt \
  --output-dir /tmp/s2tt_eval
```
</details>

<details>
<summary>👥 <b>SD</b> — Speaker Diarization</summary>

Metric: `DER`

```bash
sure-eval metric describe sd --metric der --output /tmp/sd.json
sure-eval metric run --pipeline /tmp/sd.json \
  --ref-file ref.rttm --hyp-file hyp.rttm --output-dir /tmp/sd_eval
```
</details>

<details>
<summary>🎙️👥 <b>SA-ASR</b> — Speaker-Aware ASR</summary>

Metrics: `cpWER` · companion `DER`

```bash
sure-eval metric describe sa_asr --metric cpwer --output /tmp/sa_asr.json
sure-eval metric run --pipeline /tmp/sa_asr.json \
  --ref-file ref.stm --hyp-file hyp.stm --output-dir /tmp/sa_asr_eval
```
</details>

<details>
<summary>🔊 <b>TTS / VC</b> — Speech Synthesis & Voice Conversion</summary>

Metrics: `CER/WER` · speaker similarity (`wavlm-large` · `ecapa-tdnn` · `eres2net`) · MOS (`DNSMOS` · `WV-MOS` · `UTMOS`)

```bash
sure-eval metric describe tts --language zh --metrics tts_cer,sim/wavlm-large \
  --output /tmp/tts.json

sure-eval metric run --pipeline /tmp/tts.json \
  --samples-jsonl samples.jsonl --output-dir /tmp/tts_eval \
  --device cuda --cache-dir /tmp/sure_eval_cache
```

Sample row (TTS):

```json
{"sample_id":"tts_001","prediction_audio":"out.wav","reference_text":"你好世界","reference_audio":"speaker.wav","language":"zh"}
```

Sample row (VC):

```json
{"sample_id":"vc_001","converted_audio":"converted.wav","source_audio":"source.wav","reference_audio":"speaker.wav","reference_text":"你好世界","language":"zh"}
```
</details>

<details>
<summary>🏷️ <b>Classification / SER / GR</b></summary>

Metric: `Accuracy`

```bash
sure-eval metric describe classification --output /tmp/cls.json
sure-eval metric run --pipeline /tmp/cls.json \
  --ref-file ref.txt --hyp-file hyp.txt --output-dir /tmp/cls_eval
```

SER/GR use the same route with built-in label maps.
</details>

<details>
<summary>🧠 <b>SLU</b> — Spoken Language Understanding</summary>

Metric: `Accuracy`

```bash
sure-eval metric describe slu --output /tmp/slu.json
sure-eval metric run --pipeline /tmp/slu.json \
  --ref-file ref.txt --hyp-file hyp.txt \
  --prompt-jsonl prompt.jsonl --output-dir /tmp/slu_eval
```
</details>

<details>
<summary>🔑 <b>KWS</b> — Keyword Spotting</summary>

Metrics: `accuracy` · `precision` · `recall` · `F1` · `false_reject_rate` · `false_alarm_rate`

```bash
sure-eval metric describe kws --metric accuracy --output /tmp/kws.json
sure-eval metric run --pipeline /tmp/kws.json \
  --reference-jsonl ref.jsonl --sample-output pred.jsonl --output-dir /tmp/kws_eval
```
</details>

---

## 🐍 Python API

```python
from sure_eval.evaluation.scripts import describe_pipeline, run_task

# 1️⃣ Inspect the route
desc = describe_pipeline("asr", language="zh", metric="cer")
print(desc.node_ids)
# ('normalization/aispeech_norm', 'scoring/wenet_cer')

# 2️⃣ Run
report = run_task(
    "asr",
    ref_file="ref.txt",
    hyp_file="hyp.txt",
    language="zh",
    metric="cer",
    output_dir="/tmp/asr_eval",
)
print(report.score)
```

For legacy one-liners:

```python
from sure_eval.evaluation import SUREEvaluator

evaluator = SUREEvaluator(language="zh")
result = evaluator.evaluate("asr", ref_file="ref.txt", hyp_file="hyp.txt")
print(result["cer"])
```

---

## 🏗️ Design: Why SURE-EVAL is Different

Most evaluation code is a pile of scripts. SURE-EVAL treats evaluation as a **managed pipeline**:

| Concept | What it means |
|:--------|:--------------|
| 🛤️ **Route** | A declared task + language + metric combination, e.g. `asr.zh.cer.aispeech_norm.wenet_cer` |
| 🧩 **Node** | A reusable, versioned stage: normalization, transcription, or scoring |
| 📜 **Manifest** | YAML files that declare what a node/task expects and produces |
| 📊 **Report** | `report.json` records score + pipeline trace + input files |
| 🔗 **Pipeline Description** | `pipeline_description.json` records selected nodes, versions, and config paths |

This means:

1. ⚖️ **Fair comparison** — two models evaluated with the same route use the exact same nodes and hyperparameters.
2. 🔍 **Full traceability** — every score can be traced back to the code version, config, and inputs that produced it.
3. 🧱 **Extensibility** — adding a new metric means adding a node + a route, not editing a shared script.

---

## 🤝 How to Contribute a New Metric

1. 🧩 **Add a node** under `nodes/scoring/<name>/` (or `nodes/normalization/`, `nodes/transcription/`).
2. 📜 **Write `manifest.yaml`** declaring `id`, `version`, `stage`, input/output schema, and dependencies.
3. 🛤️ **Add a route** in `tasks/<task>/routes.yaml` linking task → language → metric → nodes.
4. 🔧 **Implement the task pipeline** in `tasks/<task>/pipeline.py` if needed.
5. 🧪 **Add tests**: node test + task route test + script entrypoint test.
6. ✅ **Run and verify** with `sure-eval metric describe` and `sure-eval metric run`.

Keep wrappers thin. If a metric comes from an external toolkit, wrap it without changing its behavior unless the change is intentional and documented.

---

## 📦 Output Files

Every run writes:

- 📄 `report.json` — score, metric, task, input files, node trace, details.
- 📄 `pipeline_description.json` — route, nodes, versions, config paths, input contracts.

This makes every result auditable and reproducible.

---

## ⚙️ Installation

```bash
git clone git@github.com:PigeonDan1/sure-evaluation.git
cd sure-evaluation
pip install -e .
```

Python >= 3.10 is required. The base install supports lightweight text metrics and route inspection. Heavy ASR, TTS, VC, MOS, speaker-similarity, and learned translation metrics use optional node-local environments, so users do not need to install every toolkit up front.

```bash
# Optional: place caches and downloaded tool assets on a large disk
export SURE_EVAL_CACHE_DIR=/path/to/sure-eval-cache

# Inspect package and optional node status
sure-eval doctor
sure-eval env list

# Plan one optional node environment before creating it
sure-eval env setup --node scoring/dnsmos --dry-run
sure-eval env check --node scoring/dnsmos
sure-eval env download --node scoring/dnsmos --dry-run
```

Speaker diarization and speaker-aware ASR require `meeteval`:

```bash
pip install -e ".[diarization]"
```

Asset download helpers require the download extra:

```bash
pip install -e ".[download]"
```

Local virtual environments, checkpoints, runtime manifests, logs, and model artifacts are ignored by git. Keep them on disk for local use, but do not commit them.

More details:

- [Installation](docs/installation.md)
- [Environment management](docs/environment.md)
- [Contributing](docs/contributing.md)
- [Add a metric](docs/add_a_metric.md)
- [Reproducibility](docs/reproducibility.md)

---

## 📄 License

MIT License. See [LICENSE](LICENSE).
