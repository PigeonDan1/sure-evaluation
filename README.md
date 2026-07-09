<div align="center">

# 🎯 SURE-EVAL

**A reproducible, version-managed evaluation framework for speech & audio tasks.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Core](https://github.com/PigeonDan1/sure-evaluation/actions/workflows/core.yml/badge.svg)](https://github.com/PigeonDan1/sure-evaluation/actions/workflows/core.yml)
[![GitHub stars](https://img.shields.io/github/stars/PigeonDan1/sure-evaluation.svg?style=social&label=Stars)](https://github.com/PigeonDan1/sure-evaluation/stargazers)

🌐 [English](./README.md) · [中文](./README_ZH.md) · [📖 Docs](./docs/)

</div>

---

## ✨ What is SURE-EVAL?

SURE-EVAL is a **deterministic evaluation system** for speech and audio benchmarks:

- 🧩 **Pipeline routes** — every metric is a declared chain of versioned nodes.
- 📊 **Reproducible reports** — every run writes `report.json` + `pipeline_description.json`.
- ⚖️ **Fair comparison** — same route + same inputs always produce the same score.

Use it as a **CLI**, a **Python library**, or a module in larger agent workflows.

---

## 🚀 30-Second Quick Start

```bash
# Install the lightweight base package
pip install -e .

# Check the installation
sure-eval doctor

# Describe and run an ASR metric
sure-eval metric describe asr --language zh --metric cer --output /tmp/asr.json
sure-eval metric run --pipeline /tmp/asr.json \
  --ref-file ref.txt --hyp-file hyp.txt --output-dir /tmp/asr_eval

# View the score
cat /tmp/asr_eval/report.json | grep score
```

Input files are tab-separated: `<key>\t<text>`.

---

## 📦 What's in the Box?

| Layer | Included in base install | Optional node-local setup |
|:------|:-------------------------|:--------------------------|
| CLI, routing, reports | ✅ | — |
| ASR WER/CER, classification, SLU, KWS | ✅ | — |
| S2TT BLEU/chrF2 | ✅ | — |
| SD DER, SA-ASR cpWER | needs `[diarization]` | — |
| TTS/VC WER/CER | ✅ route | ASR transcription node |
| TTS/VC speaker similarity | — | `scoring/wavlm_large_sim`, etc. |
| TTS/VC MOS (DNSMOS, WV-MOS, UTMOS) | — | `scoring/dnsmos`, etc. |
| S2TT XCOMET-XL, BLEURT-20 | — | `scoring/xcomet_xl`, `scoring/bleurt_20` |

The base install is intentionally lightweight. Heavy metrics run in isolated node-local environments so you only install what you need.

---

## 📋 Supported Tasks

| Task | Metrics | Notes | Guide |
|:-----|:--------|:------|:------|
| **ASR** | WER, CER, MER | Text-only, base install | [docs/tasks/asr.md](./docs/tasks/asr.md) |
| **S2TT** | BLEU, chrF2, XCOMET-XL, BLEURT-20 | Base + optional heavy metrics | [docs/tasks/s2tt.md](./docs/tasks/s2tt.md) |
| **SD** | DER | Requires `[diarization]` | [docs/tasks/sd.md](./docs/tasks/sd.md) |
| **SA-ASR** | cpWER, DER | Requires `[diarization]` | [docs/tasks/sa_asr.md](./docs/tasks/sa_asr.md) |
| **TTS** | CER/WER, speaker similarity, MOS | Optional transcription + scoring nodes | [docs/tasks/tts.md](./docs/tasks/tts.md) |
| **VC** | CER/WER, speaker similarity, MOS | Optional transcription + scoring nodes | [docs/tasks/vc.md](./docs/tasks/vc.md) |
| **Classification / SER / GR** | Accuracy | Text-only, base install | [docs/tasks/classification.md](./docs/tasks/classification.md) |
| **SLU** | Accuracy | Text-only, base install | [docs/tasks/slu.md](./docs/tasks/slu.md) |
| **KWS** | accuracy, precision, recall, F1, FRR, FAR | Base + optional node | [docs/tasks/kws.md](./docs/tasks/kws.md) |

Each guide lists the exact pipeline IDs, nodes, input formats, and CLI examples.

Click any task in the CLI for its route:

```bash
sure-eval metric describe <task> --help
```

---

## 🛠️ Installing Optional Parts

```bash
# Base package
pip install sure-evaluation

# Development
pip install -e ".[dev]"

# Optional extras
pip install "sure-evaluation[diarization]"  # MeetEval for SD / SA-ASR
pip install "sure-evaluation[audio]"        # Local audio helpers
pip install "sure-evaluation[download]"     # Hugging Face / ModelScope download helpers
pip install "sure-evaluation[wetext]"       # WeTextProcessing normalization

# Put caches on a large disk
export SURE_EVAL_CACHE_DIR=/path/to/sure-eval-cache
```

Prepare a heavy metric environment:

```bash
sure-eval env list
sure-eval env setup --task tts --language zh --metrics tts_cer,dnsmos --dry-run
sure-eval env setup --task tts --language zh --metrics tts_cer,dnsmos
```

See [docs/installation.md](docs/installation.md) and [docs/environment.md](docs/environment.md) for details.

---

## 🐍 Python API

```python
from sure_eval.evaluation.scripts import describe_pipeline, run_task

# Inspect the route
desc = describe_pipeline("asr", language="zh", metric="cer")
print(desc.node_ids)
# ('normalization/aispeech_norm', 'scoring/wenet_cer')

# Run and get a report
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

Legacy one-liner:

```python
from sure_eval.evaluation import SUREEvaluator

evaluator = SUREEvaluator(language="zh")
print(evaluator.evaluate("asr", "ref.txt", "hyp.txt")["cer"])
```

---

## 🏗️ Design at a Glance

```text
User request
    │
    ▼
sure-eval metric describe  ──►  pipeline JSON (route + nodes)
    │
    ▼
sure-eval metric run       ──►  report.json + pipeline_description.json
```

Every metric is a **route**: a declared combination of task, language, and versioned nodes. Routes live in `tasks/<task>/routes.yaml`; node metadata lives in `nodes/<stage>/<name>/manifest.yaml` and `node_env.yaml`.

This makes every score traceable to the exact code, config, and inputs that produced it.

---

## 🤝 How to Contribute

1. Add or extend a node under `src/sure_eval/evaluation/nodes/`.
2. Write `manifest.yaml` and `node_env.yaml` (if non-trivial).
3. Add or update routes in `tasks/<task>/routes.yaml`.
4. Add tests and docs.
5. Run `sure-eval metric describe` and `sure-eval metric run` to verify.

See [docs/contributing.md](docs/contributing.md) and [docs/add_a_metric.md](docs/add_a_metric.md).

---

## 📄 License

MIT License. See [LICENSE](LICENSE).
