<div align="center">

# SURE-EVALUATION

**Build an evaluation pipeline you can explain, reproduce, compare, and share.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Core](https://github.com/PigeonDan1/sure-evaluation/actions/workflows/core.yml/badge.svg)](https://github.com/PigeonDan1/sure-evaluation/actions/workflows/core.yml)
[![GitHub stars](https://img.shields.io/github/stars/PigeonDan1/sure-evaluation.svg?style=social&label=Stars)](https://github.com/PigeonDan1/sure-evaluation/stargazers)

[English](./README.md) | [中文](./README_ZH.md) | [Documentation](./docs/) | [Open Bench](https://www.open-bench.net/sure)

</div>

## What SURE-EVALUATION Is

SURE-EVALUATION is a general system evaluation framework. It integrates
evaluation tools for tasks such as speech recognition, speech generation,
speech enhancement, translation, classification, and language understanding,
while turning the complete evaluation process into versioned nodes.

You can choose a task-specific pipeline and its tools, such as different
normalizers, transcription models, conversions, or scoring implementations.
Every evaluation records the full path from model output to final score and
writes it as structured data. A result is therefore tied to the exact inputs,
pipeline identity, node sequence, configuration, and report that produced it.

The framework uses a lightweight root package and independently managed
optional node environments. This design supports a broad task surface without
forcing every user to install every model and tool.

## Why Use It

- **Broad coverage:** one framework covers text, audio, generation,
  recognition, translation, classification, and structured detection tasks.
- **Convenient:** ask the CLI for the available pipelines and required inputs,
  then prepare only the environment selected by the exact pipeline.
- **Trustworthy and reviewable:** each run records `pipeline_id`, ordered
  computation nodes, input contracts, and structured output artifacts.
- **Comparable:** results can be compared against the same explicit evaluation
  definition instead of an ambiguous metric label.
- **Customizable:** the same metric can have several registered routes, so you
  can select a different normalizer, model, backend, or score implementation
  without renaming the metric.

## How A Pipeline Works

SURE-EVALUATION treats every score-affecting step as a node:

```text
model output
    -> validation / conversion
    -> transcription / normalization
    -> scoring
    -> report.json + pipeline_description.json
```

A `pipeline_id` names an exact computation as
`task.language.metric.node_version...`. For example:

```text
asr.en.wer.whisper_norm_english_v1.wenet_wer_v1
```

`metric` stays globally canonical, such as `wer`, `cer`, `spk_sim`, `dnsmos`,
`wv_mos`, or `utmos`. When two routes report the same metric, their
`pipeline_id` differs because their node chain differs. Multi-metric requests
are bundles whose identities contain their atomic member pipelines.

## Pipeline Atlas

The committed catalog can be drawn as one map: every atomic pipeline is a
colored ribbon that flows from its task through frontend, transcription,
validation, normalization, and scoring nodes into a report.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/atlas/pipeline_atlas_dark.svg">
  <img src="docs/atlas/pipeline_atlas.svg" alt="Animated map of every registered SURE pipeline, drawn as colored ribbons from task to report" width="100%">
</picture>

[`docs/atlas/index.html`](docs/atlas/index.html) is the interactive version of
the same map, with search, per-task filtering, hover tracing, and a catalog
table; open it in a browser from a local clone. Both views are generated from
`docs/pipeline_catalog.jsonl`:

```bash
python scripts/generate_pipeline_atlas.py
```

## Quick Start

SURE-EVALUATION is currently installed from source, not from PyPI. The commands
below create an isolated environment and run a committed, text-only ASR example
that does not download a model or create a node-local environment.

```bash
git clone https://github.com/PigeonDan1/sure-evaluation.git
cd sure-evaluation

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip check
sure-eval doctor
```

Discover all English ASR WER routes. The table marks the default, prints every
exact `pipeline_id`, shows the ordered computation nodes and required inputs,
and identifies optional setup:

```bash
sure-eval metric routes asr --language en --metric wer
```

Select one exact route and write its executable pipeline JSON:

```bash
sure-eval metric describe asr \
  --pipeline-id asr.en.wer.whisper_norm_english_v1.wenet_wer_v1 \
  --output .sure-eval-demo/pipeline.json
```

Inspect the setup plan and validate only the environments used by that exact
pipeline:

```bash
sure-eval env setup --pipeline .sure-eval-demo/pipeline.json --dry-run
sure-eval env check --pipeline .sure-eval-demo/pipeline.json
```

Run the pipeline and inspect the structured report:

```bash
sure-eval metric run \
  --pipeline .sure-eval-demo/pipeline.json \
  --ref-file examples/readme/asr_en_ref.txt \
  --hyp-file examples/readme/asr_en_hyp.txt \
  --output-dir .sure-eval-demo/asr-en-wer \
  --validate-env

python -m json.tool .sure-eval-demo/asr-en-wer/report.json
python -m json.tool .sure-eval-demo/asr-en-wer/pipeline_description.json
```

The same exact `pipeline_id` appears in the discovery result, pipeline JSON,
run summary, `report.json`, and `pipeline_description.json`.

## Select And Prepare Other Pipelines

Use canonical metric names to discover alternatives, then copy the exact
`pipeline_id` you intend to run:

```bash
# Three Mandarin ASR CER normalization/scoring routes
sure-eval metric routes asr --language zh --metric cer

# Three TTS speaker-similarity backends
sure-eval metric routes tts --language zh --metric spk_sim

# Three KWS input contracts for the same macro_recall metric
sure-eval metric routes kws --metric macro_recall

# Stable machine-readable discovery output
sure-eval metric routes tts --language zh --metric cer --json
```

For a pipeline with optional models or tools, prepare only its declared nodes:

```bash
sure-eval metric describe tts \
  --pipeline-id tts.zh.cer.qwen3_asr_1_7b_v1.punctuation_strip_norm_v1.wenet_cer_v1 \
  --output .sure-eval-demo/tts-qwen.json
sure-eval env setup --pipeline .sure-eval-demo/tts-qwen.json --dry-run
sure-eval env setup --pipeline .sure-eval-demo/tts-qwen.json
sure-eval env download --node transcription/qwen3_asr_1_7b --dry-run
sure-eval env download --node transcription/qwen3_asr_1_7b
sure-eval env check --pipeline .sure-eval-demo/tts-qwen.json
```

See [Installation](docs/installation.md) and
[Environment Management](docs/environment.md) before running heavyweight
nodes. Environment setup installs declared tools; model assets are downloaded
separately and should be reviewed with `env download --dry-run`. Checkpoints,
caches, and node-local virtual environments remain local and are excluded from
packages and Git.

## Supported Tasks

| Task | Canonical metrics | Guide |
|:--|:--|:--|
| ASR | WER, CER, MER | [ASR](docs/tasks/asr.md) |
| S2TT | BLEU, BLEU-char, chrF, XCOMET-XL, BLEURT-20 | [S2TT](docs/tasks/s2tt.md) |
| SD | DER | [SD](docs/tasks/sd.md) |
| SA-ASR | cpWER, DER companion result | [SA-ASR](docs/tasks/sa_asr.md) |
| TTS | CER, WER, speaker similarity, DNSMOS, WV-MOS, UTMOS | [TTS](docs/tasks/tts.md) |
| VC | CER, WER, speaker similarity, DNSMOS, WV-MOS, UTMOS | [VC](docs/tasks/vc.md) |
| SE | SI-SDR, STOI, PESQ, DNSMOS, WV-MOS, UTMOS | [SE](docs/tasks/se.md) |
| TSE | SI-SDR, speaker similarity, DNSMOS, WV-MOS, UTMOS, CER, WER | [TSE](docs/tasks/tse.md) |
| Classification / SER / GR | Accuracy | [Classification](docs/tasks/classification.md) |
| SLU | Accuracy | [SLU](docs/tasks/slu.md) |
| KWS | Accuracy, macro recall, precision, recall, F1, FRR, FAR | [KWS](docs/tasks/kws.md) |
| VAD | F1, false alarm, miss, NIST DCF, ROC AUC | [VAD](docs/tasks/vad.md) |

Each task guide defines its input contract, registered metrics, exact pipeline
IDs, nodes, and run examples. The generated
[pipeline catalog](docs/pipeline_catalog.md) contains all registered atomic
route instances for declared language profiles plus curated multi-metric
bundles.

## Customize And Share

Routes are declared in
`src/sure_eval/evaluation/tasks/<task>/routes.yaml`. Node metadata lives in
`src/sure_eval/evaluation/nodes/<stage>/<name>/manifest.yaml`; optional runtime
requirements live beside it in `node_env.yaml`. Adding a route changes routing
and identity metadata, while its computation remains owned by its versioned
nodes.

To contribute a task, metric, route, or node tool, start with
[Contributing](docs/contributing.md). It directs each PR type to a focused
manual and the repository PR template. Agents should also follow the
[Agent Contract](docs/agent_contract.md).

Community pipelines can be shared and explored on
[Open Bench](https://www.open-bench.net/sure), where usage and community
feedback help collaborators identify widely adopted evaluation routes.

Create an evaluation pipeline that is truly yours, then make it reviewable and
shareable.

## License

MIT License. See [LICENSE](LICENSE).
