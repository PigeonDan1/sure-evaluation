<div align="center">

# 🎯 SURE-EVAL

**面向语音与音频任务的可复现、版本化管理评测框架**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/PigeonDan1/sure-evaluation.svg?style=social&label=Stars)](https://github.com/PigeonDan1/sure-evaluation/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/PigeonDan1/sure-evaluation.svg?style=social&label=Forks)](https://github.com/PigeonDan1/sure-evaluation/network/members)

🌐 [English](./README.md) · [中文](./README_ZH.md)

</div>

---

## ✨ SURE-EVAL 是什么？

SURE-EVAL 不是一堆临时拼凑的评测脚本，而是一个**确定性的评测系统**，核心设计包括：

- 🧩 **流水线路由（Route）** —— 每个指标都是一条声明好的、由版本化节点组成的链。
- 📊 **可复现报告** —— 每次运行都会生成 `report.json` + `pipeline_description.json`。
- ⚖️ **公平比较** —— 相同输入在相同路由下永远得到相同分数，全程可追溯。

你可以把它当作 **CLI 工具**、**Python 库**，或嵌入到更大的 Agent 工作流中使用。

---

## 🚀 30 秒快速上手

```bash
# 1. 安装
pip install -e .

# 2. 检查基础包
sure-eval doctor

# 3. 描述评测路由
sure-eval metric describe asr --language zh --metric cer --output /tmp/asr.json

# 4. 运行评测
sure-eval metric run --pipeline /tmp/asr.json \
  --ref-file ref.txt --hyp-file hyp.txt --output-dir /tmp/asr_eval

# 5. 查看分数
cat /tmp/asr_eval/report.json | grep score
```

输入文件为制表符分隔格式：`<key>\t<text>`。

---

## 📋 支持的任务

点击任务查看支持的指标和快速示例。

<details>
<summary>🎙️ <b>ASR</b> — 自动语音识别</summary>

指标：`WER`（英文） · `CER`（中文） · `MER`（中英混合）

```bash
sure-eval metric describe asr --language zh --metric cer --output /tmp/asr.json
sure-eval metric run --pipeline /tmp/asr.json \
  --ref-file ref.txt --hyp-file hyp.txt --output-dir /tmp/asr_eval
```
</details>

<details>
<summary>🌐 <b>S2TT</b> — 语音翻译</summary>

指标：`BLEU` · `chrF2` · `XCOMET-XL` · `BLEURT-20`

```bash
sure-eval metric describe s2tt --language zh --metric bleu --output /tmp/s2tt.json
sure-eval metric run --pipeline /tmp/s2tt.json \
  --ref-file ref.txt --hyp-file hyp.txt --src-file src.txt \
  --output-dir /tmp/s2tt_eval
```
</details>

<details>
<summary>👥 <b>SD</b> — 说话人分割</summary>

指标：`DER`

```bash
sure-eval metric describe sd --metric der --output /tmp/sd.json
sure-eval metric run --pipeline /tmp/sd.json \
  --ref-file ref.rttm --hyp-file hyp.rttm --output-dir /tmp/sd_eval
```
</details>

<details>
<summary>🎙️👥 <b>SA-ASR</b> — 说话人感知语音识别</summary>

指标：`cpWER` · 辅助 `DER`

```bash
sure-eval metric describe sa_asr --metric cpwer --output /tmp/sa_asr.json
sure-eval metric run --pipeline /tmp/sa_asr.json \
  --ref-file ref.stm --hyp-file hyp.stm --output-dir /tmp/sa_asr_eval
```
</details>

<details>
<summary>🔊 <b>TTS / VC</b> — 语音合成 / 语音转换</summary>

指标：`CER/WER` · 说话人相似度（`wavlm-large` · `ecapa-tdnn` · `eres2net`） · MOS（`DNSMOS` · `WV-MOS` · `UTMOS`）

```bash
sure-eval metric describe tts --language zh --metrics tts_cer,sim/wavlm-large \
  --output /tmp/tts.json

sure-eval metric run --pipeline /tmp/tts.json \
  --samples-jsonl samples.jsonl --output-dir /tmp/tts_eval \
  --device cuda --cache-dir /tmp/sure_eval_cache
```

TTS 样例行：

```json
{"sample_id":"tts_001","prediction_audio":"out.wav","reference_text":"你好世界","reference_audio":"speaker.wav","language":"zh"}
```

VC 样例行：

```json
{"sample_id":"vc_001","converted_audio":"converted.wav","source_audio":"source.wav","reference_audio":"speaker.wav","reference_text":"你好世界","language":"zh"}
```
</details>

<details>
<summary>🏷️ <b>分类 / SER / GR</b></summary>

指标：`Accuracy`

```bash
sure-eval metric describe classification --output /tmp/cls.json
sure-eval metric run --pipeline /tmp/cls.json \
  --ref-file ref.txt --hyp-file hyp.txt --output-dir /tmp/cls_eval
```

SER/GR 使用同一路由，并内置标签映射。
</details>

<details>
<summary>🧠 <b>SLU</b> — 口语语言理解</summary>

指标：`Accuracy`

```bash
sure-eval metric describe slu --output /tmp/slu.json
sure-eval metric run --pipeline /tmp/slu.json \
  --ref-file ref.txt --hyp-file hyp.txt \
  --prompt-jsonl prompt.jsonl --output-dir /tmp/slu_eval
```
</details>

<details>
<summary>🔑 <b>KWS</b> — 关键词唤醒</summary>

指标：`accuracy` · `precision` · `recall` · `F1` · `false_reject_rate` · `false_alarm_rate`

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

# 1️⃣ 查看路由
desc = describe_pipeline("asr", language="zh", metric="cer")
print(desc.node_ids)
# ('normalization/aispeech_norm', 'scoring/wenet_cer')

# 2️⃣ 运行评测
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

传统一行式调用：

```python
from sure_eval.evaluation import SUREEvaluator

evaluator = SUREEvaluator(language="zh")
result = evaluator.evaluate("asr", ref_file="ref.txt", hyp_file="hyp.txt")
print(result["cer"])
```

---

## 🏗️ 设计理念：SURE-EVAL 有什么不同？

大多数评测代码只是一堆脚本。SURE-EVAL 把评测视为一条**受管理的流水线**：

| 概念 | 含义 |
|:----|:-----|
| 🛤️ **路由（Route）** | 一个声明好的「任务 + 语言 + 指标」组合，例如 `asr.zh.cer.aispeech_norm.wenet_cer` |
| 🧩 **节点（Node）** | 可复用、带版本的阶段：归一化（normalization）、转录（transcription）、打分（scoring） |
| 📜 **清单（Manifest）** | YAML 文件，声明节点/任务的输入输出与依赖 |
| 📊 **报告（Report）** | `report.json` 记录分数、流水线轨迹、输入文件 |
| 🔗 **流水线描述（Pipeline Description）** | `pipeline_description.json` 记录所选节点、版本和配置文件路径 |

这意味着：

1. ⚖️ **公平比较** —— 使用同一路由评测的两个模型，节点与超参数完全一致。
2. 🔍 **完全可追溯** —— 每个分数都能追溯到产生它的代码版本、配置和输入。
3. 🧱 **易于扩展** —— 新增指标只需添加节点 + 路由，无需修改公共脚本。

---

## 🤝 如何贡献一个新指标

1. 🧩 **添加节点**：在 `nodes/scoring/<name>/`（或 `nodes/normalization/`、`nodes/transcription/`）下实现。
2. 📜 **编写 `manifest.yaml`**：声明 `id`、`version`、`stage`、输入输出模式及依赖。
3. 🛤️ **添加路由**：在 `tasks/<task>/routes.yaml` 中关联任务 → 语言 → 指标 → 节点。
4. 🔧 **实现任务流水线**：如有需要，在 `tasks/<task>/pipeline.py` 中组合节点。
5. 🧪 **添加测试**：节点测试 + 任务路由测试 + 脚本入口测试。
6. ✅ **运行验证**：使用 `sure-eval metric describe` 和 `sure-eval metric run` 验证。

保持包装层轻薄。如果指标来自外部工具包，直接包装，不要改变其行为；如需改动，必须明确记录并评审。

---

## 📦 输出文件

每次运行都会生成：

- 📄 `report.json` —— 分数、指标、任务、输入文件、节点轨迹、详细统计。
- 📄 `pipeline_description.json` —— 路由、节点、版本、配置文件路径、输入约定。

这让每一次结果都可审计、可复现。

---

## ⚙️ 安装

```bash
git clone git@github.com:PigeonDan1/sure-evaluation.git
cd sure-evaluation
pip install -e .
```

需要 Python >= 3.10。基础安装支持轻量文本指标和路由查看。重型 ASR、TTS、VC、MOS、说话人相似度和学习型翻译指标走可选的节点本地环境，因此用户不需要一次性安装所有工具链。

```bash
# 可选：把缓存和下载工具资产放到大容量磁盘
export SURE_EVAL_CACHE_DIR=/path/to/sure-eval-cache

# 查看基础包和可选节点状态
sure-eval doctor
sure-eval env list

# 创建环境前先查看某个可选节点的计划
sure-eval env setup --node scoring/dnsmos --dry-run
sure-eval env check --node scoring/dnsmos
sure-eval env download --node scoring/dnsmos --dry-run
```

说话人分割和说话人感知 ASR 需要 `meeteval`：

```bash
pip install -e ".[diarization]"
```

模型/资产下载辅助命令需要 download extra：

```bash
pip install -e ".[download]"
```

本地虚拟环境、checkpoint、运行清单、日志和模型权重都已放入 gitignore。它们可以保留在本机使用，但不应提交到仓库。

更多文档：

- [安装](docs/installation.md)
- [环境管理](docs/environment.md)
- [贡献说明](docs/contributing.md)
- [添加指标](docs/add_a_metric.md)
- [可复现性](docs/reproducibility.md)

---

## 📄 许可证

MIT License。详见 [LICENSE](LICENSE)。
