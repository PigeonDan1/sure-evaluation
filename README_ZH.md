<div align="center">

# 🎯 SURE-EVAL

**面向语音与音频任务的可复现、版本化管理评测框架**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Core](https://github.com/PigeonDan1/sure-evaluation/actions/workflows/core.yml/badge.svg)](https://github.com/PigeonDan1/sure-evaluation/actions/workflows/core.yml)
[![GitHub stars](https://img.shields.io/github/stars/PigeonDan1/sure-evaluation.svg?style=social&label=Stars)](https://github.com/PigeonDan1/sure-evaluation/stargazers)

🌐 [English](./README.md) · [中文](./README_ZH.md) · [📖 文档](./docs/)

</div>

---

## ✨ SURE-EVAL 是什么？

SURE-EVAL 是一个面向语音与音频基准测试的**确定性评测系统**：

- 🧩 **流水线路由（Route）** —— 每个指标都是一条声明好的、由版本化节点组成的链。
- 📊 **可复现报告** —— 每次运行都会生成 `report.json` + `pipeline_description.json`。
- ⚖️ **公平比较** —— 相同路由 + 相同输入永远得到相同分数。

你可以把它当作 **CLI 工具**、**Python 库**，或嵌入到更大的 Agent 工作流中使用。

---

## 🚀 30 秒快速上手

```bash
# 安装轻量基础包
pip install -e .

# 检查安装
sure-eval doctor

# 描述并运行一个 ASR 指标
sure-eval metric describe asr --language zh --metric cer --output /tmp/asr.json
sure-eval metric run --pipeline /tmp/asr.json \
  --ref-file ref.txt --hyp-file hyp.txt --output-dir /tmp/asr_eval

# 查看分数
cat /tmp/asr_eval/report.json | grep score
```

输入文件为制表符分隔格式：`<key>\t<text>`。

---

## 📋 支持的任务

| 任务 | 指标 | 说明 | 指南 |
|:-----|:-----|:-----|:-----|
| **ASR** | WER、CER、MER | 纯文本，基础包可用 | [docs/tasks/asr.md](./docs/tasks/asr.md) |
| **S2TT** | BLEU、chrF2、XCOMET-XL、BLEURT-20 | 基础 + 可选重指标 | [docs/tasks/s2tt.md](./docs/tasks/s2tt.md) |
| **SD** | DER | 需 `[diarization]` | [docs/tasks/sd.md](./docs/tasks/sd.md) |
| **SA-ASR** | cpWER、DER | 需 `[diarization]` | [docs/tasks/sa_asr.md](./docs/tasks/sa_asr.md) |
| **TTS** | CER/WER、说话人相似度、MOS | 可选转录 + 打分节点 | [docs/tasks/tts.md](./docs/tasks/tts.md) |
| **VC** | CER/WER、说话人相似度、MOS | 可选转录 + 打分节点 | [docs/tasks/vc.md](./docs/tasks/vc.md) |
| **分类 / SER / GR** | Accuracy | 纯文本，基础包可用 | [docs/tasks/classification.md](./docs/tasks/classification.md) |
| **SLU** | Accuracy | 纯文本，基础包可用 | [docs/tasks/slu.md](./docs/tasks/slu.md) |
| **KWS** | accuracy、precision、recall、F1、FRR、FAR | 基础 + 可选节点 | [docs/tasks/kws.md](./docs/tasks/kws.md) |

每份指南都列出了具体的 pipeline ID、节点、输入格式和 CLI 示例。

在 CLI 中查看任意任务的路由：

```bash
sure-eval metric describe <task> --help
```

---

## 📝 流水线输入格式

SURE-EVAL 使用显式的角色定位输入。

**制表符分隔的 key-text 文件**（用于 ASR、S2TT、分类）：

```text
utt_001\t你好世界
utt_002\t今天天气不错
```

**TTS 音频样本 JSONL**：

```jsonl
{"sample_id":"tts_001","prediction_audio":"out.wav","reference_text":"你好世界","reference_audio":"speaker.wav","language":"zh"}
```

**VC 音频样本 JSONL**：

```jsonl
{"sample_id":"vc_001","converted_audio":"converted.wav","reference_audio":"speaker.wav","reference_text":"你好世界","language":"zh"}
```

完整格式说明：[docs/pipeline_inputs.md](docs/pipeline_inputs.md)。

---

## 🛠️ 安装可选组件

```bash
# 基础包
pip install sure-evaluation

# 开发
pip install -e ".[dev]"

# 可选 extras
pip install "sure-evaluation[diarization]"  # SD / SA-ASR 所需的 MeetEval
pip install "sure-evaluation[audio]"        # 本地音频辅助库
pip install "sure-evaluation[download]"     # Hugging Face / ModelScope 下载辅助
pip install "sure-evaluation[wetext]"       # WeTextProcessing 归一化

# 把缓存放到大容量磁盘
export SURE_EVAL_CACHE_DIR=/path/to/sure-eval-cache
```

准备一个重指标环境：

```bash
sure-eval env list
sure-eval env setup --task tts --language zh --metrics tts_cer,dnsmos --dry-run
sure-eval env setup --task tts --language zh --metrics tts_cer,dnsmos
```

详见 [docs/installation.md](docs/installation.md) 和 [docs/environment.md](docs/environment.md)。

---

## 🐍 Python API

```python
from sure_eval.evaluation.scripts import describe_pipeline, run_task

# 查看路由
desc = describe_pipeline("asr", language="zh", metric="cer")
print(desc.node_ids)
# ('normalization/aispeech_norm', 'scoring/wenet_cer')

# 运行并获取报告
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
print(evaluator.evaluate("asr", "ref.txt", "hyp.txt")["cer"])
```

---

## 🏗️ 设计一览

```text
用户请求
    │
    ▼
sure-eval metric describe  ──►  pipeline JSON（路由 + 节点）
    │
    ▼
sure-eval metric run       ──►  report.json + pipeline_description.json
```

每个指标都是一条**路由**：任务 + 语言 + 版本化节点的声明式组合。路由定义在 `tasks/<task>/routes.yaml` 中；节点元数据在 `nodes/<stage>/<name>/manifest.yaml` 和 `node_env.yaml` 中。

这让每个分数都可以追溯到产生它的确切代码、配置和输入。

---

## 🤝 如何贡献

1. 在 `src/sure_eval/evaluation/nodes/` 下添加或扩展节点。
2. 编写 `manifest.yaml` 和 `node_env.yaml`（如节点较复杂）。
3. 在 `tasks/<task>/routes.yaml` 中添加或更新路由。
4. 添加测试和文档。
5. 用 `sure-eval metric describe` 和 `sure-eval metric run` 验证。

详见 [docs/contributing.md](docs/contributing.md) 和 [docs/add_a_metric.md](docs/add_a_metric.md)。

---

## 📄 许可证

MIT License。详见 [LICENSE](LICENSE)。
