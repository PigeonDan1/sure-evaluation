<div align="center">

# 🎯 SURE-EVAL

**面向语音与音频任务的可复现、版本化管理评测框架**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Core](https://github.com/PigeonDan1/sure-evaluation/actions/workflows/core.yml/badge.svg)](https://github.com/PigeonDan1/sure-evaluation/actions/workflows/core.yml)
[![GitHub stars](https://img.shields.io/github/stars/PigeonDan1/sure-evaluation.svg?style=social&label=Stars)](https://github.com/PigeonDan1/sure-evaluation/stargazers)

🌐 [English](./README.md) · [中文](./README_ZH.md) · [📖 文档](./docs/) · [🚀 Demo](https://sure-eval.com/)

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
| **ASR** | WER、CER、MER | 纯文本；canonical 归一化路由需 `[canonical]` | [docs/tasks/asr.md](./docs/tasks/asr.md) |
| **S2TT** | BLEU、chrF2、XCOMET-XL、BLEURT-20 | 基础 + 可选重指标 | [docs/tasks/s2tt.md](./docs/tasks/s2tt.md) |
| **SD** | DER | 需 `[diarization]` | [docs/tasks/sd.md](./docs/tasks/sd.md) |
| **SA-ASR** | cpWER、DER | 需 `[diarization]` | [docs/tasks/sa_asr.md](./docs/tasks/sa_asr.md) |
| **TTS** | CER/WER、说话人相似度、MOS | 可选转录 + 打分节点 | [docs/tasks/tts.md](./docs/tasks/tts.md) |
| **VC** | CER/WER、说话人相似度、MOS | 可选转录 + 打分节点 | [docs/tasks/vc.md](./docs/tasks/vc.md) |
| **分类 / SER / GR** | Accuracy | 纯文本，基础包可用 | [docs/tasks/classification.md](./docs/tasks/classification.md) |
| **SLU** | Accuracy | 纯文本，基础包可用 | [docs/tasks/slu.md](./docs/tasks/slu.md) |
| **KWS** | accuracy、precision、recall、F1、FRR、FAR | 基础 + 可选节点 | [docs/tasks/kws.md](./docs/tasks/kws.md) |

每份指南都列出了具体的 pipeline ID、节点、输入格式和 CLI 示例。

如需 metrics → pipelines → nodes 的机器可读对照表，查看 [docs/pipeline_catalog.jsonl](./docs/pipeline_catalog.jsonl) 和 [docs/pipeline_catalog.md](./docs/pipeline_catalog.md)。

在 CLI 中查看任意任务的路由：

```bash
sure-eval metric describe <task> --help
```

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
pip install "sure-evaluation[wetext]"       # 兼容空 extra；WeTextProcessing 已包含在基础依赖中
pip install "sure-evaluation[canonical]"    # canonical ASR CER/MER/WER 路由

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

## 📑 流水线路由目录

SURE-EVAL 把每个指标都暴露为声明式流水线。`pipeline_catalog.jsonl` 将每个支持的 `任务 + 语言 + 指标` 映射到所选节点和必填输入角色：

- [docs/pipeline_catalog.jsonl](./docs/pipeline_catalog.jsonl) — 每行一个 JSON 对象
- [docs/pipeline_catalog.md](./docs/pipeline_catalog.md) — schema 和使用说明

示例条目：

```jsonl
{"task":"ASR","language":"zh","metric":"cer","pipeline_id":"asr.zh.cer.wetext_zh_itn.wenet_cer","nodes":["normalization/wetext_norm","scoring/wenet_cer"],"required_roles":["hyp","ref"]}
{"task":"TTS","language":"zh","metric":"tts_cer","pipeline_id":"tts.zh.tts_cer.funasr_loader_16k_mono.paraformer_zh.punctuation_strip_norm.wenet_cer","nodes":["frontend/funasr_loader_16k_mono","transcription/paraformer_zh","normalization/punctuation_strip_norm","scoring/wenet_cer"],"required_roles":["prediction_audio","reference_text"]}
```

添加新路由后可重新生成：

```bash
python scripts/generate_pipeline_catalog.py
```

---

## 🐍 Python API

```python
from sure_eval.evaluation.scripts import describe_pipeline, run_task

# 查看路由
desc = describe_pipeline("asr", language="zh", metric="cer")
print(desc.node_ids)
# ('normalization/wetext_norm', 'scoring/wenet_cer')

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
