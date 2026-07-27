# Installation

## Base Package

```bash
git clone https://github.com/PigeonDan1/sure-evaluation.git
cd sure-evaluation
pip install -e .
sure-eval doctor
sure-eval metric describe asr --language zh --metric cer --json
sure-eval agent plan asr --language zh --metric cer --json
```

The project is not currently published to PyPI. Install from a source checkout.

For local development, use the development extra from the repository root:

```bash
pip install -e ".[dev]"
```

The base package is intentionally lightweight. It must support route inspection,
normalization, reporting, and lightweight metrics without downloading model
weights or creating node-local environments.

Mandarin ASR CER selects `normalization/wetext_norm` (`zh_itn`) by default.
That node owns its pinned WeTextProcessing/Pynini environment under
`src/sure_eval/evaluation/nodes/normalization/wetext_norm/`.

Optional extras:

```bash
pip install -e ".[audio]"        # local audio helpers
pip install -e ".[download]"     # Hugging Face / ModelScope asset download helpers
pip install -e ".[diarization]"  # MeetEval for SD and SA-ASR
pip install -e ".[wetext]"       # compatibility no-op; wetext_norm uses node-local uv
pip install -e ".[canonical]"    # canonical ASR CER/MER/WER routes
```

For maintainers:

```bash
pip install -e ".[dev,audio,download,diarization,wetext,canonical]"
```

## Optional Cache Root

```bash
export SURE_EVAL_CACHE_DIR=/path/to/sure-eval-cache
```

If unset, SURE-EVAL uses `~/.cache/sure-eval`.

## Optional Node Environments

```bash
sure-eval agent plan asr --language zh --metric cer --json
sure-eval env setup --task asr --language zh --metric cer --dry-run
sure-eval agent plan tts --language zh --metrics cer,dnsmos --json
sure-eval env list
sure-eval env setup --task tts --language zh --metrics cer,dnsmos --dry-run
sure-eval env check --task tts --language zh --metrics cer,dnsmos
sure-eval agent plan tts \
  --pipeline-id tts.zh.cer.qwen3_asr_1_7b_v1.punctuation_strip_norm_v1.wenet_cer_v1 \
  --json
sure-eval env setup --node transcription/qwen3_asr_1_7b --dry-run
```

Node environments are declared by `node_env.yaml` files under
`src/sure_eval/evaluation/nodes/**`.
Heavy transcription alternatives such as `transcription/qwen3_asr_1_7b`
declare their own node-local uv project and checkpoint target.
For agent-facing route and environment readiness, see
[`docs/agent_contract.md`](agent_contract.md).
