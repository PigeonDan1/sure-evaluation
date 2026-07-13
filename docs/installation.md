# Installation

## Base Package

```bash
pip install sure-evaluation
sure-eval doctor
sure-eval metric describe asr --language zh --metric cer --json
```

For local development:

```bash
pip install -e ".[dev]"
```

The base package is intentionally lightweight. It must support route inspection,
normalization, reporting, and lightweight metrics without downloading model
weights or creating node-local environments.

Optional extras:

```bash
pip install "sure-evaluation[audio]"        # local audio helpers
pip install "sure-evaluation[download]"     # Hugging Face / ModelScope asset download helpers
pip install "sure-evaluation[diarization]"  # MeetEval for SD and SA-ASR
pip install "sure-evaluation[wetext]"       # WeTextProcessing normalization
pip install "sure-evaluation[canonical]"    # canonical ASR CER/MER/WER routes
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
sure-eval env list
sure-eval env setup --task tts --language zh --metrics tts_cer,dnsmos --dry-run
sure-eval env check --task tts --language zh --metrics tts_cer,dnsmos
```

Node environments are declared by `node_env.yaml` files under
`src/sure_eval/evaluation/nodes/**`.
