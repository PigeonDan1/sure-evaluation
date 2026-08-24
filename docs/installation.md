# Installation

## Base Package

```bash
git clone https://github.com/PigeonDan1/sure-evaluation.git
cd sure-evaluation
pip install -e .
sure-eval doctor
printf "utt1\thello world\nutt2\tthis is a test\n" > /tmp/sure_ref.txt
printf "utt1\thello world\nutt2\tthis is test\n" > /tmp/sure_hyp.txt
sure-eval agent plan asr --language en --metric wer --json
sure-eval metric describe asr --language en --metric wer --output /tmp/asr.json
sure-eval metric run --pipeline /tmp/asr.json \
  --ref-file /tmp/sure_ref.txt --hyp-file /tmp/sure_hyp.txt --output-dir /tmp/asr_eval
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
The base smoke test above uses English ASR WER because it runs entirely
in-process and does not require a node-local uv environment.

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
sure-eval agent plan asr --language es --metric wer --json
sure-eval env setup --node normalization/funasr_itn --dry-run
```

Node environments are declared by `node_env.yaml` files under
`src/sure_eval/evaluation/nodes/**`.
Heavy transcription alternatives such as `transcription/qwen3_asr_1_7b`
declare their own node-local uv project and checkpoint target.
Multilingual ASR routes for `ja`, `ko`, `es`, `fr`, `de`, `ru`, `pt`, `vi`,
`id`, and `tl` use the optional `normalization/funasr_itn` node. Its setup uses
the committed dependency lock and fetches an immutable FunASR source revision.
`sure-eval doctor` checks the base installation by default; use
`sure-eval doctor --optional-nodes`, `sure-eval env check --all`, or a
task/pipeline-specific `sure-eval env check ...` command when you want optional
node-local environment diagnostics.
For agent-facing route and environment readiness, see
[`docs/agent_contract.md`](agent_contract.md).
