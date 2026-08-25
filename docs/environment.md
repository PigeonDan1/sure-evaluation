# Environment Management

SURE-EVAL uses two levels of environment:

- Root environment: CLI, routing, reports, lightweight scoring.
- Node-local environments: heavyweight ASR, MOS, speaker similarity, learned MT metrics, and external binaries.

Inspect environments:

```bash
sure-eval agent plan asr --language zh --metric cer --json
sure-eval env list --json
sure-eval env check --node scoring/dnsmos --json
sure-eval env check --task tts --language zh --metrics cer,dnsmos --json
sure-eval env check --node transcription/qwen3_asr_1_7b --json
sure-eval env check --node normalization/funasr_itn --json
```

Use `agent plan` first when an agent or harness needs a single readiness
payload. It resolves the selected route, checks only the selected nodes, and
returns setup hints for blocking requirements. Use `env check/setup/download`
when you need lower-level node inspection or preparation.

Prepare environments:

```bash
sure-eval env setup --task asr --language zh --metric cer --dry-run
sure-eval env setup --node scoring/dnsmos --dry-run
sure-eval env setup --node transcription/qwen3_asr_1_7b --dry-run
sure-eval env setup --node normalization/funasr_itn --dry-run
sure-eval env setup --group tts-vc-mos --dry-run
```

Download assets:

```bash
sure-eval env download --node scoring/dnsmos --dry-run
```

`--dry-run` is recommended first. It prints the provider, target path, and
environment-variable override for each declared asset.

Runtime assets remain local and ignored by git:

- `.venv/`
- `.venv.hostbak/`
- `**/.venv/`
- `**/checkpoints/`
- model files such as `*.ckpt`, `*.pt`, `*.onnx`, `*.safetensors`, `*.bin`

A uv node may declare `frozen: true` and a `post_setup_script` in
`node_env.yaml`. Setup then installs exactly from the committed `uv.lock` and
runs the packaged script with the node-local Python. For example,
`normalization/funasr_itn` uses this mechanism to fetch a commit-pinned source
subdirectory and records its Git tree in local runtime metadata.
