# Environment Management

SURE-EVALUATION separates two environment levels:

- **Root environment:** CLI, routing, input contracts, reports, and lightweight
  scoring.
- **Node-local environment:** optional models, toolkits, learned metrics, or
  external binaries owned by one versioned node.

This separation keeps the base installation light and makes each pipeline's
runtime requirements explicit.

## Recommended Exact-Pipeline Flow

```bash
# 1. Discover exact alternatives.
sure-eval metric routes tts --language zh --metric cer --json

# 2. Describe the selected identity.
sure-eval metric describe tts \
  --pipeline-id tts.zh.cer.qwen3_asr_1_7b_v1.punctuation_strip_norm_v1.wenet_cer_v1 \
  --output .sure-eval-demo/tts-qwen.json

# 3. Review setup before making changes.
sure-eval env setup --pipeline .sure-eval-demo/tts-qwen.json --dry-run --json

# 4. Prepare and validate only selected optional nodes.
sure-eval env setup --pipeline .sure-eval-demo/tts-qwen.json
sure-eval env download --node transcription/qwen3_asr_1_7b --dry-run
sure-eval env download --node transcription/qwen3_asr_1_7b
sure-eval env check --pipeline .sure-eval-demo/tts-qwen.json --json
```

`env setup --pipeline` and `env check --pipeline` rebuild the registered route
and validate the pipeline identity before resolving node environments. They
deduplicate repeated nodes while preserving computation order.

## Other Selection Modes

Use lower-level selectors for maintenance or bulk preparation:

```bash
sure-eval env list --json
sure-eval env check --node scoring/dnsmos --json
sure-eval env setup --node scoring/dnsmos --dry-run
sure-eval env setup --task asr --language ar --metric cer --dry-run
sure-eval env setup --group tts-vc-mos --dry-run
sure-eval env check --all --json
```

`--node`, `--pipeline`, `--task`, `--group`, and `--all` are mutually
exclusive. Prefer `--pipeline` for an evaluation run because it preserves the
same exact selection across discovery, description, setup, checking, running,
and reporting.

## Assets And Locks

Inspect declared downloads without fetching them:

```bash
sure-eval env download --node scoring/dnsmos --dry-run --json
sure-eval env download --node transcription/qwen3_asr_1_7b --dry-run --json
```

A node may declare a Python version, runtime type, project file, frozen lock,
post-setup script, model provider, checkpoint target, and environment-variable
override in `node_env.yaml`. A frozen uv node installs from its committed
`uv.lock`. Source fetched during setup must use an immutable revision rather
than a moving branch.

Runtime assets remain local and ignored by Git:

- `.venv/` and node-local `**/.venv/`
- `**/checkpoints/`
- generated runtime metadata and logs
- model files such as `*.ckpt`, `*.pt`, `*.onnx`, `*.safetensors`, and `*.bin`

Do not add private absolute paths, credentials, or local checkpoints to route,
node, or documentation files.
