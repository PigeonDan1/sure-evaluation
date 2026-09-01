# SURE-EVAL Open-Source Readiness Plan

This plan turns the standalone `sure-evaluation` directory into a clean, reproducible, researcher-friendly open-source evaluation package.

The key rule is:

> Keep local environments and checkpoints on disk for development, but keep them out of git. Commit the declarations, setup commands, download logic, checks, and documentation needed to rebuild them.

## Execution Status

Status as of this implementation pass:

- P0 is implemented: `sure-eval` and `sure-evaluation` both point to the top-level CLI, README commands match the CLI, hardcoded cache paths moved behind `SURE_EVAL_CACHE_DIR`, local runtime assets are ignored, `config/default.yaml` and a minimal packaged SOTA baseline are present, and lightweight tests pass in the clean root environment.
- P1 is implemented for declaration, discovery, checking, dry-run setup, grouped setup, and task/metric-derived setup. Actual uv/binary setup execution is available, while automated checkpoint downloads remain a P2 item.
- P2 is partially implemented: `sure-eval env download --dry-run` reports declared model/tool/package assets, Hugging Face and ModelScope downloads have an explicit execution path, and SCTK setup uses the portable cache root. Checksum/license/citation completeness and provider-specific smoke checks remain.
- P3 is implemented for docs, contribution templates, CI workflow, package build rules, and local wheel/sdist build validation. Isolated wheel install smoke was attempted but the pip dependency download timed out on the network.

Verified commands:

```bash
.venv/bin/python -m pytest tests/test_evaluation_cli.py tests/test_evaluation_env_check.py tests/test_rps_manager.py tests/test_whisper_normalization_node.py tests/test_sctk_sclite_scoring_node.py -q
.venv/bin/sure-eval doctor --json
.venv/bin/sure-eval metric describe asr --language zh --metric cer --json
.venv/bin/sure-eval env list --json
.venv/bin/sure-eval env setup --task tts --language zh --metrics tts_cer,dnsmos --dry-run --json
.venv/bin/sure-eval env download --node scoring/dnsmos --dry-run --json
.venv/bin/python -m build --wheel
.venv/bin/python -m build --sdist
```

## Goals

1. Make the package easy to install and use for lightweight evaluation.
2. Make heavy speech/audio metrics optional, explicit, and reproducible.
3. Make contribution paths clear for new tasks, metrics, nodes, and model-backed evaluators.
4. Keep the git repository small, portable, and free of local HPC paths, private environments, and model weights.

## Non-Goals

1. Do not commit `.venv`, `.venv.hostbak`, node-local `.venv`, checkpoints, caches, or model artifacts.
2. Do not make every heavy node a mandatory dependency of the base package.
3. Do not require users to build all environments before running simple text metrics.
4. Do not encode local user paths such as `/hpc_stor03/sjtu_home/<user>` in source code or public docs.

## Current Constraints

- The source package is mostly complete: `evaluation`, `core`, `compat`, `reports`, task routes, nodes, scripts, and tests are present.
- Local runtime assets exist and can remain on this machine, but `.gitignore` must keep them untracked:
  - `.venv/`
  - `.venv.*/`
  - `.venv.hostbak/`
  - `**/.venv/`
  - `**/checkpoints/`
  - `runtime/`
  - large model artifacts such as `*.ckpt`, `*.pt`, `*.onnx`, `*.safetensors`, `*.bin`
- Local HPC cache defaults have been replaced with a portable cache API.
- README examples now match `sure-eval metric ...`, and `pyproject.toml` exposes both `sure-eval` and `sure-evaluation`.
- Heavy nodes now have a consistent declaration/check/dry-run setup story through `node_env.yaml`.

## Repository Boundary

### Keep In Git

- Source code under `src/`
- Tests and small fixtures
- `pyproject.toml`
- README and docs
- Node manifests and task routes
- Environment declarations such as `node_env.yaml`
- Lock files or reproducible requirement snapshots where practical
- Setup/check/download scripts
- Small metadata manifests for model sources, license, checksum, and expected file paths

### Keep Local But Ignored

- Root and node-local virtual environments
- Downloaded checkpoints and model caches
- Generated reports and runtime artifacts
- Local benchmark outputs
- Local migration manifests such as `MANIFEST.runtime.json`, unless explicitly useful as internal provenance

### Optional Large Assets

If a model or third-party asset is needed, publish how to obtain it:

- Provider: Hugging Face, ModelScope, official URL, or manual source
- Version or revision
- Expected target path under the node cache/checkpoint directory
- Checksum if available
- License and citation
- Environment variable override

## User Experience Target

### Base Install

The base install should be enough for route description, lightweight metrics, and docs examples:

```bash
pip install sure-evaluation
sure-eval doctor
sure-eval metric describe asr --language zh --metric cer
```

Base install must not create node-local environments or download model weights.

### Heavy Metric Setup

Users opt into heavy nodes:

```bash
sure-eval env list
sure-eval env setup --task tts --metrics tts_cer,sim/wavlm-large,dnsmos
sure-eval env check --task tts --metrics tts_cer,sim/wavlm-large,dnsmos
```

### Expert/Server Setup

Maintainers or benchmark servers can prepare everything:

```bash
sure-eval env setup --all
sure-eval env check --all
```

This path may take a long time and require GPU/CUDA/toolchain access.

## CLI Design

Expose one primary command and keep compatibility aliases:

```toml
[project.scripts]
sure-eval = "sure_eval.evaluation.cli:app"
sure-evaluation = "sure_eval.evaluation.cli:app"
```

Suggested command tree:

```text
sure-eval
  metric
    describe
    run
  env
    list
    check
    setup
    download
  doctor
  version
```

Required behavior:

- `sure-eval metric describe` never requires heavy node environments.
- `sure-eval metric run` only requires environments for nodes selected by that pipeline.
- `sure-eval metric run --validate-env` fails early with actionable setup instructions.
- `sure-eval env check` returns machine-readable JSON with `--json`.
- `sure-eval env setup` supports `--dry-run`.

## Environment Model

Use a two-level environment strategy.

### Root Environment

The root package contains only shared orchestration and lightweight dependencies:

- CLI
- route parsing
- report generation
- normalization utilities that are pure Python or light
- lightweight scoring wrappers
- environment diagnostics

Root dependencies should not include model-specific heavyweight stacks unless required for core functionality.

### Node-Local Environments

Each heavy node owns its own dependency environment. This avoids conflicts between incompatible Python, PyTorch, TensorFlow, COMET, Fairseq, FunASR, and ONNX stacks.

Candidate heavy node groups:

| Group | Nodes |
|:--|:--|
| `asr-transcription` | `transcription/paraformer_zh`, `transcription/whisper_large_v3` |
| `tts-vc-basic` | ASR transcription nodes used for TTS/VC WER/CER |
| `tts-vc-speaker` | `scoring/wavlm_large_sim`, `scoring/ecapa_tdnn_sim`, `scoring/eres2net_sim` |
| `tts-vc-mos` | `scoring/dnsmos`, `scoring/wv_mos`, `scoring/utmos` |
| `s2tt-heavy` | `scoring/bleurt_20`, `scoring/xcomet_xl` |
| `kws-extra` | `scoring/wekws_det` |
| `sd-sa-asr` | `scoring/meeteval`, `scoring/sctk_sclite` |
| `normalization-extra` | `normalization/wetext_norm` |

## Node Environment Declaration

Add a machine-readable file for each non-trivial node, for example `node_env.yaml`:

```yaml
id: scoring/dnsmos
runtime:
  type: uv
  python: "3.11"
  project: pyproject.toml
  optional: true
  gpu: false
models:
  - id: DNSMOS/model_v8.onnx
    provider: manual
    target: checkpoints/DNSMOS/model_v8.onnx
    env: DNSMOS_CHECKPOINT
verify:
  imports:
    - librosa
    - onnxruntime
  files:
    - checkpoints/DNSMOS/model_v8.onnx
smoke:
  command:
    - .venv/bin/python
    - -m
    - sure_eval.evaluation.nodes.scoring.dnsmos.node
```

For SCTK-style binary tools:

```yaml
id: scoring/sctk_sclite
runtime:
  type: binary
  build_script: build_sctk.sh
  binary: bin/sclite
  optional: true
env:
  SURE_EVAL_SCLITE_BIN: /path/to/sclite
verify:
  commands:
    - sclite -h
```

The setup/check implementation should consume this declaration rather than hardcoding all node logic in Python.

## Cache And Checkpoint Policy

Implement one shared cache helper:

```python
def get_cache_dir(*parts: str) -> Path:
    base = os.environ.get("SURE_EVAL_CACHE_DIR")
    root = Path(base).expanduser() if base else Path.home() / ".cache" / "sure-eval"
    path = root.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
```

Use it for:

- TTS/VC metric caches
- SCTK build/install root
- Hugging Face cache roots
- ModelScope cache roots
- node setup artifacts

User override examples:

```bash
export SURE_EVAL_CACHE_DIR=/data/cache/sure-eval
export WHISPER_LARGE_V3_CHECKPOINT=/data/models/whisper-large-v3/model.safetensors
export SURE_EVAL_SCLITE_BIN=/opt/sctk/bin/sclite
```

## Model Download Strategy

`sure-eval env setup` should support:

1. Create or update node-local `.venv`.
2. Download or validate checkpoints when metadata is available.
3. Skip model download with `--no-download`.
4. Use existing paths via environment variables.
5. Print manual instructions when license or provider restrictions prevent automated download.

Example:

```bash
sure-eval env setup --node transcription/whisper_large_v3
sure-eval env setup --node scoring/xcomet_xl --no-download
sure-eval env check --node scoring/xcomet_xl --json
```

Expected failure style:

```text
FAILED scoring/xcomet_xl
missing checkpoint:
  ~/.cache/sure-eval/nodes/scoring/xcomet_xl/checkpoints/xcomet_xl/modelscope/evalscope/XCOMET-XL/checkpoints/model.ckpt
fix:
  sure-eval env setup --node scoring/xcomet_xl --download-models
or:
  export XCOMET_XL_CHECKPOINT_PATH=/path/to/model.ckpt
```

## Implementation Phases

## P0: Open-Source Safety And Usability

Target: make the repo safe to initialize as git and install as a lightweight package.

Status: implemented.

Tasks:

1. Done: `.gitignore` excludes local environments, checkpoints, caches, model artifacts, reports, runtime outputs, and migration manifests while keeping `reports/sota/**`.
2. Done: local environments and checkpoints were not deleted.
3. Done: `sure-eval` is the primary CLI command and `sure-evaluation` is kept as an alias.
4. Done: CLI exposes a real `metric` command group.
5. Done: `doctor` checks Python version, package import, cache root, `uv`, and optional node status.
6. Done: hardcoded HPC cache paths were replaced by `SURE_EVAL_CACHE_DIR` and `Path.home() / ".cache" / "sure-eval"`.
7. Done: `config/default.yaml` and `reports/sota/sota_baseline.yaml` are present.
8. Done: README examples match the actual command line.
9. Done: README describes optional heavy metrics and node-local environments.
10. Done: lightweight tests pass in the clean root environment.

Acceptance:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
sure-eval --help
sure-eval metric describe asr --language zh --metric cer --json
sure-eval doctor --json
pytest tests/test_whisper_normalization_node.py tests/test_sctk_sclite_scoring_node.py -q
git status --ignored
```

## P1: Declarative Node Environment Management

Target: users can install/check only the environments they need.

Status: implemented for declarations, checks, dry-run setup, grouped selection, task/metric selection, and basic uv/binary setup execution. Automated checkpoint downloads remain in P2.

Tasks:

1. Done: `node_env.yaml` exists for ASR transcription, TTS/VC speaker, TTS/VC MOS, S2TT heavy, KWS, SD/SA-ASR, SCTK, SacreBLEU, and WeText nodes.
2. Done: node discovery scans `node_env.yaml` and keeps the old known-node list only as compatibility fallback.
3. Done:
   - `sure-eval env list`
   - `sure-eval env check --node`
   - `sure-eval env check --task --metrics`
   - `sure-eval env setup --node`
   - `sure-eval env setup --task --metrics`
4. Done: `--all`, `--group`, `--dry-run`, `--no-download`, and `--json` are supported.
5. Done: Python nodes use `uv` by default.
6. Done: setup reuses existing `.venv` unless `--force` is passed.
7. Done: setup logs are written under `${SURE_EVAL_CACHE_DIR:-~/.cache/sure-eval}/logs/env-setup`.
8. Partially done: tests cover cache root handling, optional doctor warnings, node_env metadata, task/metric dry-run setup, and group dry-run setup. Provider download mocking belongs to P2.

Acceptance:

```bash
sure-eval env list --json
sure-eval env check --node scoring/dnsmos --json
sure-eval env setup --node scoring/dnsmos --dry-run
sure-eval env setup --task tts --metrics tts_cer,dnsmos --dry-run
```

## P2: Checkpoint And External Tool Reproducibility

Target: heavy metrics have clear, repeatable model/tool setup.

Status: partially implemented. Dry-run asset reporting exists for every declared `node_env.yaml`; Hugging Face and ModelScope have explicit download execution hooks; manual/restricted assets report target/env instructions; SCTK uses `build_sctk.sh` under the portable cache root. Checksums, license/citation completeness, and cheap provider-specific smoke tests remain.

Tasks:

1. Partially done: checkpoint-backed nodes declare provider, id, expected target path, and env override in `node_env.yaml`; checksums/license/citation fields still need completion.
   - provider
   - model id or URL
   - revision
   - expected target path
   - checksum if available
   - license
   - citation
2. Partially done: `sure-eval env download` supports dry-run for all declared assets and execution hooks for:
   - Hugging Face
   - ModelScope
   Direct URL download remains to be added when a licensed source is declared.
3. Done for dry-run: manual/restricted assets print target and env override; richer human docs remain.
4. Done: SCTK setup uses `build_sctk.sh` and `${SURE_EVAL_CACHE_DIR:-~/.cache/sure-eval}/tools/sctk`.
5. Done: `sure-eval env download --node ...` exists as a separate path.
6. Partially done: tests validate dry-run metadata without downloads; provider-specific import/file smoke tests remain.

Acceptance:

```bash
sure-eval env setup --node transcription/whisper_large_v3 --dry-run
sure-eval env check --node transcription/whisper_large_v3 --json
sure-eval env setup --node scoring/sctk_sclite --dry-run
```

## P3: Documentation, CI, And Contribution Workflow

Target: external researchers can use and extend the package without reading internal code first.

Status: implemented for documentation skeleton, issue/PR templates, CI workflow, and package build validation. README can still be shortened before the first public release.

Tasks:

1. Done: docs were added:
   - `docs/installation.md`
   - `docs/environment.md`
   - `docs/tasks/asr.md`
   - `docs/tasks/tts_vc.md`
   - `docs/tasks/s2tt.md`
   - `docs/contributing.md`
   - `docs/add_a_metric.md`
   - `docs/reproducibility.md`
2. Partially done: README links to the docs and keeps quick examples, but can still be shortened before release:
   - install
   - one text metric example
   - one optional heavy metric example
   - supported tasks table
   - contribution link
3. Done: issue templates were added:
   - bug report
   - metric request
   - new node contribution
   - environment/setup problem
4. Done: PR template includes:
   - route added
   - manifest added
   - node env declared if needed
   - tests added
   - docs updated
   - no checkpoints or `.venv` committed
5. Done: CI workflow added for:
   - core tests on Python 3.10/3.11
   - package build
   - CLI smoke
   Pending for a later hardening pass:
   - lint/format enforcement
   - optional scheduled heavy-node CI on a prepared machine

Acceptance:

```bash
python -m build
pip install dist/*.whl
sure-eval metric describe asr --language zh --metric cer --json
pytest -q
```

Local build validation produced a 279 KB wheel and a 229 KB sdist, with no
`checkpoints/`, `.venv/`, `*.ckpt`, `*.pt`, `*.onnx`, `*.safetensors`, or
`*.bin` entries. The isolated `pip install dist/*.whl` smoke was attempted but
stopped on a PyPI read timeout while downloading dependencies.

## Testing Strategy

### Core Tests

Run on every PR. No model download. No GPU.

- route parsing
- CLI describe
- lightweight scoring
- normalization
- report generation
- environment check logic with mocked files

### Node Environment Tests

Run with mocks or small local fixtures.

- `node_env.yaml` schema validation
- setup command generation
- missing checkpoint diagnostics
- env var override handling

### Heavy Smoke Tests

Run manually or scheduled on a prepared runner.

- one short ASR transcription sample
- one TTS/VC WER/CER sample
- one speaker similarity sample
- one MOS sample
- one S2TT heavy metric sample if resources allow

## Contribution Rules

Any new node must include:

1. `manifest.yaml`
2. `node_env.yaml` if it has non-trivial dependencies, checkpoints, external binaries, or model downloads
3. README with installation, inputs, outputs, and citation
4. Unit tests
5. route test if exposed through a task
6. smoke check or mocked environment test

Any new checkpoint-backed metric must include:

1. model source
2. model revision
3. expected local path
4. license
5. citation
6. download or manual setup instructions
7. environment variable override

Any PR must not include:

1. `.venv`
2. checkpoints
3. generated reports
4. local cache paths
5. private absolute paths
6. API keys or credentials

## Recommended Immediate Next Steps

1. Implement P0 CLI alignment.
2. Replace hardcoded cache paths with a shared cache helper.
3. Add `sure-eval doctor`.
4. Add initial `sure-eval env list/check` without setup.
5. Add `node_env.yaml` for ASR/TTS/VC main-chain nodes first.
6. Add dry-run setup support.
7. Update README and docs to describe base install vs optional heavy metrics.
8. Initialize git after verifying ignored runtime assets:

```bash
git init
git status --ignored
git add .
git status --short
```

Before the first public push, inspect `git status --short` and confirm no `.venv`, `checkpoints`, `runtime`, or large model files are staged.
