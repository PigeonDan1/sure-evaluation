# XNorm Normalization Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a framework-compatible multilingual XNorm normalization node and expose it through explicit ASR normalizer selection.

**Architecture:** Package the existing XNorm source inside the normalization node's local runtime, adapt its compiled pipeline to SURE key-text files, and return standard trace metadata. Extend ASR selector parsing and pipeline identity for `xnorm:<language>` while preserving default routes.

**Tech Stack:** Python 3.10+, pytest, PyYAML manifests, setuptools node-local environment, XNorm 1.5.0, `opencc-python-reimplemented`.

---

### Task 1: Add failing node contract tests

**Files:**
- Create: `tests/test_xnorm_normalization_node.py`

- [ ] **Step 1: Write tests for profile coverage and text normalization**

```python
def test_xnorm_exposes_all_supported_profiles():
    from sure_eval.evaluation.nodes.normalization.xnorm import SUPPORTED_PROFILES

    assert set(SUPPORTED_PROFILES) == {
        "ar", "de", "en", "es", "fr", "he", "hi", "hu", "hy", "id",
        "it", "ja", "ko", "mr", "pt", "ru", "sv", "th", "vi", "zh",
    }


def test_xnorm_normalizes_representative_text():
    from sure_eval.evaluation.nodes.normalization.xnorm import normalize_xnorm_text

    assert normalize_xnorm_text("I have 123 books.", language="en") == "I have one hundred twenty-three books."
    assert normalize_xnorm_text("我有123本书。", language="zh") == "我有一百二十三本书"


def test_xnorm_rejects_unknown_language():
    from sure_eval.evaluation.nodes.normalization.xnorm import normalize_xnorm_text

    with pytest.raises(ValueError, match="Unsupported xnorm language"):
        normalize_xnorm_text("123", language="xx")
```

- [ ] **Step 2: Run the focused tests and confirm they fail because the node is absent**

Run: `pytest -q tests/test_xnorm_normalization_node.py`

Expected: FAIL with an import error for `sure_eval.evaluation.nodes.normalization.xnorm`.

### Task 2: Implement the node-local XNorm adapter

**Files:**
- Create: `src/sure_eval/evaluation/nodes/normalization/xnorm/__init__.py`
- Create: `src/sure_eval/evaluation/nodes/normalization/xnorm/node.py`
- Create: `src/sure_eval/evaluation/nodes/normalization/xnorm/manifest.yaml`
- Create: `src/sure_eval/evaluation/nodes/normalization/xnorm/node_env.yaml`
- Create: `src/sure_eval/evaluation/nodes/normalization/xnorm/pyproject.toml`
- Copy: `xnorm/text_normalization/**` to `src/sure_eval/evaluation/nodes/normalization/xnorm/runtime/xnorm/text_normalization/**`

- [ ] **Step 1: Add the public API and supported profile validation**

Implement `normalize_xnorm_text(text, language)` by compiling one XNorm
pipeline for the requested language and returning `result.text_norm.strip()`.
Expose `SUPPORTED_PROFILES` from the XNorm language registry and raise
`ValueError("Unsupported xnorm language: ...")` before importing optional NeMo
components.

- [ ] **Step 2: Add key-text file adaptation**

Implement `normalize_xnorm_key_text_files(files, language)` to preserve each
`key<TAB>text` pair, skip malformed lines, write temporary UTF-8 files, and
return `PipelineNodeResult(stage="normalization", node_id="normalization/xnorm", version="v1")` with language, profile, backend, row counts, empty counts, and row details. Delete both temporary files if either side fails.

- [ ] **Step 3: Add node metadata and dependency isolation**

Declare an in-process node with all 20 profiles, XNorm 1.5.0 Apache-2.0
source metadata, and a node-local Python project depending on
`opencc-python-reimplemented==0.1.7`. The node runtime must import the bundled
source without relying on the repository root or a global `xnorm` install.

- [ ] **Step 4: Run the focused tests and confirm they pass**

Run: `pytest -q tests/test_xnorm_normalization_node.py`

Expected: PASS.

### Task 3: Add file and environment contract coverage

**Files:**
- Modify: `tests/test_xnorm_normalization_node.py`
- Modify: `tests/test_evaluation_env_check.py`

- [ ] **Step 1: Test key preservation, diagnostics, and failure cleanup**

Create reference/hypothesis key-text fixtures under `tmp_path`, assert keys and
normalized values are preserved, assert row/empty counts and trace metadata,
and monkeypatch the normalizer to raise after creating output to verify output
cleanup.

- [ ] **Step 2: Test manifest and environment discovery**

Assert `NodeEnvChecker().check_node("normalization/xnorm")` reports
`runtime == "in_process"`, and assert the node manifest contains all 20
profiles, `input_schema == output_schema == "key_text_files"`, and the pinned
package metadata.

- [ ] **Step 3: Run contract tests**

Run: `pytest -q tests/test_xnorm_normalization_node.py tests/test_evaluation_env_check.py`

Expected: PASS.

### Task 4: Wire explicit XNorm selection into ASR

**Files:**
- Modify: `src/sure_eval/evaluation/tasks/asr/pipeline.py`
- Modify: `tests/test_xnorm_normalization_node.py`

- [ ] **Step 1: Add failing selector and route tests**

Assert `_normalize_normalizer(language="en", metric="wer", normalizer="xnorm:en") == "xnorm:en"`, reject `xnorm:zh` for English, create an ASR evaluation with `normalizer="xnorm:en"`, and assert the trace node, pipeline ID component, and score are correct.

- [ ] **Step 2: Implement selector and pipeline dispatch**

Import the XNorm adapter, validate profile membership and exact language match,
dispatch `xnorm:<profile>` in `_normalization_node`, label it
`xnorm_<profile>` in `_normalizer_component`, and preserve default selection
when `normalizer` is omitted.

- [ ] **Step 3: Run ASR integration tests**

Run: `pytest -q tests/test_xnorm_normalization_node.py tests/test_asr_pipeline_nodes.py tests/test_pipeline_identity.py`

Expected: PASS.

### Task 5: Verify packaging and full changed-surface behavior

**Files:**
- Modify: `pyproject.toml` only if package-data rules need the bundled runtime
- Modify: `src/sure_eval/evaluation/nodes/normalization/xnorm/README.md`

- [ ] **Step 1: Document setup and supported profiles**

Document explicit usage (`normalizer="xnorm:<lang>"`), all profiles, output
semantics, and the fact that the default ASR routes are unchanged.

- [ ] **Step 2: Validate static/package contracts**

Run: `python -m compileall -q src/sure_eval/evaluation/nodes/normalization/xnorm`

Run: `python -m build --wheel --no-isolation`

Expected: both commands exit 0 and the wheel contains the node manifest,
metadata, and bundled XNorm runtime.

- [ ] **Step 3: Run the complete relevant test suite**

Run: `pytest -q tests/test_xnorm_normalization_node.py tests/test_funasr_normalization_node.py tests/test_arabic_nemo_asr_pipeline.py tests/test_evaluation_architecture_contracts.py tests/test_pipeline_identity.py tests/test_evaluation_env_check.py`

Expected: PASS with zero failures.

