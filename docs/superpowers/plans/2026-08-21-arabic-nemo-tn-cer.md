# Arabic NeMo TN + CER Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the incorrect Arabic NeMo ITN + WER route with Arabic NeMo TN + CER.

**Architecture:** Keep `normalization/nemo_norm` as a node-local NeMo wrapper, but call the public Arabic `Normalizer` and identify the profile as `ar_tn`. Route its normalized key-text output into the existing in-process `scoring/wenet_cer` node.

**Tech Stack:** Python 3.11, NVIDIA NeMo text processing, Pynini, WeNet edit-distance scorer, pytest, Ruff, uv.

---

### Task 1: Lock the corrected route behavior

**Files:**
- Modify: `tests/test_arabic_nemo_asr_pipeline.py`

- [ ] **Step 1: Change the route regression to request Arabic CER**

Assert pipeline ID `asr.ar.cer.nemo_norm_ar_tn_v1.wenet_cer_v1` and nodes
`normalization/nemo_norm`, `scoring/wenet_cer`.

- [ ] **Step 2: Change the normalization regression to written-to-spoken**

Call `normalize_nemo_text("21")` and assert `"واحد وعشرون"`.

- [ ] **Step 3: Change end-to-end scoring to CER**

Evaluate `21 كتابا` against `واحد وعشرون كتابا` with `metric="cer"` and assert
zero score plus the corrected pipeline ID.

- [ ] **Step 4: Run the focused tests and verify RED**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_arabic_nemo_asr_pipeline.py`

Expected: failures show the current route is WER/ITN and normalization returns `21`.

### Task 2: Correct the NeMo normalization node

**Files:**
- Modify: `src/sure_eval/evaluation/nodes/normalization/nemo_norm/node.py`
- Modify: `src/sure_eval/evaluation/nodes/normalization/nemo_norm/manifest.yaml`
- Modify: `src/sure_eval/evaluation/nodes/normalization/nemo_norm/README.md`
- Replace: `src/sure_eval/evaluation/nodes/normalization/nemo_norm/ar/`

- [ ] **Step 1: Replace the node-local Arabic grammar copy**

Replace `nemo_norm/ar` with the exact contents of vendored
`nemo_text_processing/text_normalization/ar`.

- [ ] **Step 2: Switch the wrapper to NeMo Normalizer**

Import `Normalizer`, instantiate it with `input_case="cased", lang="ar"`, call
`normalize(text, verbose=False)`, rename profile and trace stage to `ar_tn`, and
report backend `NeMo Normalizer`.

- [ ] **Step 3: Correct node metadata and documentation**

Declare profile `ar_tn`, direction `tn`, and the `Normalizer` class. Describe
written-to-spoken Arabic normalization before CER.

- [ ] **Step 4: Run the text-normalization regression**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_arabic_nemo_asr_pipeline.py::test_asr_ar_nemo_normalizer_converts_written_number_to_spoken_form`

Expected: PASS.

### Task 3: Route Arabic ASR through CER

**Files:**
- Modify: `src/sure_eval/evaluation/tasks/asr/routes.yaml`
- Modify: `src/sure_eval/evaluation/tasks/asr/manifest.yaml`
- Modify: `src/sure_eval/evaluation/tasks/asr/pipeline.py`
- Modify: `src/sure_eval/evaluation/scripts/asr.py`
- Modify: `docs/tasks/asr.md`

- [ ] **Step 1: Replace Arabic route identity**

Register metric `cer`, profile `ar_tn`, scorer `scoring/wenet_cer`, and pipeline ID
`asr.ar.cer.nemo_norm_ar_tn_v1.wenet_cer_v1`.

- [ ] **Step 2: Change Arabic defaults and selectors**

Default Arabic metric to CER, resolve empty Arabic CER normalizer to `nemo:ar_tn`,
and map the route node to that selector.

- [ ] **Step 3: Update ASR documentation**

Describe Arabic TN + CER and remove the incorrect Arabic ITN + WER claim.

- [ ] **Step 4: Run the focused suite and verify GREEN**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_arabic_nemo_asr_pipeline.py`

Expected: all tests pass.

### Task 4: Verify repository contracts

**Files:**
- Verify: `tests/test_asr_pipeline_nodes.py`
- Verify: `tests/test_evaluation_scripts_contracts.py`
- Verify: `tests/test_agent_plan.py`
- Verify: `tests/test_pipeline_identity.py`
- Verify: `tests/test_pipeline_catalog_identity.py`

- [ ] **Step 1: Run targeted ASR and catalog tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_arabic_nemo_asr_pipeline.py tests/test_asr_pipeline_nodes.py tests/test_evaluation_scripts_contracts.py tests/test_agent_plan.py tests/test_pipeline_identity.py tests/test_pipeline_catalog_identity.py`

Expected: zero failures.

- [ ] **Step 2: Verify agent planning and runtime**

Run `sure-eval agent plan asr --language ar --metric cer --json`, then describe and
run the Arabic CER pipeline on a written/spoken-number fixture. Expected route is
TN + CER and score is zero.

- [ ] **Step 3: Run lint and wheel build**

Run targeted Ruff checks and `uv build --wheel`. Expected: both exit zero and the
wheel contains the Arabic normalization Python/data resources.

### Task 5: Commit and synchronize remotes

**Files:**
- Commit all corrected source, tests, docs, and design artifacts.

- [ ] **Step 1: Inspect the final diff and ignored files**

Confirm no `.venv`, cache, lock, or build directory is tracked.

- [ ] **Step 2: Commit the correction**

Run: `git commit -m "fix: use Arabic NeMo TN with CER"`.

- [ ] **Step 3: Push GitHub and GitLab branches**

Push HEAD to `fork/feat/arabic-nemo-itn` and `origin/multilingual-version`.

- [ ] **Step 4: Verify remote equality**

Use `git ls-remote` to confirm local, GitHub, and GitLab SHAs are identical, then
verify GitHub PR #8 points at that SHA.
