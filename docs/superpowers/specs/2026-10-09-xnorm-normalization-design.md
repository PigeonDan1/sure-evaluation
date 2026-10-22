# XNorm Normalization Node Design

## Goal

Integrate the existing multilingual XNorm implementation as a SURE
normalization node under `src/sure_eval/evaluation/nodes/normalization`, and
make it usable through the ASR pipeline without changing existing default
routes.

## Context and Constraints

SURE normalization nodes expose file adapters that accept `KeyTextFiles`,
return normalized `KeyTextFiles`, and emit a `PipelineNodeResult`. Node
metadata is declared in `manifest.yaml`; optional dependencies are isolated in
a node-local environment described by `node_env.yaml` and `pyproject.toml`.

XNorm exposes `compile_pipeline(language).normalize(text).text_norm` and
supports `ar`, `de`, `en`, `es`, `fr`, `he`, `hi`, `hu`, `hy`, `id`, `it`,
`ja`, `ko`, `mr`, `pt`, `ru`, `sv`, `th`, `vi`, and `zh`. Its default direct
normalization path requires only `opencc-python-reimplemented`; NeMo remains an
optional XNorm feature and is outside this node's required environment.

## Architecture

Add a `normalization/xnorm` node with three layers:

1. A small runtime adapter imports XNorm from the node-local package, validates
   the requested language against the supported profile map, compiles one
   pipeline per file pair, and normalizes each valid `key<TAB>text` row.
2. A public module exposes single-text and key-text-file functions matching the
   conventions of existing normalization nodes. It preserves keys, skips
   malformed rows consistently with the existing adapters, records row counts
   and empty outputs, and removes temporary outputs on failure.
3. A manifest and node-local project metadata describe the in-process node,
   pinned XNorm source/version, license, supported profiles, and the required
   `opencc-python-reimplemented` dependency. The XNorm source is packaged with
   this node so framework execution does not depend on a global installation.

The ASR pipeline accepts `xnorm`, `xnorm:<language>`, and the node selector
`normalization/xnorm` as explicit normalizers. The profile must match the ASR
language. Route identity includes the selected XNorm profile. Existing default
normalizer selection and existing route IDs remain unchanged.

## Data Flow

`evaluate_asr_files` resolves `xnorm:<language>` ->
`normalization/xnorm` -> `normalize_xnorm_key_text_files` -> the existing
WeNet/SCTK scorer. The node trace includes backend, profile, input/output
schemas, row statistics, and the temporary output paths used by cleanup.

## Error Handling

- Unsupported languages raise a value error listing supported XNorm profiles.
- An explicit profile that differs from the ASR route language raises a value
  error before scoring.
- Import or environment failures raise the node-local setup instruction.
- Malformed key-text rows are skipped using the established adapter behavior;
  valid rows remain aligned by key.
- Any partially written temporary output is deleted before the error escapes.

## Verification

Tests will cover:

- supported-language metadata and unsupported-language rejection;
- representative normalization for spaced and unspaced languages;
- key-text preservation, row statistics, trace metadata, and cleanup;
- ASR normalizer parsing, profile validation, pipeline identity, and route
  execution through the existing scorer;
- manifest loading and environment checking;
- packaging/import behavior from the node-local source layout.

The implementation will run the focused XNorm tests, the existing ASR and
environment contract tests, and the full test suite relevant to changed
modules.

