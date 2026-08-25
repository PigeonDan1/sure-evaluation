# NeMo Arabic Text Normalization

## Purpose

`normalization/nemo_norm` applies NVIDIA NeMo text normalization (TN) to Arabic
ASR reference and hypothesis transcripts. It converts written tokens such as
digits to spoken Arabic before scoring. It does not transcribe audio or calculate
CER.

## Task Scenarios

- Default route: `asr/ar/cer`.
- Exact pipeline: `asr.ar.cer.nemo_norm_ar_tn_v1.wenet_cer_v1`.
- Node chain: `normalization/nemo_norm` (`ar_tn`) -> `scoring/wenet_cer`.

The TN direction is intentional for this route: written-form reference tokens
are expanded to the spoken form commonly emitted by Arabic ASR systems, and both
reference and hypothesis pass through the same normalizer.

## Input

- Schema: `key_text_files`.
- Encoding: UTF-8.
- Each nonblank line must be `<key><TAB><text>`.
- Keys must be nonempty and unique within each file.
- Blank lines are ignored. Empty text after a valid tab is preserved.
- Malformed rows fail with the input path and line number; they are never
  silently dropped.

Both reference and hypothesis files must follow this contract. The downstream
CER scorer aligns rows by key and reports missing or extra utterances.

## Output

- Schema: `key_text_files`, preserving every input data-row key.
- Trace fields include language, profile, package and pinned package version,
  input/output schemas, and per-file row statistics.
- Internal stages are `key_text_parse`, `ar_tn`, and `key_text_write`.
- Temporary normalized files are deleted by the ASR task after scoring.

## Versioned Computation

- Node id: `normalization/nemo_norm`.
- Node version: `v1`.
- Profile: `ar_tn`.
- Direction: TN, written form to spoken form.
- Backend: `nemo_text_processing.text_normalization.normalize.Normalizer` with
  `input_case="cased"` and `lang="ar"`.
- Package: `nemo-text-processing==1.2.0`.

NeMo 1.2.0 declares `cdifflib` through an eagerly imported audio-alignment
helper. Arabic TN does not use that helper. The node excludes the C extension
from dependency resolution and supplies the compatible standard-library
`difflib.SequenceMatcher` API during import. This does not change the Arabic TN
grammar or normalization call.

## Runtime and Assets

- Runtime: optional node-local uv project, Python 3.11.
- Dependencies install with `uv sync --frozen` from the committed `uv.lock`.
- No model checkpoint, GPU, or API key is required.
- The base package can describe and plan the route, but scoring requires the
  node-local environment.
- The SURE wheel does not vendor NeMo source or grammar data; uv installs the
  pinned upstream package.

```bash
sure-eval agent plan asr --language ar --metric cer --json
sure-eval env setup --node normalization/nemo_norm --dry-run --json
sure-eval env setup --node normalization/nemo_norm
sure-eval env check --node normalization/nemo_norm --json
```

## Source and References

- [NVIDIA NeMo-text-processing repository](https://github.com/NVIDIA/NeMo-text-processing)
- [nemo-text-processing 1.2.0 on PyPI](https://pypi.org/project/nemo-text-processing/1.2.0/)
- Upstream license: Apache-2.0.

## Limitations

- Only Arabic TN profile `ar_tn` is exposed by this node version.
- ITN (spoken form to written form) is not exposed.
- Grammar coverage and spoken-form choices follow NeMo 1.2.0 and can differ
  from dataset-specific normalization conventions.
- Grammar construction adds startup cost; the lower-level node API accepts a
  NeMo cache directory when repeated execution needs persistent FST caching.
