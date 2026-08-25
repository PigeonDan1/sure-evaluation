# FunASR Inverse Text Normalization

## Purpose

`normalization/funasr_itn` wraps FunASR `fun_text_processing` inverse text
normalization (ITN). It converts spoken-form transcripts to written form before
ASR error-rate scoring. It does not transcribe audio or calculate CER/WER.

## Task Scenarios

The node is the default ASR normalizer for:

| Metric | Languages | Pipeline pattern |
|:-------|:----------|:-----------------|
| `cer` | `ja`, `ko` | `asr.<lang>.cer.funasr_itn_<lang>_v1.wenet_cer_v1` |
| `wer` | `es`, `fr`, `de`, `ru`, `pt`, `vi`, `id`, `tl` | `asr.<lang>.wer.funasr_itn_<lang>_v1.wenet_wer_v1` |

The node also exposes `zh` and `en` profiles for explicit lower-level use;
their default ASR routes continue to use WeText and Whisper normalization.

## Input

- Schema: `key_text_files`.
- Encoding: UTF-8.
- Each nonblank line must be `<key><TAB><text>`.
- Keys must be nonempty and unique within a file.
- Blank lines are ignored. Empty text after a valid tab is preserved.
- Malformed rows fail with the input path and line number; they are never
  silently dropped.
- The selected profile must equal the ASR route language.

Supported profiles are `zh`, `en`, `ja`, `es`, `fr`, `de`, `ko`, `ru`, `pt`,
`vi`, `id`, and `tl`.

## Output

- Schema: `key_text_files` with the same data-row keys as the input.
- Trace fields include profile, language, direction, source revision and tree,
  dependency versions, row counts, blank-line counts, and empty-text counts.
- Temporary normalized files are runtime artifacts and are not embedded as
  full rows in the report.

## Versioned Computation

- Node id: `normalization/funasr_itn`.
- Node version: `v1`.
- Direction: ITN, spoken form to written form.
- Upstream class: `fun_text_processing...InverseNormalizer(lang=<profile>)`.
- Internal stages: key-text parsing, Pynini WFST ITN, key-text writing.
- FST caches are separated by FunASR revision, Pynini version, and profile.

The upstream source is locked in `source_lock.json` to FunASR commit
`3c58cb56a56598232c3efffa15d313d7e82a4307` and `fun_text_processing` tree
`8dea23a54787d1cdd145425c212774a16e825f87`. Setup verifies both identities and
records them in local runtime metadata.

## Runtime and Assets

- Runtime: optional node-local uv project, Python 3.11.
- Dependencies are installed with `uv sync --frozen` from the committed
  `uv.lock`.
- `prepare_funasr_itn.py` fetches only the locked `fun_text_processing`
  subdirectory plus the upstream license.
- No model checkpoint, GPU, or API key is required.
- Network access to GitHub is required for first setup.

```bash
sure-eval env setup --node normalization/funasr_itn --dry-run
sure-eval env setup --node normalization/funasr_itn
sure-eval env check --node normalization/funasr_itn --json
```

The node-local `.venv`, fetched runtime source, and compiled FST cache are
runtime state and must not be committed. Override the shared cache root with
`SURE_EVAL_CACHE_DIR` when needed.

## Source and References

- [FunASR repository](https://github.com/modelscope/FunASR)
- [Locked FunASR revision](https://github.com/modelscope/FunASR/commit/3c58cb56a56598232c3efffa15d313d7e82a4307)
- [Pynini](https://www.opengrm.org/twiki/bin/view/GRM/Pynini)
- Upstream license: Apache-2.0, copied into the local runtime during setup.

## Limitations

- This node supports ITN only; written-to-spoken TN is not exposed.
- The base package can describe and plan these routes, but scoring requires the
  optional node setup first.
- Grammar coverage and output style are defined by the locked upstream
  implementation and vary by language.
- First use of a profile compiles or loads its FST cache and is slower than
  subsequent runs.
