# NIST SCTK sclite Scoring

This node wraps the NIST SCTK `sclite` binary as an optional ASR scoring
backend. It does not install SCTK into the main Python environment and is not
used by default ASR routes.

The first supported path is:

```text
normalized key-text files -> TRN -> sclite -> parsed WER/CER report
```

## Binary Resolution

At execution time the node resolves `sclite` in this order:

1. Explicit `sclite_bin=` argument
2. `SURE_EVAL_SCLITE_BIN`
3. `PATH`
4. `SURE_EVAL_SCTK_ROOT/<pinned_commit>/bin/sclite`
5. `${SURE_EVAL_CACHE_DIR:-~/.cache/sure-eval}/tools/sctk/<pinned_commit>/bin/sclite`
6. This node's `.local/sctk/<pinned_commit>/bin/sclite`

If no executable is found, the runtime error includes every searched path.

## Build

Build SCTK explicitly:

```bash
bash src/sure_eval/evaluation/nodes/scoring/sctk_sclite/build_sctk.sh
```

The default install prefix is:

```text
${SURE_EVAL_CACHE_DIR:-~/.cache/sure-eval}/tools/sctk/9688a26882a688132a5e414cadcb4c19b6fffaba
```

You can override it:

```bash
bash src/sure_eval/evaluation/nodes/scoring/sctk_sclite/build_sctk.sh \
  --prefix /path/to/sctk \
  --commit 9688a26882a688132a5e414cadcb4c19b6fffaba
```

Then either export:

```bash
export SURE_EVAL_SCLITE_BIN=/path/to/sctk/bin/sclite
```

or install under the default cache path.

## ASR Use

The ASR task can select this scorer explicitly:

```python
from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

report = evaluate_asr_files(
    "ref.txt",
    "hyp.txt",
    language="en",
    metric="wer",
    scorer="sctk_sclite",
)
```

Default ASR scoring remains `scoring/wenet_wer` or `scoring/wenet_cer`.

## Docker

For Docker execution, either install SCTK inside the image and set:

```bash
SURE_EVAL_SCLITE_BIN=/usr/local/bin/sclite
```

or mount the host cache read-only and set:

```bash
-v ${SURE_EVAL_CACHE_DIR:-~/.cache/sure-eval}/tools/sctk:/opt/sure-eval/tools/sctk:ro
-e SURE_EVAL_SCLITE_BIN=/opt/sure-eval/tools/sctk/9688a26882a688132a5e414cadcb4c19b6fffaba/bin/sclite
```

## Notes

- Input files must be `key<TAB>text`.
- Reference and hypothesis key sets must match exactly.
- Normalization is upstream; this node does not normalize text.
- CER uses `sclite -c NOASCII` after upstream normalization.
- STM/CTM timed scoring is intentionally left for a later extension.
