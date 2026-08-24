# Arabic NeMo Package Dependency Design

## Goal

Keep the existing Arabic NeMo TN + WeNet CER behavior while removing the
vendored NeMo source tree and duplicated Arabic grammar from this repository.

## Current Problem

`normalization/nemo_norm` currently injects a local `vendor/` directory into
`PYTHONPATH`. That directory contains 1,965 tracked files and about 104,000
lines. A second `ar/` directory duplicates the Arabic TN grammar. These files
make the change difficult to review and maintain even though the node manifest
already identifies the implementation as `nemo_text_processing==1.2.0`.

The official PyPI wheel for `nemo-text-processing==1.2.0` contains the complete
Arabic TN grammar, its TSV resources, and the public `Normalizer` entrypoint.
The Arabic grammar and `normalize.py` in that wheel were compared with the
current vendored copies and are identical apart from ignored cache files and
line endings.

## Architecture

The node-local project installs `nemo-text-processing==1.2.0` through `uv`.
The normalization wrapper imports
`nemo_text_processing.text_normalization.normalize.Normalizer` directly from
that environment. The root SURE environment remains free of Pynini and NeMo.

The runtime no longer appends `nemo_norm/vendor` to `PYTHONPATH`. The node-local
environment resolver and subprocess boundary remain unchanged.

## Repository Changes

- Add exact dependency `nemo-text-processing==1.2.0` to the node-local
  `pyproject.toml`.
- Declare the same package in `node_env.yaml` and verify both
  `nemo_text_processing` and `pynini` imports.
- Remove direct dependency declarations that are already provided transitively
  by the pinned NeMo package, except where the node directly imports them.
- Remove `nemo_norm/vendor/` and `nemo_norm/ar/` entirely.
- Remove the vendor-specific `extra_pythonpath` injection from `node.py`.
- Remove root wheel package-data patterns for the deleted source trees.
- Update the node README to document package installation rather than vendoring.

## Behavior And Identity

The public node ID, profile, and route identity do not change:

```text
normalization/nemo_norm (ar_tn)
asr.ar.cer.nemo_norm_ar_tn_v1.wenet_cer_v1
```

This is an implementation packaging change only. The node continues to convert
written Arabic forms to spoken forms and passes both reference and hypothesis to
WeNet CER. The pinned package version remains recorded in the node trace.

## Environment And Reproducibility

`sure-eval env setup --node normalization/nemo_norm` installs the pinned wheel
from PyPI or the configured company Python mirror. The generated `.venv`, cache,
and `uv.lock` remain local and are not committed. Exact NeMo source behavior is
fixed by `nemo-text-processing==1.2.0`; Pynini is fixed by that package to
`2.1.6.post1`.

If the package cannot be resolved, agent planning must report that node setup is
required or failed rather than silently falling back to repository source.

## Validation

- Recreate the node-local environment from the edited dependency declaration.
- Verify `nemo_text_processing==1.2.0` and `pynini==2.1.6.post1` are installed.
- Verify `21` normalizes to `واحد وعشرون`.
- Run the Arabic ASR route fixture and require CER `0.0`.
- Run the existing Arabic ASR, agent-plan, route identity, and script-contract
  test suites.
- Build the root wheel and verify it contains the wrapper and declarations but
  no `nemo_norm/vendor` or `nemo_norm/ar` paths.
- Confirm no virtual environment, lock, cache, or build output is tracked.

## Delivery

After local verification, commit the lightweight implementation and push the
same commit to GitHub branch `feat/arabic-nemo-itn` (PR #8) and GitLab branch
`multilingual-version`. Local, GitHub, GitLab, and PR head SHAs must match.
