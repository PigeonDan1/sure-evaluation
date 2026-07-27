# Classification Scoring

## Purpose

`scoring/classify` computes deterministic classification accuracy from
aligned key-label files. Dataset-specific label options are supplied by a label
spec with canonical ids, aliases, and optional numeric ids.

The node does not parse prompts. Prompt or option normalization should happen
in `normalization/prompt_norm` before this scoring node runs.

## Task Scenarios

- Classification accuracy.
- SER accuracy.
- GR accuracy.
- SLU accuracy after prompt answer normalization.

## Input

- Schema: `key_label_files_plus_label_spec`.
- Required roles:
  - `hyp`: predicted labels
  - `ref`: reference labels
  - `label_spec`: canonical labels, aliases, and optional numeric ids
- Row format: `<key><TAB><label>`.

## Output

- Schema: `accuracy_report`.
- Metric: `accuracy`.
- Aggregation: `correct / evaluated`.
- Higher scores are better.

## Versioned Computation

- Node id: `scoring/classify`.
- Version: `v1`.
- Supports arbitrary label ids, aliases, numeric ids, and dataset label specs.

## Runtime and Assets

- Runtime: `in_process`.
- No optional packages or model checkpoints.

## Source and References

- Source: local SURE deterministic label-scoring implementation.

## Limitations

- Unparseable or unmapped labels cannot be guessed by the scorer.
- If a task uses prompt text, keep normalization in `prompt_norm` so this node
  remains a pure scorer.
