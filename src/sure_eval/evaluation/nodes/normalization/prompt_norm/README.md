# Prompt Normalization

## Purpose

`normalization/prompt_norm` converts prompt-based model answers into canonical
choice labels before classification-style accuracy scoring. It exists so SLU,
classification, SER, and GR routes can compare labels rather than raw prompt
surface forms.

The node does not compute accuracy. `scoring/classify` performs the comparison.

## Task Scenarios

- SLU accuracy where predictions may be prompt answers such as `A`, `A. yes`,
  or free-form option text.
- Classification/SER/GR routes that need a label spec with aliases or numeric
  ids.

## Input

- Schema: `key_text_files_plus_prompt_jsonl`.
- Required text files: aligned `ref` and `hyp` key-answer files.
- Optional structured label source: prompt JSONL or label spec.
- Supports:
  - structured choices as lists
  - structured choices as maps
  - legacy prompt choice lines
  - arbitrary choice ids
  - variable choice counts

## Output

- Schema: `key_text_files`.
- Output rows are `<key><TAB><canonical_choice>`.
- Profiles:
  - `choice_id`: outputs canonical choice ids.
  - `choice_text`: outputs canonical choice text for legacy compatibility.

## Versioned Computation

- Node id: `normalization/prompt_norm`.
- Version: `v1`.
- Internal behavior includes prompt parsing, alias matching, numeric-id
  matching, and choice canonicalization.

## Runtime and Assets

- Runtime: `in_process`.
- No optional packages or model assets.

## Source and References

- Source: local SURE prompt/label normalization implementation.

## Limitations

- Ambiguous answers that match multiple choices should be resolved by a
  stricter label spec.
- This node cannot infer task semantics that are absent from the prompt or
  label spec.
