# Classification Scoring

`scoring/classify` computes deterministic classification accuracy from aligned
`key<TAB>label` files. Dataset or task-specific label options are supplied by a
label spec with canonical ids, aliases, and optional numeric ids.

This node does not parse prompts. Prompt or option normalization should happen
in `normalization/prompt_norm` before this scoring node runs.
