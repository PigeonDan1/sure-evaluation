# Prompt Normalization

`normalization/prompt_norm` converts prompt-based classification answers into
canonical choices before scoring. It supports structured `choices`, map-style
choices, and legacy prompt text such as `A. option`.

The node does not compute accuracy. It only rewrites aligned `key<TAB>answer`
files so `scoring/classify` can compare normalized labels.
