# SLU Task Route

SLU is a two-node pipeline:

1. `normalization/prompt_norm` converts prompt-based raw answers into canonical
   choice ids or choice text.
2. `scoring/classify` computes accuracy over the normalized labels.

The prompt normalization node supports arbitrary choice ids and arbitrary
choice counts. Legacy `A. option` prompt text remains supported as a fallback.
