# WavLM-Large SIM Scoring Node

`scoring/wavlm_large_sim` scores speaker similarity with a WavLM-large style provider and reports `sim/wavlm-large`.

The node wraps SURE's shared SIM normalization and aggregation. It does not load the WavLM model directly;
model loading stays inside the injected provider.
