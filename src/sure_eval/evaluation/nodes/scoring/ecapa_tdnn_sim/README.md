# ECAPA-TDNN SIM Scoring Node

`scoring/ecapa_tdnn_sim` scores speaker similarity with an ECAPA-TDNN style
provider. It reports the canonical metric `spk_sim`; `sim/ecapa-tdnn` is the
method selector recorded in `execution_metrics`.

The node wraps SURE's shared SIM normalization and aggregation. It does not load SpeechBrain directly;
model loading stays inside the injected provider.
