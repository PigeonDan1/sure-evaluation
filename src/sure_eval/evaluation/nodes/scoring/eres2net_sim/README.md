# ERes2Net SIM Scoring Node

`scoring/eres2net_sim` scores speaker similarity with an ERes2Net style
provider. It reports the canonical metric `spk_sim`; `sim/eres2net` is the
method selector recorded in `execution_metrics`.

The node wraps SURE's shared SIM normalization and aggregation. It does not load ModelScope directly;
model loading stays inside the injected provider.
