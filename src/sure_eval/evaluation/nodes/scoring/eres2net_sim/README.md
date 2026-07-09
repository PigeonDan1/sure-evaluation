# ERes2Net SIM Scoring Node

`scoring/eres2net_sim` scores speaker similarity with an ERes2Net style provider and reports `sim/eres2net`.

The node wraps SURE's shared SIM normalization and aggregation. It does not load ModelScope directly;
model loading stays inside the injected provider.
