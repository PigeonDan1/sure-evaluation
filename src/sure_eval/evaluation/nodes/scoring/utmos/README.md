# UTMOS Scoring Node

`scoring/utmos` scores generated audio with a UTMOS provider and reports `utmos`.

The node wraps SURE's UTMOS metric normalization and aggregation. It does not load UTMOS directly;
model execution stays inside the injected provider.
