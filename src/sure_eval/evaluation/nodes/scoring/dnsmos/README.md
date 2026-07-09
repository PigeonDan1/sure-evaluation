# DNSMOS Scoring Node

`scoring/dnsmos` scores generated audio with a DNSMOS provider and reports `dnsmos`.

The node wraps SURE's DNSMOS metric normalization and aggregation. It does not load ONNX models directly;
DNSMOS execution stays inside the injected provider.
