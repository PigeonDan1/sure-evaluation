# WV-MOS Scoring Node

`scoring/wv_mos` scores generated audio with a Wav2Vec2MOS provider and
reports the canonical metric `wv_mos`; `wv-mos` is accepted as an execution
selector for compatibility with upstream naming.

The node wraps SURE's WV-MOS metric normalization and aggregation. It does not load Wav2Vec2MOS directly;
model execution stays inside the injected provider.
