# normalization/vad_timebase

Normalizes validated VAD rows on the reference seconds timebase.

This first task version supports only `profile: strict` with zero collar and
zero boundary exclusion. The validation contract already rejects invalid,
out-of-range, and overlapping intervals. This node keeps a stable order and
records scored-region summaries on the reference `duration` timebase.
