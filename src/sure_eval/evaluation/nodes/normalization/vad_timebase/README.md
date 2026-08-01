# normalization/vad_timebase

Normalizes validated VAD rows on the reference seconds timebase.

This first task version supports only `profile: strict` with zero collar and
zero boundary exclusion. Reference and prediction speech segments are clipped to
`[0, duration]`, invalid intervals are dropped, and overlapping segments are
merged before scoring. Frame-score intervals are clipped and sorted but not
merged because each interval carries its own score.
