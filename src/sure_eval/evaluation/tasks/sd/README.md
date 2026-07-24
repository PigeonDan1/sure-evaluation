# SD Evaluation

Speaker diarization uses the generic `scoring/meeteval` node and reports DER.

The default pipeline is `sd.any.der.meeteval_v1`; its route id is
`sd.der.meeteval` with `collar=0.25`. Reference and
hypothesis files are passed directly to `meeteval.io.load`; use a MeetEval
supported annotation format such as RTTM for DER.
