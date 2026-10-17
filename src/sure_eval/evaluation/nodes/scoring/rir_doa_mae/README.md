# RIR DOA MAE

## Purpose

Reports TDOA-derived binaural direction error.

## Contract and computation

Input is `multichannel_rir_samples_jsonl` with at least two channels and
`metadata.microphone_spacing_m`; output is a lower-is-better mean in degrees.
Version `v1` reports each extra-channel-versus-channel-0 pair and converts its
broadband ITD lag to an angle using the declared spacing and a 343 m/s default
sound speed.

## Runtime and limitations

The node is in-process with no checkpoint or node-local environment. It assumes
a far-field pair geometry and does not estimate a full spatial scene.

Source: SURE RIR task implementation in `tasks/rir/metrics.py`.
