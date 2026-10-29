# RIR Estimation Task

RIR estimation evaluates predicted room impulse responses against reference responses.
The required row roles are `prediction_rir`, `reference_rir`, and `sample_id`.
Rows are loaded from `rir_samples_jsonl`; paths are resolved relative to the
JSONL file and aggregated by arithmetic mean over samples. WAV/FLAC and NumPy
`.npy`/`.npz` RIR files are accepted; NumPy rows must include `sample_rate`.
Each pair is
truncated to its shorter file; samples must already share a time origin.

The task is language-independent. The single-channel branch validates exactly
one channel; the multi-channel branch validates two or more channels. IACC
requires exactly two channels. Its bundle reports full, early, and late
windows: full RIR, ch0 direct peak through 80 ms, and the remainder after
80 ms. Each window uses normalized cross-correlation in a +/-1 ms lag window.
ITD, ILD, IPD, and DOA use channel 0 as the reference and report every `chN`
pair. ILD and IPD are frequency-domain interaural errors.
The default bundle reports `rir_npm` as its top-level score and keeps the
individual metric scores in `details.results`.

Unscorable channels and undefined spatial correlations fail explicitly rather
than being omitted from the mean. Reports retain input paths, sample rate,
layout, geometry metadata, and original/paired lengths. See the
[task guide](../../../../../docs/tasks/rir.md) for the metric definitions and
the limits of the truncation and fixed +/-1 ms ITD/DOA search policies.
