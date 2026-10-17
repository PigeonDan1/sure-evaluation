# RIR Task

RIR evaluates predicted room impulse responses against reference responses.
The required row roles are `prediction_rir`, `reference_rir`, and `sample_id`.
Rows are loaded from `rir_samples_jsonl`; paths are resolved relative to the
JSONL file and aggregated by arithmetic mean over samples. WAV/FLAC and NumPy
`.npy`/`.npz` RIR files are accepted; NumPy rows must include `sample_rate`.
Each pair is
truncated to its shorter file; samples must already share a time origin.

The task is language-independent. The single-channel branch validates exactly
one channel; the multi-channel branch validates two or more channels. Binaural
metrics require exactly two channels and use them as left/right. IACC and ITD
are measured from normalized cross-correlation in a +/-1 ms lag window; ILD
and IPD are frequency-domain interaural errors.
The default bundle reports `rir_npm` as its top-level score and keeps the
NPM and RT60 scores in `details.results`.
