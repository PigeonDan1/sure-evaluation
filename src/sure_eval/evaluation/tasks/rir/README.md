# RIR Task

RIR evaluates predicted room impulse responses against reference responses.
The required row roles are `prediction_rir`, `reference_rir`, and `sample_id`.
Rows are loaded from `rir_samples_jsonl`; paths are resolved relative to the
JSONL file and aggregated by arithmetic mean over samples. Each pair is
truncated to its shorter file; samples must already share a time origin.

The task is language-independent. Binaural metrics use the first two channels
as left/right and require matching sample rates and channel counts.
The default bundle reports `rir_nmse` as its top-level score and keeps every
metric in `details.results`.
