# Recipe: Open ASR Leaderboard with Qwen3-ASR-1.7B

Reproduce an Open ASR Leaderboard-style English WER evaluation of
`Qwen/Qwen3-ASR-1.7B` on a local copy of the leaderboard datasets, through the
exact pipeline `asr.en.wer.openasr_norm_english_v1.wenet_wer_v1`.

The recipe covers seven datasets whose names match the leaderboard scoring
keys: `ami_cleaned_test`, `earnings22_cleaned_aa_test`,
`gigaspeech_cleaned_test`, `librispeech_test.clean`, `librispeech_test.other`,
`spgispeech_test`, `voxpopuli_cleaned_aa_test`.

## 1. Prepare the datasets

Point the script at a local copy of the leaderboard data (Hugging Face parquet
shards; Earnings22-Cleaned-AA is a JSONL + mp3 layout). Each dataset is
materialized as `<output-root>/<dataset>/v1.0.0/` with 16 kHz mono wavs, a
tab-separated `ref.txt`, and a `manifest.json` carrying source shard hashes:

```bash
python scripts/prepare_openasr_leaderboard.py \
    --source-root /path/to/open-asr-leaderboard \
    --output-root data/datasets/openasr_leaderboard

# smoke subset
python scripts/prepare_openasr_leaderboard.py \
    --source-root /path/to/open-asr-leaderboard \
    --datasets librispeech_test.clean --limit 20
```

Parquet reading needs `pyarrow` and audio decoding needs the `audio` extra
(`soundfile`, `librosa`); both are imported lazily by the script.

## 2. Prepare the Qwen3-ASR environment

Inference reuses the existing `transcription/qwen3_asr_1_7b` node
(`qwen-asr` runtime, transformers backend):

```bash
sure-eval env setup --node transcription/qwen3_asr_1_7b
sure-eval env download --node transcription/qwen3_asr_1_7b
sure-eval env check --node transcription/qwen3_asr_1_7b
```

A CUDA device is expected by default (`--device` overrides).

## 3. Run the evaluation

```bash
python scripts/run_openasr_leaderboard_eval.py \
    --data-root data/datasets/openasr_leaderboard

# smoke run
python scripts/run_openasr_leaderboard_eval.py \
    --data-root data/datasets/openasr_leaderboard --limit 20

# rescore existing hypotheses without inference
python scripts/run_openasr_leaderboard_eval.py \
    --data-root data/datasets/openasr_leaderboard --hyp-dir results/hyps
```

Each dataset run writes `hyp.txt` beside `ref.txt`, scores it through the
pipeline above, and the script writes `report.json` under
`results/openasr_leaderboard/<timestamp>/` with per-dataset WER, the
arithmetic mean WER (the leaderboard composite), and informational RTFx.

## Comparability boundary

- Text normalization is the vendored leaderboard normalizer
  (`normalization/openasr_norm`, pinned to upstream commit
  `8f37837f988e3d641e726b61087b9c761c060013`), including the official filter
  that drops references normalizing to empty or the ignore marker.
- The official pipeline computes WER with `kaldialign.batch_error_rate`
  (`merge_compounds=True`); this recipe scores with `scoring/wenet_wer`, so
  compound split/join mismatches may differ slightly from official numbers.
- RTFx depends on local hardware and is not comparable to leaderboard values.
- Inference runs inside the `transcription/qwen3_asr_1_7b` node-local uv
  environment with that node's declared generation defaults; long-form sets
  (Earnings22, 14-22 minutes per sample) inherit the node defaults, so check
  `hyp.txt` coverage (`num_hyp_missing_utts` in the report) before trusting
  those rows.
- The node batches inputs per subprocess (default batch size 32). On
  memory-constrained machines force sequential transcription with
  `SURE_EVAL_TRANSCRIPTION_BATCH_SIZE=1` (each chunk reloads the model, so
  expect slower runs).
