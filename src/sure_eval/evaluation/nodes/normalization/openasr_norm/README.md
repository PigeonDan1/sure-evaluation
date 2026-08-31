# Open ASR Leaderboard Normalizer

## Purpose

Normalizes ASR reference and hypothesis key-text files with the exact text
normalizer used by the Hugging Face Open ASR Leaderboard
(`huggingface/open_asr_leaderboard`), and replays the leaderboard's
scoring-side filter that drops rows whose normalized reference is empty or the
literal `ignore time segment in scoring` marker. It does not rescore or realign
audio and does not modify hypothesis-only rows beyond the same normalization.

## Task Scenarios

- `ASR/en/wer` alternative route `asr.en.wer.openasr_norm_english_v1.wenet_wer_v1`,
  used by the Open ASR Leaderboard recipe. The default English WER route remains
  `normalization/whisper_norm`.

## Input

- Schema: `key_text_files`.
- Roles: `ref_file`, `hyp_file`; rows are `key<TAB>text`.
- Alignment key: the leading key column; ref and hyp are matched by key.

## Output

- Schema: `key_text_files`; normalized `ref_file` and `hyp_file` with dropped
  reference rows removed from both sides.
- Trace details: row counts, empties after normalization, dropped ref keys,
  drop policy, and upstream identity (package, commit, vendored flag).

## Versioned Computation

- Node id `normalization/openasr_norm`, version `v1`.
- Internal stages: `key_text_parse`, `openasr_normalize`, `empty_ref_filter`.
- The vendored `EnglishTextNormalizer` extends the Whisper English normalizer
  with a larger filler-word ignore list, British-American spelling and name
  canonicalization maps, acronym collapsing, and compound-word mappings, all
  from `normalizer.py` + `english_abbreviations.py` at the pinned upstream
  commit.

## Runtime and Assets

- Runtime: `in_process`; runs in the base package (requires the core `regex`
  dependency only). No checkpoints, node-local environments, or binaries.

## Source and References

- Upstream: <https://github.com/huggingface/open_asr_leaderboard> directory
  `normalizer/`, commit `8f37837f988e3d641e726b61087b9c761c060013`,
  Apache-2.0 (see `normalization_impl/LICENSE.open_asr_leaderboard`).
- Official scoring filter: `is_target_text_in_range` in
  `normalizer/data_utils.py` at the same commit.

## Limitations

- English profile only.
- The official leaderboard computes WER with `kaldialign.batch_error_rate`
  using `merge_compounds=True` (split compounds such as "white paper" vs
  "whitepaper" score as zero error). This engine scores with
  `scoring/wenet_wer`, a plain word-level edit distance without compound
  merging, so scores can differ slightly from the official leaderboard numbers
  on hypothesis/reference compound mismatches.
- Hypothesis rows whose key has no reference are left to the scoring node's
  coverage accounting, same as the other normalization nodes.
