# GigaSpeechBench Normalization

## Purpose

`normalization/giga_norm` reproduces the text normalization used by the
GigaSpeechBench multilingual speech-to-text benchmark. It vendors the upstream
`text_norm` package so SURE-EVAL can recompute the WER/CER numbers reported on
the GigaSpeechBench leaderboard through a declared pipeline route.

The node covers 17 language profiles: Mandarin, English, Japanese, Korean,
Thai, Indonesian, Malay, Filipino, Vietnamese, and seven Arabic locales.
Chinese dialects, English accents, vertical-domain subsets, and age-group
subsets resolve to the `CHN` or `USA` profile through the upstream alias table.

The node normalizes text only. It does not compute WER, CER, or MER, and it
does not perform hotword-weighted (B-WER/B-CER) scoring.

## Task Scenarios

- ASR Mandarin CER alternative route:
  `asr.zh.cer.giga_norm_chn_v1.wenet_cer_v1`.
- ASR English WER alternative route:
  `asr.en.wer.giga_norm_usa_v1.wenet_wer_v1`.
- ASR default route for language profiles that had no prior SURE-EVAL route:
  `th`, `ms`, `fil`, `ar-ae`, `ar-dz`, `ar-eg`, `ar-iq`, `ar-ma`, `ar-sa`,
  `ar-sy`. The `ar` profile is an explicit WER route because Arabic already
  has a separate default CER route.
- ASR alternative route for existing FunASR profiles: `ja`, `ko`, `id`, and
  `vi`.
- For `zh` and `en` this is an alternative route. The existing
  `wetext_norm` (zh) and `whisper_norm` (en) defaults are unchanged.

## Input

- Schema: `key_text_files`.
- Row format: `<key><TAB><text>`.
- Required roles: `ref`, `hyp`.
- Alignment key: `key`.
- Rows without a tab separator are skipped, matching the other normalization
  nodes in this stage.

## Output

- Schema: `key_text_files`.
- Output files preserve keys and carry normalized text for the scoring node.
- Trace fields: `profile`, `requested_profile`, `num_rows`,
  `num_empty_after_normalization`, `ref_rows`, `hyp_rows`, and a
  `normalization` block with upstream provenance and installed package
  versions.
- Not a scoring node; it reports no score direction.

## Versioned Computation

- Node id: `normalization/giga_norm`.
- Version: `v1`.
- Internal stages:
  - `key_text_parse`
  - `giga_<profile>_normalize`
- Profile selection: an explicit `profile` argument wins; otherwise the ASR
  route language is mapped through `LANGUAGE_PROFILES`. Alias resolution
  follows the upstream order: alias table, exact code, hyphen prefix,
  underscore prefix.
- Normalization rules per profile, unmodified from upstream:
  - `CHN` / `USA` share one implementation. Repeated n-grams are truncated
    first (hallucination guard), then the text is tokenized on whitespace and
    each token is language-detected into `en`, `chn_en`, or `not_chn_en`.
    English spans run the Whisper English normalizer and score formatting.
    Mixed and non-Latin spans additionally run TN (`normalize_nsw`), ITN
    (`all_convert`), and Simplified Chinese conversion (`zhconv`). Chinese
    output is tokenized per character.
  - Other profiles apply paralinguistic tag removal, script-specific
    punctuation and diacritic rules, and number-to-word expansion via
    `num2words`.
- Scores depend on the installed `whisper_normalizer`, `zhconv`, `num2words`,
  and `underthesea` versions. Each run records them in
  `details.normalization.package_versions`.
- Any change to the vendored rules must bump the node version, because WER and
  CER are sensitive to normalization.
- Reproduction check against the upstream per-language summaries in
  `data/results_CH-EN-Dialects/<lang>/<model>/`, using Azure ASR outputs:
  - `SGP-EN` WER over 9327 matched segments: GigaSpeechBench publishes 14.51,
    this node with `scoring/wenet_wer` reports 14.51.
  - `YUE` CER over 8807 matched segments: GigaSpeechBench publishes 11.79,
    this node with `scoring/wenet_cer` reports 11.86.
  - Normalized text was byte-identical to the upstream normalized corpus for
    3000 of 3000 sampled segments in each language.
  - The Cantonese gap comes from the scorer, not this node. GigaSpeechBench
    splits every character including Latin ones, while `scoring/wenet_cer`
    keeps an embedded Latin word as one token. Re-scoring the identical
    normalized text under each tokenization gives 11.7861 and 11.8575, which
    accounts for the whole difference, and affects only the 5.64% of Cantonese
    segments that contain code-switched English.

## Runtime and Assets

- Runtime type: `pip`, optional.
- Install with `pip install -e ".[giga]"`.
- Packages: `whisper_normalizer`, `zhconv`, `num2words`, and `underthesea`
  (`underthesea` is imported only by the `VNM` profile).
- Data assets: `normalization_impl/ref_code/{currency,measurements,timezones}.tsv`,
  used by the `IDN` profile.
- No model checkpoints, binaries, or environment variables.
- The node cannot run in the base package; missing dependencies raise
  `ImportError` instead of silently degrading.

## Source and References

- Upstream repository: GigaSpeechBench,
  https://github.com/SpeechColab/GigaSpeechBench (vendored from `text_norm/`
  at commit `ca782bf`).
- Paper: *GigaSpeechBench: A Real-World Multilingual Speech-to-Text Benchmark*,
  https://arxiv.org/abs/2606.28884
- Dataset: https://huggingface.co/datasets/speechcolab/GigaSpeechBench
- The `CHN` and `USA` profiles depend on the OpenAI Whisper text normalizers
  through the `whisper_normalizer` package,
  https://github.com/openai/whisper/tree/main/whisper/normalizers
- Upstream license: MIT. Vendored license, including its third-party notices:
  `normalization_impl/LICENSE.gigaspeechbench`.
- The `CHN` and `USA` profiles are derivative works of
  `speechio/chinese_text_normalization` (`cn_tn.py`, MIT), which credits
  `atomicoo/chn_text_norm` for its NSW normalizers. The Chinese punctuation
  constants come from the Zhon project (MIT), https://github.com/tsroten/zhon
- `DZA` follows the evaluation script of the Open Universal Arabic ASR
  Leaderboard,
  https://github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard
- Upstream copyright notices are preserved in the vendored file headers and in
  `normalization_impl/LICENSE.gigaspeechbench`.

## Limitations

- Two deliberate deviations from upstream make the vendored code library-safe:
  missing dependencies raise `ImportError` instead of calling `sys.exit(1)`,
  and unknown profiles raise `ValueError` instead of silently falling back to
  tag stripping. Normalization rules are unchanged.
- Scores are comparable with published GigaSpeechBench numbers for
  whitespace-delimited languages. For CJK text containing code-switched Latin
  words, the WeNet scorer's tokenization introduces a small offset as measured
  above; treat those numbers as route-specific.
- GigaSpeechBench evaluates hotword-weighted B-WER/B-CER for its
  vertical-domain subsets. That aggregation is a scoring concern and is not
  provided by this node.
- The `CHN` and `USA` profiles emit character-tokenized Chinese, so their
  output is intended for the WeNet CER/WER scorers and is not comparable with
  `canonical_itn` token streams.
- Arabic locale profiles apply dialect-specific rules and are not
  interchangeable; selecting the wrong locale changes the score.
- Upstream applies a duration filter (segments longer than 0.5s) before
  producing its leaderboard numbers. That filtering happens outside this node
  and must be reproduced by the caller to match the leaderboard rather than the
  unfiltered per-language summaries used above.
