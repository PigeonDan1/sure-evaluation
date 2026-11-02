# ETC canonicalization

## Purpose

This optional node adapts the model canonicalizer released with
[ETC-Bench](https://github.com/zh-qian2004/etc-bench). It independently maps
reference and hypothesis transcripts to a deterministic shared written form so
speech-equivalent transcript variants can be compared fairly. It does not
replace the task scorer or use benchmark pairs as a lookup table.

## Task Scenarios

- `asr/<language>/cer` and `asr/<language>/wer` alternative routes for the 12
  ETC-Bench languages: `ar`, `bn`, `de`, `en`, `fa`, `fr`, `hi`, `id`, `ja`,
  `ru`, `vi`, and `zh`.
- This is an optional alternative backend; no existing default route changes.

## Input

- `key_text_files` containing reference and hypothesis UTF-8 key-tab-text files.
- Each valid row has an utterance key, a tab, and one transcript. Reference and
  hypothesis files are canonicalized independently.

## Output

- `key_text_files` with the original keys preserved and canonical text values.
- The node trace records the model, language, row counts, changed counts,
  empty-row counts, and per-row before/after values.

## Versioned Computation

- Node id: `normalization/etc_canonicalization`, version `v1`.
- Internal stages: key-text parsing, ETC model canonicalization, key-text
  writing.
- The adapter follows ETC-Bench's evaluation-side transcript canonicalization
  contract and uses the published ETC-Qwen3.5 model prompt with greedy
  decoding. A model is loaded once per reference or hypothesis file batch.

## Runtime and Assets

- Runtime type: optional node-local `uv` environment, Python 3.11.
- Prepare it after cloning:

  ```bash
  sure-eval env setup --node normalization/etc_canonicalization
  ```

- Set `SURE_ETC_CANONICALIZATION_MODEL` to a user-downloaded local checkpoint,
  or use the published `zhiheng-qian/etc-qwen3.5-0.8b` identifier when it is
  already present in the local Hugging Face cache.
- The runner uses `local_files_only=True`; it never downloads weights during
  evaluation. Model weights, caches, and virtual environments remain outside
  the repository.

## Source and References

- [ETC-Bench](https://github.com/zh-qian2004/etc-bench), upstream revision
  `c317d5de3f9462370057df42441a3b7f3ecee728`.
- Published model family: `zhiheng-qian/etc-qwen3.5-0.8b` and
  `zhiheng-qian/etc-qwen3.5-4b`.

## Limitations

- Real inference requires a user-prepared checkpoint and the optional node
  environment; the base SURE installation does not include either.
- The model may conservatively retain ambiguous homophones or abbreviations.
- Generation uses greedy decoding with `max_new_tokens=256`; output that
  reaches the budget without EOS is rejected rather than scored as complete.
- The learned canonicalizer may change fillers, repetitions, casing,
  punctuation, numbers, symbols, and other transcript surface content.
- The node currently exposes the ETC-Bench 12-language profile set.
