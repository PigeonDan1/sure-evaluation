# Token-Level CER Scoring

Scores canonical written-form key-text files (see
`normalization/canonical_itn`) with token-level edit distance.

- Tokenization matches the canonical chain: CJK one token per character,
  latin letter runs one token per word, digits one token per character,
  surviving semantic symbols (% $ ¥ ° . -) one token each.
- Distance: rapidfuzz Levenshtein with unit costs; S/D/I decomposition from
  minimal edit operations. Corpus score is the micro-average
  `(sub + del + ins) / total_reference_tokens`.
- Coverage policy matches the other ASR scoring nodes: missing hypothesis
  utterances are scored as empty hypotheses (pure deletions) and reported
  via `num_hyp_missing_utts`; hypothesis-only utterances are counted in
  `num_hyp_extra_utts`; covering zero reference tokens raises instead of
  reporting a perfect 0.0.
- An empty reference with a non-empty hypothesis contributes pure
  insertions, so corpus scores may exceed 100% on hallucination-style sets
  by design.
