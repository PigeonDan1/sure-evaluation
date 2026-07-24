# Canonical Written-Form (ITN) Normalization

Canonicalizes ASR reference/hypothesis text into the written space before
token-level CER scoring. The core idea: inverse text normalization (spoken ->
written) is a many-to-one mapping, so every reading variant of the same
number collapses to one canonical string (2024 ≡ 二零二四 ≡ 两千零二十四),
while TN (written -> spoken) must pick a single reading and mis-scores the
others. This makes the metric insensitive to writing-form conventions without
forgiving real recognition errors.

Chain: NFKC + lowercase → mask numeral-bearing idioms / 百分之 / unit words →
cn2an ITN → span-wise second pass for leftover CJK numerals (万/亿 cut points
preferred; a dangling 点 is never swallowed) → 百分之X → X% → mixed
written-number expansion (exact Decimal) → punctuation replaced by spaces
(never deleted; % $ ¥ ° and digit-context . / - survive; letter-internal
apostrophes deleted).

Guarantees and caveats:

- The ITN engine (cn2an) is required. A missing engine raises instead of
  silently degrading — availability-dependent fallbacks would change the
  metric between environments. The engine version is recorded in every run's
  node trace; identical scores require an identical cn2an version.
- Per-string cn2an failures deterministically fall back to the unconverted
  string and are counted as `itn_fallback_rows` in the node trace.
- Known limitations (frozen; any change bumps the rules version): clock
  times (两点半 vs 2:30) and unit lexemes (千克 vs kg) are not unified;
  approximate numerals (七八个) may merge with exact ones (78个); cn2an
  context quirks can convert only one side near a real error.

Pair with `scoring/token_cer`, which applies the matching tokenization
(CJK per char / latin words / digits per char) and rapidfuzz edit distance.
Public selection uses the exact pipeline id
`asr.zh.cer.canonical_itn_zh_v1.token_cer_v1`; the reported metric remains
`cer`. Empty-text rows are preserved (scored as deletions), and files with no
parseable `<key>\t<text>` rows raise.
