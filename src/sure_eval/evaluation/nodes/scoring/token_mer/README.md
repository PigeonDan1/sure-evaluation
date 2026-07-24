# Mixed / English Token Error Rate

Scores canonical written-form key-text files with mixed-token edit distance:
English at the word level, CJK at the character level, digits per character.
This is the right口径 for English-heavy and code-switch references, where
pure char-CER is unfair and can flip comparisons.

The node applies the **exact same scorer and tokenizer** as
`scoring/token_cer`: CJK at character level, latin spans at word level, digits
per character, and symbols per character. The route difference lives in the
normalization stage and in the public `pipeline_id`. For `en`/`cs`
canonical-ITN routes, `normalization/canonical_itn` additionally
whisper-normalizes latin spans before the shared token chain, using the
Whisper English normalizer already vendored under `normalization/whisper_norm`:

- contraction expansion (`it's been` → `it has been`, `haven't` → `have not`),
- spoken numbers to digits (`fifty percent` → `50%`, ITN direction),
- spoken-filler removal on both sides (`um uh hmm mm mhm mmm`),
- British→American spelling fold (`colour` ≡ `color`).

Spans without latin letters pass through the whisper stage unchanged, so the
degeneration guarantees hold by construction and are locked by tests:

- text with **no latin letters** scores identically for the canonical ASR MER
  and CER pipeline IDs (token-for-token);
- text with **no CJK** scores identically for the canonical ASR MER and WER
  pipeline IDs.

Scoring and coverage policy match `scoring/token_cer`: rapidfuzz minimal
edit operations, corpus micro-average `(sub+del+ins)/ref_tokens`, missing
hypotheses scored as deletions, zero covered reference tokens raise, and
the shared word-spacing repair cancels pure spacing artifacts (a word whose
letters exactly equal 2-4 consecutive words on the other side is split;
any letter difference stays fully scored; count reported as
`spacing_repairs`).

English apostrophe policy (chosen for maximal equivalence under the
inherent non-transitivity of contractions): unambiguous bare forms are
restored before Whisper expansion (`dont` ≡ `don't` ≡ `do not`), while `'s`
is collapsed instead of expanded (`it's` ≡ `its`, `john's` ≡ `johns`) since
it is three-ways ambiguous (possessive / is / has). The one forgone
equivalence — `it's` vs `it is` — is documented in tests as an intentional
trade-off.
