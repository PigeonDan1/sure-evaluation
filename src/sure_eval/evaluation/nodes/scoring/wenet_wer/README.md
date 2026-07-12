# WeNet WER/CER/MER Scoring

This node wraps `sure_eval.evaluation.nodes.scoring.wenet_wer.wenet_compute_cer.compute_wer`.

The wrapped script is treated as a composite scoring backend because it includes
tokenization, optional character splitting, case normalization, tag stripping,
and edit-distance counting. The pipeline report records these internal stages
instead of pretending the backend is only a single edit-distance function.

The vendored `compute_wer` script keeps upstream WeNet scoring semantics. The
only intentional deviation is performance-only: the DP matrix reset touches
just the submatrix a given utterance uses, instead of every previously grown
row. Upstream reset made every utterance after one long utterance pay
O(longest^2), which turned long-form corpora with one long hypothesis into
multi-minute runs; scores are unchanged and covered by an order-independence
regression test.

Utterance coverage is enforced in the wrapper layer:

- Reference utterances missing from the hypothesis file are scored as empty
  hypotheses (pure deletions) rather than silently skipped. The scoring result
  reports `num_ref_utts`, `num_hyp_utts`, `num_hyp_missing_utts`,
  `num_hyp_extra_utts`, and key samples for any mismatch.
- If scoring covers zero reference tokens (empty or malformed inputs, or fully
  disjoint keys), the node raises instead of reporting a perfect 0.0 error
  rate. The zh-only / en-only side scores of the code-switch MER route stay
  lenient because a monolingual subset legitimately covers zero tokens.
