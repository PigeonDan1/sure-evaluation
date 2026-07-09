# G-STAR SA-ASR Normalization

This node applies the G-STAR-compatible text normalization rule to key-text
files.

The input contract is `key<TAB>text`. Task-specific formats such as STM are
converted before and after this node by modules under
`src/sure_eval/evaluation/conversion/`.

It matches the text behavior used by `SUREEvaluator._eval_sa_asr`:

- parse key-text rows;
- normalize only the text field;
- use `case_sensitive=False` and `remove_tag=True`;

The default SA-ASR route uses this node before `scoring/meeteval` so cpWER and
DER stay aligned with the old evaluator. The SA-ASR task layer converts STM to
key-text before this node and converts normalized key-text back to STM before
MeetEval scoring.
