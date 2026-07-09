# S2TT

S2TT routes score source, hypothesis, and reference text depending on the
metric.

```bash
sure-eval metric describe s2tt --language zh --metric bleu --output /tmp/s2tt.json
sure-eval metric run --pipeline /tmp/s2tt.json \
  --ref-file ref.txt --hyp-file hyp.txt --src-file src.txt --output-dir /tmp/s2tt_eval
```

Lightweight metrics:

- `bleu`
- `bleu_char`
- `chrf`

Optional learned metrics:

- `xcomet_xl`
- `bleurt_20`

Inspect their environments before running:

```bash
sure-eval env setup --task s2tt --language zh --metric xcomet_xl --dry-run
sure-eval env download --task s2tt --language zh --metric xcomet_xl --dry-run
```

