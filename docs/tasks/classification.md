# Classification / SER / GR

Generic classification routes align `key<TAB>label` files and compute accuracy.

## Metrics

| Task alias | Metric | Pipeline ID | Nodes |
|:-----------|:-------|:------------|:------|
| `classification` | `accuracy` | `classification.accuracy.classify` | `scoring/classify` |
| `ser` | `accuracy` | `ser.accuracy.classify` | `scoring/classify` |
| `gr` | `accuracy` | `gr.accuracy.classify` | `scoring/classify` |

SER and GR use built-in label specs so legacy artifacts keep the same accuracy behavior.

## Input Format

Tab-separated key-label files:

```text
utt_001<TAB>happy
utt_002<TAB>neutral
utt_003<TAB>angry
```

For SER, labels such as `happy` are mapped to canonical `hap`. For GR, `male`/`man` and `female`/`woman` are normalized.

## CLI Usage

### Generic classification

```bash
sure-eval metric describe classification --output /tmp/cls.json
sure-eval metric run --pipeline /tmp/cls.json \
  --ref-file ref.txt --hyp-file hyp.txt --output-dir /tmp/cls_eval
```

### SER / GR

```bash
sure-eval metric describe ser --output /tmp/ser.json
sure-eval metric run --pipeline /tmp/ser.json \
  --ref-file ref.txt --hyp-file hyp.txt --output-dir /tmp/ser_eval

sure-eval metric describe gr --output /tmp/gr.json
sure-eval metric run --pipeline /tmp/gr.json \
  --ref-file ref.txt --hyp-file hyp.txt --output-dir /tmp/gr_eval
```

## Python API

```python
from sure_eval.evaluation.scripts import run_task

report = run_task(
    "classification",
    ref_file="ref.txt",
    hyp_file="hyp.txt",
    output_dir="/tmp/cls_eval",
)
print(report.score)
```

## Output

- `report.json` — `score` (accuracy), correct/total counts.
- `pipeline_description.json` — selected route and node versions.

## Label Spec

For dataset-specific classification, pass a label spec file that declares canonical ids, aliases, and optional numeric ids. SER and GR specs are built into the scoring node.
