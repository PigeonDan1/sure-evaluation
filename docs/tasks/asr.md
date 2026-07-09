# ASR

ASR routes score key-text reference and hypothesis files.

```bash
sure-eval metric describe asr --language zh --metric cer --output /tmp/asr.json
sure-eval metric run --pipeline /tmp/asr.json \
  --ref-file ref.txt --hyp-file hyp.txt --output-dir /tmp/asr_eval
```

Input format:

```text
utt_id<TAB>text
```

Default routes:

- `zh` uses AISpeech normalization and CER.
- `en` uses Whisper-style normalization and WER.
- code-switching uses AISpeech normalization and MER.

