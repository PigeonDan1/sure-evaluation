# TTS And VC

TTS and VC use JSONL sample manifests because one sample may include text,
generated audio, reference audio, and source audio.

TTS example:

```json
{"sample_id":"tts_001","prediction_audio":"out.wav","reference_text":"你好世界","reference_audio":"speaker.wav","language":"zh"}
```

VC example:

```json
{"sample_id":"vc_001","converted_audio":"converted.wav","source_audio":"source.wav","reference_audio":"speaker.wav","reference_text":"你好世界","language":"zh"}
```

Prepare selected optional nodes:

```bash
sure-eval env setup --task tts --language zh --metrics tts_cer,dnsmos --dry-run
sure-eval env check --task tts --language zh --metrics tts_cer,dnsmos
```

Run:

```bash
sure-eval metric describe tts --language zh --metrics tts_cer,dnsmos --output /tmp/tts.json
sure-eval metric run --pipeline /tmp/tts.json \
  --samples-jsonl samples.jsonl --output-dir /tmp/tts_eval --validate-env
```

