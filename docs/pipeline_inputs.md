# Pipeline Input Formats

SURE-EVAL uses explicit, role-addressed inputs. The CLI and Python API require the same files; only the packaging differs by task.

---

## Key-Text Files

Used by: **ASR**, **S2TT**, **Classification / SER / GR**.

Format: one row per sample, tab-separated:

```text
<key>\t<text>
```

Example `ref.txt`:

```text
utt_001	你好世界
utt_002	今天天气不错
utt_003	请问您需要什么帮助
```

Example `hyp.txt`:

```text
utt_001	你好世界
utt_002	今天天气很好
utt_003	请问你需要什么帮助
```

Keys must match between `ref` and `hyp`. The order does not matter.

### S2TT source text

XCOMET-XL also needs a source text file:

```text
utt_001	你好世界
utt_002	今天天气不错
```

---

## Audio Samples JSONL

Used by: **TTS**, **VC**.

Each line is one JSON object. Paths are resolved relative to the JSONL file unless absolute.

### TTS sample schema

| Field | Required | Description |
|:------|:---------|:------------|
| `sample_id` | yes | Unique sample identifier |
| `prediction_audio` | yes | Path to synthesized audio |
| `language` | yes | `zh`, `en`, `cmn`, `yue`, ... |
| `reference_text` | for `tts_cer`/`tts_wer` | Ground-truth text |
| `reference_audio` | for speaker / MOS metrics | Reference speaker audio |
| `metadata` | no | Arbitrary extra fields (ignored by scoring) |

Example `tts_samples.jsonl` for a semantic-only run:

```jsonl
{"sample_id":"tts_001","prediction_audio":"out/zh_001.wav","reference_text":"你好世界","language":"zh"}
{"sample_id":"tts_002","prediction_audio":"out/zh_002.wav","reference_text":"今天天气不错","language":"zh"}
```

Example for semantic + speaker + MOS:

```jsonl
{"sample_id":"tts_001","prediction_audio":"out/zh_001.wav","reference_text":"你好世界","reference_audio":"ref/zh_001.wav","language":"zh","metadata":{"speaker":"spk_01"}}
{"sample_id":"tts_002","prediction_audio":"out/zh_002.wav","reference_text":"今天天气不错","reference_audio":"ref/zh_002.wav","language":"zh","metadata":{"speaker":"spk_02"}}
```

Run it:

```bash
sure-eval metric describe tts --language zh --metrics tts_cer,sim/wavlm-large,dnsmos --output /tmp/tts.json
sure-eval metric run --pipeline /tmp/tts.json \
  --samples-jsonl tts_samples.jsonl \
  --output-dir /tmp/tts_eval \
  --device cuda \
  --validate-env
```

### VC sample schema

| Field | Required | Description |
|:------|:---------|:------------|
| `sample_id` | yes | Unique sample identifier |
| `converted_audio` | yes | Path to converted audio |
| `language` | yes | `zh`, `en`, `cmn`, `yue`, ... |
| `reference_text` | alternative to `reference_audio` | Ground-truth text |
| `reference_audio` | for audio-reference semantic / speaker / MOS | Reference audio |
| `source_audio` | no | Original source audio (used by some metrics) |
| `metadata` | no | Arbitrary extra fields |

Example with text reference:

```jsonl
{"sample_id":"vc_001","converted_audio":"out/vc_001.wav","reference_text":"你好世界","language":"zh"}
{"sample_id":"vc_002","converted_audio":"out/vc_002.wav","reference_text":"今天天气不错","language":"zh"}
```

Example with audio reference:

```jsonl
{"sample_id":"vc_001","converted_audio":"out/vc_001.wav","reference_audio":"ref/spk_01.wav","source_audio":"src/vc_001.wav","language":"zh"}
{"sample_id":"vc_002","converted_audio":"out/vc_002.wav","reference_audio":"ref/spk_02.wav","source_audio":"src/vc_002.wav","language":"zh"}
```

When `reference_text` is absent, the pipeline transcribes `reference_audio` and uses that transcript as the reference.

---

## Diarization Annotations

Used by: **SD**, **SA-ASR**.

MeetEval supports RTTM, STM, CTM, and SegLST. SD typically uses RTTM:

```text
SPEAKER meeting_001 1 12.34 2.50 <NA> <NA> speaker_A <NA>
SPEAKER meeting_001 1 15.10 1.80 <NA> <NA> speaker_B <NA>
```

SA-ASR expects STM six-field rows:

```text
session_id channel speaker start end transcript
```

Example:

```text
meeting_001 1 spk_A 12.34 14.80 hello world
meeting_001 1 spk_B 15.10 16.90 hello back
```

---

## SLU Prompt JSONL

Used by: **SLU**.

One line per sample, keyed by `key`:

```jsonl
{"key":"utt_001","prompt":"A. hello world\nB. goodbye\nC. maybe"}
{"key":"utt_002","prompt":"A. yes\nB. no"}
```

The prompt normalization node extracts choice ids or text and matches them against the reference and hypothesis answers.

---

## KWS Input Modes

Used by: **KWS**.

Three modes are supported. See [docs/tasks/kws.md](./tasks/kws.md) for full details.

### SURE JSON mode

`ref.jsonl`:

```jsonl
{"key":"utt_001","keyword":"hello","detected":true}
{"key":"utt_002","keyword":"hello","detected":false}
```

`pred.jsonl`:

```jsonl
{"key":"utt_001","keyword":"hello","detected":true,"score":0.95}
{"key":"utt_002","keyword":"hello","detected":false,"score":0.12}
```

### WeKWS score CTC mode

```text
# labels.txt
utt_001 1
utt_002 0

# scores.txt
utt_001 0.95
utt_002 0.12
```

---

## Common Rules

1. **Keys must be unique** within a file.
2. **Paths in JSONL are relative to the JSONL file** unless absolute.
3. **One language per JSONL** for TTS/VC samples.
4. **Extra fields are allowed** in `metadata`; top-level unknown fields are generally ignored but may be validated.
