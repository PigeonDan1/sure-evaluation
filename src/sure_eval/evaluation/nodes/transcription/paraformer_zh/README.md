# Paraformer-ZH Transcription Node

## Purpose

`transcription/paraformer_zh` converts Mandarin-family speech into transcript
text before semantic error-rate scoring. It is an ASR frontend for TTS/VC/TSE
semantic metrics, not a scoring node.

The node does not calculate CER. CER is calculated by the task route after
transcription and normalization.

## Task Scenarios

- TTS Mandarin semantic CER default route.
- VC Mandarin semantic CER default route.
- Routes whose exact pipeline id includes
  `funasr_loader_16k_mono_v1.paraformer_zh_v1`.

## Input

- Schema: `audio_path`.
- Required role: generated/prediction audio path.
- Languages: `zh`, `cmn`, `yue`.

## Output

- Schema: `transcript_text`.
- Output is one transcript string per input audio sample.
- Trace records model runner, language, and transcription stage details.

## Versioned Computation

- Node id: `transcription/paraformer_zh`.
- Version: `v1`.
- Default runner:
  `sure_eval.evaluation.nodes.transcription.ParaformerZHTranscriber`.
- Internal stages:
  - `audio_decode`
  - `asr_inference`
  - `text_extraction`

## Runtime and Assets

- Runtime: optional node-local `uv` project.
- Python: `3.11`.
- GPU: optional.
- Required model env var: `PARAFORMER_ZH_CHECKPOINT`.
- Model id:
  `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`.
- Verify imports: `funasr`, `modelscope`.

Setup:

```bash
sure-eval env setup --node transcription/paraformer_zh
```

## Source and References

- FunASR toolkit: https://github.com/modelscope/FunASR
- ModelScope model:
  https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch

## Limitations

- The model checkpoint is a runtime asset and must not be committed.
- The node accepts an injected runner for tests and smoke checks; production
  scoring should use the prepared node-local runtime.
