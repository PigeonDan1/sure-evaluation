# FunASR Loader 16k Mono Frontend

## Purpose

`frontend/funasr_loader_16k_mono` is a pipeline-visible audio frontend
contract for Mandarin-family FunASR transcription routes. It records that the
selected ASR runtime is expected to decode audio, convert to mono, and resample
to 16 kHz before Paraformer inference.

This node does not materialize a new WAV file. The downstream Paraformer node
still receives the original audio path so FunASR can apply its own loader
behavior.

## Task Scenarios

- TTS Mandarin semantic CER before `transcription/paraformer_zh`.
- VC Mandarin semantic CER before `transcription/paraformer_zh`.
- Routes using exact pipeline ids that include
  `funasr_loader_16k_mono_v1.paraformer_zh_v1`.

## Input

- Schema: `audio_path`.
- Required field: `prediction_audio` for TTS or converted VC/TSE audio roles.
- Languages: `zh`, `cmn`, `yue`.

## Output

- Schema: `original_audio_path_for_funasr`.
- Trace records the frontend contract and internal stages. It does not write a
  normalized audio artifact.

## Versioned Computation

- Node id: `frontend/funasr_loader_16k_mono`.
- Version: `v1`.
- Internal stages: `audio_decode`, `channel_mean`, `resample_if_needed`.
- The computation is runtime-managed by FunASR; SURE-EVAL exposes this node so
  the pipeline id can name the frontend expectation.

## Runtime and Assets

- Runtime: `in_process` contract node.
- No root package dependency beyond the selected downstream transcription
  node.
- The actual FunASR model and runtime are owned by
  `transcription/paraformer_zh`.

## Source and References

- FunASR toolkit: https://github.com/modelscope/FunASR
- SeACoParaformer ModelScope model:
  https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch

## Limitations

- This is not a general-purpose audio conversion node.
- It should not be reused for ASR runtimes whose frontend is not FunASR
  compatible.
