"""Nemotron-3.5-ASR-0.6B streaming transcription node wrapper.

Wraps the Hugging Face transformers ``Nemotron3_5Asr`` model and
``Nemotron3_5AsrProcessor`` for both offline and chunked streaming inference.
The node does not calculate WER or CER; scoring is performed by the task route
after normalization.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Protocol

from sure_eval.evaluation.core.types import PipelineNodeResult

NODE_ID = "transcription/nemotron_asr_streaming_06b"
NODE_VERSION = "v1"
MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
CHECKPOINT_ENV = "NEMOTRON_ASR_STREAMING_06B_CHECKPOINT"
SAMPLE_RATE_HZ = 16000
DEFAULT_CHUNK_S = 0.56
MAX_NEW_TOKENS = 256
STRIP_LANGUAGE_TAGS = True
NODE_DIR = Path(__file__).resolve().parent


class TranscriptionRunner(Protocol):
    def transcribe(self, audio_path: str, *, language: str = "auto") -> str:
        """Transcribe one audio file."""
        ...


def _resolve_checkpoint() -> str:
    explicit = os.environ.get(CHECKPOINT_ENV)
    if explicit:
        return explicit
    return MODEL_ID


def _load_audio_as_float32_vector(audio_path: str | Path) -> Any:
    """Load an audio file as a 1-D float32 numpy array at 16 kHz."""
    import numpy as np
    import soundfile as sf

    data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=-1)
    if sample_rate != SAMPLE_RATE_HZ:
        import librosa

        data = librosa.resample(data, orig_sr=sample_rate, target_sr=SAMPLE_RATE_HZ)
    return np.asarray(data, dtype="float32")


def _transcribe_batched(
    model: Any,
    processor: Any,
    audio_arrays: list[Any],
    language: str = "auto",
) -> list[str]:
    inputs = processor(
        audio_arrays,
        sampling_rate=SAMPLE_RATE_HZ,
        language=language,
        return_tensors="pt",
    )
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    outputs = model.generate(**inputs)
    if hasattr(outputs, "sequences"):
        outputs = outputs.sequences
    if hasattr(outputs, "shape") and outputs.ndim == 3:
        outputs = outputs.squeeze(0)
    texts = processor.batch_decode(outputs, skip_special_tokens=True)
    if isinstance(texts, str):
        texts = [texts]
    return [str(text).strip() for text in texts]


def transcribe_nemotron_asr_streaming_06b(
    audio_path: str,
    *,
    language: str = "auto",
    runner: TranscriptionRunner | None = None,
    role: str = "prediction_audio",
) -> tuple[str, PipelineNodeResult]:
    """Transcribe an audio file and return the transcript and a trace node."""

    if runner is not None:
        transcript = runner.transcribe(audio_path, language=language)
        return transcript, PipelineNodeResult(
            stage="transcription",
            node_id=NODE_ID,
            version=NODE_VERSION,
            details={
                "audio_path": audio_path,
                "language": language,
                "role": role,
                "transcript": transcript,
                "runner": type(runner).__name__,
            },
        )

    from transformers import AutoModelForRNNT, AutoProcessor

    checkpoint = _resolve_checkpoint()
    model = AutoModelForRNNT.from_pretrained(checkpoint, device_map="auto")
    processor = AutoProcessor.from_pretrained(checkpoint)
    model.eval()

    audio = _load_audio_as_float32_vector(audio_path)
    transcript = _transcribe_batched(model, processor, [audio], language=language)[0]
    return (
        transcript,
        PipelineNodeResult(
            stage="transcription",
            node_id=NODE_ID,
            version=NODE_VERSION,
            details={
                "audio_path": audio_path,
                "language": language,
                "role": role,
                "model_id": MODEL_ID,
                "checkpoint": checkpoint,
                "backend": "transformers",
                "streaming": False,
                "chunk_s": DEFAULT_CHUNK_S,
                "transcript": transcript,
            },
            internal_stages=("runtime_managed_audio_frontend", "asr_inference", "text_extraction"),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe audio with Nemotron-3.5-ASR-0.6B.")
    parser.add_argument("--audio-path")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    if bool(args.audio_path) == bool(args.input_jsonl):
        parser.error("exactly one of --audio-path or --input-jsonl is required")

    if args.checkpoint:
        os.environ[CHECKPOINT_ENV] = args.checkpoint

    if args.input_jsonl:
        import json

        from sure_eval.evaluation.nodes.transcription.common.providers import (
            NodeLocalTranscriber,
        )

        transcriber = NodeLocalTranscriber(
            node_id=NODE_ID,
            node_dir=NODE_DIR,
            device=args.device,
        )
        input_path = Path(args.input_jsonl)
        for line_no, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            audio_path = str(row["audio_path"])
            language = str(row.get("language") or args.language)
            role = str(row.get("role") or "prediction_audio")
            transcript, trace = transcribe_nemotron_asr_streaming_06b(
                audio_path,
                language=language,
                runner=transcriber,
                role=role,
            )
            payload = {
                "node_id": NODE_ID,
                "version": NODE_VERSION,
                "audio_path": audio_path,
                "language": language,
                "transcript": transcript,
                "trace": trace.details,
                "line_no": line_no,
            }
            if args.json_output:
                sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            else:
                print(transcript)
        return 0

    if args.json_output:
        import json
        from contextlib import redirect_stdout

        with redirect_stdout(sys.stderr):
            transcript, trace = transcribe_nemotron_asr_streaming_06b(
                args.audio_path,
                language=args.language,
            )
    else:
        transcript, trace = transcribe_nemotron_asr_streaming_06b(
            args.audio_path,
            language=args.language,
        )
    payload = {
        "node_id": NODE_ID,
        "version": NODE_VERSION,
        "audio_path": args.audio_path,
        "language": args.language,
        "transcript": transcript,
        "trace": trace.details,
    }
    if args.json_output:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    else:
        print(transcript)
    return 0


if __name__ == "__main__":
    sys.exit(main())
