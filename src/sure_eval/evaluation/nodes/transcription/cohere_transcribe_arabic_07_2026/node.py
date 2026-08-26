"""Cohere Transcribe Arabic node-local transcription wrapper."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Protocol

from sure_eval.compat.deepspeed_stub import install_deepspeed_stub
from sure_eval.evaluation.core.types import PipelineNodeResult

NODE_ID = "transcription/cohere_transcribe_arabic_07_2026"
NODE_VERSION = "v1"
MODEL_ID = "CohereLabs/cohere-transcribe-arabic-07-2026"
MODEL_REVISION = "c3e911b42149bf7a1e53d5cef9878aee87515a23"
CHECKPOINT_ENV = "COHERE_TRANSCRIBE_ARABIC_07_2026_CHECKPOINT"
RUNTIME_PACKAGE = "transformers"
RUNTIME_PACKAGE_VERSION = "5.4.0"
SAMPLE_RATE_HZ = 16000
DEFAULT_BATCH_SIZE = 8
NODE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DIR = NODE_DIR / "checkpoints" / "cohere-transcribe-arabic-07-2026"
INTERNAL_STAGES = (
    "runtime_managed_audio_frontend",
    "batching",
    "asr_inference",
    "text_extraction",
)


class TranscriptionRunner(Protocol):
    def transcribe(self, audio_path: str, *, language: str = "ar") -> str:
        """Transcribe one audio file."""


class CohereArabicTranscriber:
    """Load the local Cohere checkpoint and run its official transcription API."""

    model_id = MODEL_ID

    def __init__(
        self,
        *,
        device: str = "cuda",
        checkpoint_dir: str | Path | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else _checkpoint_dir()
        self.batch_size = batch_size
        self.resolved_model_path = str(self.checkpoint_dir)
        self.device_map: str | None = None
        self.dtype_name: str | None = None
        self._model: Any | None = None
        self._processor: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            if not (self.checkpoint_dir / "model.safetensors").is_file():
                raise RuntimeError(
                    f"{NODE_ID} checkpoint is missing at {self.checkpoint_dir}. "
                    f"Set {CHECKPOINT_ENV} or prepare the node asset."
                )
            install_deepspeed_stub()
            import torch
            from transformers import AutoProcessor, CohereAsrForConditionalGeneration

            requested_device = _normalize_device_map(self.device)
            use_cuda = requested_device.startswith("cuda:") and torch.cuda.is_available()
            self.device_map = requested_device if use_cuda else "cpu"
            dtype, self.dtype_name = _select_torch_dtype(torch, use_cuda=use_cuda)
            self._processor = AutoProcessor.from_pretrained(
                self.checkpoint_dir,
                trust_remote_code=True,
                local_files_only=True,
            )
            self._model = CohereAsrForConditionalGeneration.from_pretrained(
                self.checkpoint_dir,
                device_map=self.device_map,
                dtype=dtype,
                local_files_only=True,
            )
            self._model.eval()
        return self._model, self._processor

    def transcribe(self, audio_path: str, *, language: str = "ar") -> str:
        return self.transcribe_batch([audio_path], language=language)[0]

    def transcribe_batch(self, audio_paths: list[str], *, language: str = "ar") -> list[str]:
        if not audio_paths:
            return []
        normalized_language = _normalize_language(language)
        model, processor = self._load()
        transcripts: list[str] = []
        for audio_chunk in _chunks(audio_paths, self.batch_size):
            waveforms = [_load_audio_file(path) for path in audio_chunk]
            inputs = processor(
                waveforms,
                sampling_rate=SAMPLE_RATE_HZ,
                return_tensors="pt",
                language=normalized_language,
            )
            inputs.to(model.device, dtype=model.dtype)
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                num_beams=1,
            )
            transcripts.extend(
                str(text).strip() for text in processor.batch_decode(outputs, skip_special_tokens=True)
            )
        return transcripts


def transcribe_cohere_transcribe_arabic_07_2026(
    audio_path: str,
    *,
    language: str = "ar",
    runner: TranscriptionRunner | None = None,
    role: str = "prediction_audio",
) -> tuple[str, PipelineNodeResult]:
    """Transcribe Arabic audio and return a stable computation trace."""

    _normalize_language(language)
    selected_runner = runner or CohereArabicTranscriber()
    transcript = selected_runner.transcribe(audio_path, language=language)
    return transcript, _trace_result(
        audio_path=audio_path,
        language=language,
        role=role,
        transcript=transcript,
        runner=selected_runner,
    )


def _trace_result(
    *,
    audio_path: str,
    language: str,
    role: str,
    transcript: str,
    runner: Any,
) -> PipelineNodeResult:
    return PipelineNodeResult(
        stage="transcription",
        node_id=NODE_ID,
        version=NODE_VERSION,
        details={
            "audio_path": audio_path,
            "language": language,
            "role": role,
            "transcript": transcript,
            "model_id": getattr(runner, "model_id", MODEL_ID),
            "model_revision": MODEL_REVISION,
            "resolved_model_path": getattr(runner, "resolved_model_path", None),
            "runtime_package": RUNTIME_PACKAGE,
            "runtime_package_version": _installed_version(RUNTIME_PACKAGE) or RUNTIME_PACKAGE_VERSION,
            "backend": "transformers",
            "audio_input_mode": "path",
            "audio_frontend_policy": "runtime_managed",
            "resample_policy": "cohere_asr_runtime_managed",
            "runtime_normalized_sample_rate_hz": SAMPLE_RATE_HZ,
            "runtime_normalized_channels": 1,
            "runtime_audio_dtype": "float32",
            "max_audio_clip_s": 35,
            "punctuation": True,
            "batch_size": getattr(runner, "batch_size", None),
            "dtype": getattr(runner, "dtype_name", None),
            "device_map": getattr(runner, "device_map", None),
        },
        internal_stages=INTERNAL_STAGES,
    )


def _checkpoint_dir() -> Path:
    configured = os.environ.get(CHECKPOINT_ENV)
    path = Path(configured).expanduser() if configured else DEFAULT_CHECKPOINT_DIR
    return path.parent if path.name == "model.safetensors" else path


def _select_torch_dtype(torch_module: Any, *, use_cuda: bool) -> tuple[Any, str]:
    if not use_cuda:
        return torch_module.float32, "float32"
    if torch_module.cuda.is_bf16_supported():
        return torch_module.bfloat16, "bfloat16"
    return torch_module.float16, "float16"


def _normalize_device_map(device: str) -> str:
    normalized = str(device).strip().lower()
    if normalized in {"", "auto", "cuda"}:
        return "cuda:0"
    if normalized.isdigit():
        return f"cuda:{normalized}"
    return normalized


def _normalize_language(language: str) -> str:
    normalized = str(language).strip().lower().replace("_", "-")
    if normalized == "ara" or normalized.startswith("ar"):
        return "ar"
    raise ValueError(f"{NODE_ID} supports only Arabic, got language={language!r}")


def _load_audio_file(path: str) -> Any:
    from transformers.audio_utils import load_audio

    return load_audio(path, sampling_rate=SAMPLE_RATE_HZ)


def _chunks(values: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("batch_size must be positive")
    return [values[index : index + size] for index in range(0, len(values), size)]


def _installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _emit(
    *,
    audio_path: str,
    language: str,
    transcript: str,
    trace: PipelineNodeResult,
    json_output: bool,
    line_no: int | None = None,
) -> None:
    payload = {
        "node_id": NODE_ID,
        "version": NODE_VERSION,
        "audio_path": audio_path,
        "language": language,
        "transcript": transcript,
        "trace": trace.details,
        "internal_stages": list(trace.internal_stages),
    }
    if line_no is not None:
        payload["line_no"] = line_no
    if json_output:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    else:
        print(transcript)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe Arabic audio with Cohere Transcribe Arabic.")
    parser.add_argument("--audio-path")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--language", default="ar")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)
    if bool(args.audio_path) == bool(args.input_jsonl):
        parser.error("exactly one of --audio-path or --input-jsonl is required")

    runner = CohereArabicTranscriber(device=args.device, batch_size=args.batch_size)
    if args.input_jsonl:
        rows = [
            (line_no, json.loads(line))
            for line_no, line in enumerate(Path(args.input_jsonl).read_text(encoding="utf-8").splitlines(), start=1)
            if line.strip()
        ]
        languages = {_normalize_language(str(row.get("language") or args.language)) for _, row in rows}
        if len(languages) > 1:
            raise ValueError("batch input must use one language")
        language = next(iter(languages), "ar")
        audio_paths = [str(row["audio_path"]) for _, row in rows]
        if args.json_output:
            with redirect_stdout(sys.stderr):
                transcripts = runner.transcribe_batch(audio_paths, language=language)
        else:
            transcripts = runner.transcribe_batch(audio_paths, language=language)
        for (line_no, row), audio_path, transcript in zip(rows, audio_paths, transcripts, strict=True):
            role = str(row.get("role") or "prediction_audio")
            trace = _trace_result(
                audio_path=audio_path,
                language=language,
                role=role,
                transcript=transcript,
                runner=runner,
            )
            _emit(
                audio_path=audio_path,
                language=language,
                transcript=transcript,
                trace=trace,
                line_no=line_no,
                json_output=args.json_output,
            )
        return 0

    if args.json_output:
        with redirect_stdout(sys.stderr):
            transcript, trace = transcribe_cohere_transcribe_arabic_07_2026(
                args.audio_path,
                language=args.language,
                runner=runner,
            )
    else:
        transcript, trace = transcribe_cohere_transcribe_arabic_07_2026(
            args.audio_path,
            language=args.language,
            runner=runner,
        )
    _emit(
        audio_path=args.audio_path,
        language=args.language,
        transcript=transcript,
        trace=trace,
        json_output=args.json_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
