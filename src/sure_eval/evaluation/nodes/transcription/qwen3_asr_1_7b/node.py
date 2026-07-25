"""Qwen3-ASR-1.7B transcription node wrapper."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import sys
from pathlib import Path
from typing import Protocol

from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.transcription.common.providers import (
    QWEN3_ASR_1_7B_NODE_ID,
    QWEN3_ASR_1_7B_NODE_VERSION,
    qwen3_asr_language_hint,
    qwen3_asr_trace_details,
)

NODE_ID = QWEN3_ASR_1_7B_NODE_ID
NODE_VERSION = QWEN3_ASR_1_7B_NODE_VERSION
NODE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = NODE_DIR / "checkpoints"


class TranscriptionRunner(Protocol):
    def transcribe(self, audio_path: str, *, language: str = "en") -> str:
        """Transcribe one audio file."""
        ...


def transcribe_qwen3_asr_1_7b(
    audio_path: str,
    *,
    language: str = "en",
    runner: TranscriptionRunner | None = None,
    role: str = "prediction_audio",
) -> tuple[str, PipelineNodeResult]:
    """Transcribe audio with Qwen3-ASR-1.7B and return a trace node."""

    if runner is None:
        from sure_eval.evaluation.nodes.transcription.common.providers import Qwen3ASR17BTranscriber

        runner = Qwen3ASR17BTranscriber(cache_dir=DEFAULT_CACHE_DIR)
    transcript = runner.transcribe(audio_path, language=language)
    language_hint = getattr(runner, "last_language_hint", None) or qwen3_asr_language_hint(language)
    detected_language = getattr(runner, "last_detected_language", None)
    return (
        transcript,
        PipelineNodeResult(
            stage="transcription",
            node_id=NODE_ID,
            version=NODE_VERSION,
            details=qwen3_asr_trace_details(
                audio_path=audio_path,
                language=language,
                role=role,
                transcript=transcript,
                runner=runner,
                detected_language=detected_language,
                language_hint=language_hint,
            ),
            internal_stages=(
                "runtime_managed_audio_frontend",
                "asr_inference",
                "text_extraction",
            ),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe audio with Qwen3-ASR-1.7B.")
    parser.add_argument("--audio-path")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    if bool(args.audio_path) == bool(args.input_jsonl):
        parser.error("exactly one of --audio-path or --input-jsonl is required")

    from sure_eval.evaluation.nodes.transcription.common.providers import Qwen3ASR17BTranscriber

    runner = Qwen3ASR17BTranscriber(device=args.device, cache_dir=DEFAULT_CACHE_DIR)
    if args.input_jsonl:
        input_path = Path(args.input_jsonl)
        rows = [
            (line_no, json.loads(line))
            for line_no, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1)
            if line.strip()
        ]
        if rows and _batchable(rows, default_language=args.language):
            language = str(rows[0][1].get("language") or args.language)
            role = str(rows[0][1].get("role") or "prediction_audio")
            audio_paths = [str(row["audio_path"]) for _, row in rows]
            if args.json_output:
                with redirect_stdout(sys.stderr):
                    batch_results = runner.transcribe_batch(
                        audio_paths,
                        language=language,
                        role=role,
                    )
            else:
                batch_results = runner.transcribe_batch(
                    audio_paths,
                    language=language,
                    role=role,
                )
            for (line_no, _), (transcript, trace) in zip(rows, batch_results, strict=True):
                _emit_payload(
                    audio_path=str(trace.details["audio_path"]),
                    language=language,
                    transcript=transcript,
                    trace=trace,
                    line_no=line_no,
                    json_output=args.json_output,
                )
            return 0

        for line_no, row in rows:
            audio_path = str(row["audio_path"])
            language = str(row.get("language") or args.language)
            role = str(row.get("role") or "prediction_audio")
            if args.json_output:
                with redirect_stdout(sys.stderr):
                    transcript, trace = transcribe_qwen3_asr_1_7b(
                        audio_path,
                        language=language,
                        runner=runner,
                        role=role,
                    )
            else:
                transcript, trace = transcribe_qwen3_asr_1_7b(
                    audio_path,
                    language=language,
                    runner=runner,
                    role=role,
                )
            _emit_payload(
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
            transcript, trace = transcribe_qwen3_asr_1_7b(
                args.audio_path,
                language=args.language,
                runner=runner,
            )
    else:
        transcript, trace = transcribe_qwen3_asr_1_7b(
            args.audio_path,
            language=args.language,
            runner=runner,
        )
    _emit_payload(
        audio_path=args.audio_path,
        language=args.language,
        transcript=transcript,
        trace=trace,
        json_output=args.json_output,
    )
    return 0


def _batchable(rows: list[tuple[int, dict]], *, default_language: str) -> bool:
    languages = {str(row.get("language") or default_language) for _, row in rows}
    roles = {str(row.get("role") or "prediction_audio") for _, row in rows}
    return len(languages) == 1 and len(roles) == 1


def _emit_payload(
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


if __name__ == "__main__":
    raise SystemExit(main())
