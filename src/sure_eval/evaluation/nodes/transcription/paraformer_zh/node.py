"""Paraformer-ZH transcription node wrapper."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import sys
from pathlib import Path
from typing import Protocol

from sure_eval.evaluation.core.types import PipelineNodeResult

NODE_ID = "transcription/paraformer_zh"
NODE_VERSION = "v1"
NODE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = NODE_DIR / "checkpoints"


class TranscriptionRunner(Protocol):
    def transcribe(self, audio_path: str, *, language: str = "zh") -> str:
        """Transcribe one audio file."""
        ...


def transcribe_paraformer_zh(
    audio_path: str,
    *,
    language: str = "zh",
    runner: TranscriptionRunner | None = None,
    role: str = "prediction_audio",
) -> tuple[str, PipelineNodeResult]:
    """Transcribe Mandarin audio and return a trace node."""

    if runner is None:
        from sure_eval.evaluation.nodes.transcription.common.providers import ParaformerZHTranscriber

        runner = ParaformerZHTranscriber(cache_dir=DEFAULT_CACHE_DIR)
    transcript = runner.transcribe(audio_path, language=language)
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
                "transcript": transcript,
            },
            internal_stages=("audio_decode", "asr_inference", "text_extraction"),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe Mandarin audio with Paraformer-ZH.")
    parser.add_argument("--audio-path")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    if bool(args.audio_path) == bool(args.input_jsonl):
        parser.error("exactly one of --audio-path or --input-jsonl is required")

    from sure_eval.evaluation.nodes.transcription.common.providers import ParaformerZHTranscriber

    runner = ParaformerZHTranscriber(device=args.device, cache_dir=DEFAULT_CACHE_DIR)
    if args.input_jsonl:
        input_path = Path(args.input_jsonl)
        for line_no, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            audio_path = str(row["audio_path"])
            language = str(row.get("language") or args.language)
            role = str(row.get("role") or "prediction_audio")
            if args.json_output:
                with redirect_stdout(sys.stderr):
                    transcript, trace = transcribe_paraformer_zh(
                        audio_path,
                        language=language,
                        runner=runner,
                        role=role,
                    )
            else:
                transcript, trace = transcribe_paraformer_zh(
                    audio_path,
                    language=language,
                    runner=runner,
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
        with redirect_stdout(sys.stderr):
            transcript, trace = transcribe_paraformer_zh(
                args.audio_path,
                language=args.language,
                runner=runner,
            )
    else:
        transcript, trace = transcribe_paraformer_zh(
            args.audio_path,
            language=args.language,
            runner=runner,
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
    raise SystemExit(main())
