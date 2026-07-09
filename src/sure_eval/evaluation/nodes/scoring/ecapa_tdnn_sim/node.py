"""ECAPA-TDNN speaker-similarity scoring node."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import sys
from pathlib import Path

from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.scoring._audio_quality import SpeakerProvider, SpeakerRow, score_speaker_backend

NODE_ID = "scoring/ecapa_tdnn_sim"
NODE_VERSION = "v1"
NODE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = NODE_DIR / "checkpoints"


def score_ecapa_tdnn_sim(rows: list[SpeakerRow], *, provider: SpeakerProvider) -> PipelineNodeResult:
    return score_speaker_backend(
        rows,
        backend_name="ecapa-tdnn",
        metric_name="sim/ecapa-tdnn",
        node_id=NODE_ID,
        provider=provider,
        version=NODE_VERSION,
    )


def build_default_provider(*, device: str = "cuda") -> SpeakerProvider:
    from sure_eval.evaluation.nodes.scoring.common.speaker_providers import (
        ECAPATDNNEmbeddingProvider,
        EmbeddingSpeakerSimilarityProvider,
    )

    return EmbeddingSpeakerSimilarityProvider(
        ECAPATDNNEmbeddingProvider(device=device, cache_dir=DEFAULT_CACHE_DIR),
        backend="speechbrain-ecapa-tdnn-cosine",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score speaker similarity with ECAPA-TDNN.")
    parser.add_argument("--prediction-audio")
    parser.add_argument("--reference-audio")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    if bool(args.input_jsonl) == bool(args.prediction_audio and args.reference_audio):
        parser.error("exactly one of --input-jsonl or --prediction-audio/--reference-audio is required")

    provider = build_default_provider(device=args.device)
    if args.input_jsonl:
        input_path = Path(args.input_jsonl)
        rows: list[SpeakerRow] = []
        for line in input_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append((str(row.get("key", "")), str(row["prediction_audio"]), str(row["reference_audio"])))
        if args.json_output:
            with redirect_stdout(sys.stderr):
                trace = score_ecapa_tdnn_sim(rows, provider=provider)
        else:
            trace = score_ecapa_tdnn_sim(rows, provider=provider)
        payload = {
            "node_id": NODE_ID,
            "version": NODE_VERSION,
            "result": trace.details["result"],
            "keys": trace.details["keys"],
        }
        if args.json_output:
            sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        else:
            print(payload["result"].get("score", ""))
        return 0

    if args.json_output:
        with redirect_stdout(sys.stderr):
            result = provider(args.prediction_audio, args.reference_audio)
    else:
        result = provider(args.prediction_audio, args.reference_audio)
    payload = {
        "node_id": NODE_ID,
        "version": NODE_VERSION,
        "prediction_audio": args.prediction_audio,
        "reference_audio": args.reference_audio,
        "result": result,
    }
    if args.json_output:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    else:
        print(result.get("score", result.get("ASV", "")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
