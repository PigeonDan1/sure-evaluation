"""SI-SDR scoring node."""

from __future__ import annotations

import argparse
import json
import sys

from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.scoring._full_reference_audio import (
    FullReferenceAudioProvider,
    FullReferenceAudioRow,
    SISDRProvider,
    score_full_reference_audio_backend,
)

NODE_ID = "scoring/si_sdr"
NODE_VERSION = "v1"


def score_si_sdr(
    rows: list[FullReferenceAudioRow],
    *,
    provider: FullReferenceAudioProvider | None = None,
) -> PipelineNodeResult:
    return score_full_reference_audio_backend(
        rows,
        metric_name="si-sdr",
        node_id=NODE_ID,
        provider=provider or build_default_provider(),
        version=NODE_VERSION,
    )


def build_default_provider() -> FullReferenceAudioProvider:
    return SISDRProvider()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score enhanced speech with SI-SDR.")
    parser.add_argument("--prediction-audio")
    parser.add_argument("--reference-audio")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    has_pair_arg = bool(args.prediction_audio or args.reference_audio)
    has_complete_pair = bool(args.prediction_audio and args.reference_audio)
    if (args.input_jsonl and has_pair_arg) or (not args.input_jsonl and not has_complete_pair):
        parser.error("use either --input-jsonl or --prediction-audio plus --reference-audio")

    provider = build_default_provider()
    if args.input_jsonl:
        rows = _read_rows(args.input_jsonl)
        trace = score_si_sdr(rows, provider=provider)
        payload = {
            "node_id": NODE_ID,
            "version": NODE_VERSION,
            "result": trace.details["result"],
            "keys": trace.details["keys"],
        }
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
        print(payload["result"].get("score", payload["result"].get("si_sdr", "")))
    return 0


def _read_rows(path: str) -> list[FullReferenceAudioRow]:
    rows: list[FullReferenceAudioRow] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                (
                    str(row.get("key", "")),
                    str(row.get("enhanced_audio", row.get("prediction_audio", ""))),
                    str(row["reference_audio"]),
                )
            )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
