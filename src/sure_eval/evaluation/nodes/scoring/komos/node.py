"""KOMOS scoring node for Korean speech."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
from typing import Any

from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.scoring._audio_quality import MOSProvider, MOSRow, score_mos_backend

NODE_ID = "scoring/komos"
NODE_VERSION = "v1"
NODE_DIR = Path(__file__).resolve().parent
MODEL_ID = "upskyy/komos"
MODEL_REVISION = "c558983b9777d4f01aa61003b151dea37db0cc2d"
DEFAULT_MODEL_DIR = NODE_DIR / "checkpoints" / "komos"


class KOMOSProvider:
    """Lazily load the pinned KOMOS model and score audio paths."""

    def __init__(self, *, device: str = "cuda") -> None:
        self.device = device
        self._model: Any | None = None

    def __call__(
        self,
        prediction: str,
        reference: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        score = float(self._load_model().score(prediction))
        return {"score": score, "backend": MODEL_ID}

    def _load_model(self) -> Any:
        if self._model is None:
            from komos import KOMOS

            local_model = os.environ.get("KOMOS_MODEL")
            if not local_model and (DEFAULT_MODEL_DIR / "config.json").is_file():
                local_model = str(DEFAULT_MODEL_DIR)
            if local_model:
                model = KOMOS.from_pretrained(local_model)
            else:
                model = KOMOS.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
            self._model = model.to(self.device).eval()
        return self._model


def build_default_provider(*, device: str = "cuda") -> MOSProvider:
    """Build a lazy provider for the pinned KOMOS model revision."""
    return KOMOSProvider(device=device)


def score_komos(rows: list[MOSRow], *, provider: MOSProvider) -> PipelineNodeResult:
    """Score and aggregate Korean speech quality with an injected provider."""
    return score_mos_backend(
        rows,
        metric_name="komos",
        node_id=NODE_ID,
        provider=provider,
        version=NODE_VERSION,
    )


def main(argv: list[str] | None = None) -> int:
    """Run single-file or JSONL batch scoring for the node-local adapter."""
    parser = argparse.ArgumentParser(description="Score Korean speech quality with KOMOS.")
    parser.add_argument("--prediction-audio")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    if bool(args.input_jsonl) == bool(args.prediction_audio):
        parser.error("exactly one of --input-jsonl or --prediction-audio is required")

    provider = build_default_provider(device=args.device)
    if args.input_jsonl:
        input_path = Path(args.input_jsonl)
        rows: list[MOSRow] = []
        for line in input_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append((str(row.get("key", "")), str(row["prediction_audio"])))
        if args.json_output:
            with redirect_stdout(sys.stderr):
                trace = score_komos(rows, provider=provider)
        else:
            trace = score_komos(rows, provider=provider)
        payload = {
            "node_id": NODE_ID,
            "version": NODE_VERSION,
            "result": trace.details["result"],
            "keys": trace.details["keys"],
        }
        if args.json_output:
            sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        else:
            print(payload["result"]["score"])
        return 0

    if args.json_output:
        with redirect_stdout(sys.stderr):
            result = provider(args.prediction_audio)
    else:
        result = provider(args.prediction_audio)
    payload = {
        "node_id": NODE_ID,
        "version": NODE_VERSION,
        "prediction_audio": args.prediction_audio,
        "result": result,
    }
    if args.json_output:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    else:
        print(result["score"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
