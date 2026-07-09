"""BLEURT-20 scoring wrapper for S2TT."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult

NODE_ID = "scoring/bleurt_20"
NODE_VERSION = "v1"
MODEL_ID = "BLEURT-20"
NODE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DIR = NODE_DIR / "checkpoints" / "bleurt_20" / "saved_model"


@dataclass(frozen=True)
class SegmentScore:
    """One segment-level BLEURT score."""

    key: str
    score: float


BLEURTRunner = Callable[[List[Dict[str, str]]], List[SegmentScore]]


def score_bleurt_20(
    files: KeyTextFiles,
    *,
    language: str,
    runner: BLEURTRunner | None = None,
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Score aligned hyp/ref key-text files with BLEURT-20."""

    rows = _load_hyp_ref(ref_file=files.ref_file, hyp_file=files.hyp_file)
    scorer = runner or run_bleurt_20_model
    segment_scores = scorer(rows)
    result = _aggregate_segment_scores(segment_scores)
    return (
        files,
        PipelineNodeResult(
            stage="scoring",
            node_id=NODE_ID,
            version=NODE_VERSION,
            details={
                "backend": "bleurt",
                "metric": "bleurt_20",
                "model": MODEL_ID,
                "language": language,
                "result": result,
                "num_samples": len(rows),
            },
            internal_stages=("aligned_input_loading", "bleurt_20_inference", "segment_mean"),
        ),
    )


def run_bleurt_20_model(rows: list[dict[str, str]]) -> list[SegmentScore]:
    """Run BLEURT-20 with the official BLEURT scorer."""

    checkpoint = os.environ.get("BLEURT_20_CHECKPOINT", str(DEFAULT_CHECKPOINT_DIR))
    if not Path(checkpoint).exists():
        raise RuntimeError(
            "scoring/bleurt_20 requires a BLEURT-20 checkpoint at "
            f"{checkpoint}. Set BLEURT_20_CHECKPOINT to override this path."
        )
    try:
        from bleurt import score
    except ImportError as exc:
        raise RuntimeError(
            "scoring/bleurt_20 requires the BLEURT environment. "
            "Run it from src/sure_eval/evaluation/nodes/scoring/bleurt_20 with uv."
        ) from exc

    scorer = score.BleurtScorer(checkpoint)
    raw_scores = scorer.score(
        references=[row["ref"] for row in rows],
        candidates=[row["hyp"] for row in rows],
    )
    if len(raw_scores) != len(rows):
        raise RuntimeError("BLEURT-20 scorer returned a different number of scores than inputs")
    return [
        SegmentScore(key=row["key"], score=float(score_value))
        for row, score_value in zip(rows, raw_scores)
    ]


def _load_hyp_ref(*, ref_file: str, hyp_file: str) -> list[dict[str, str]]:
    ref_rows = _load_key_text(ref_file)
    hyp_rows = _load_key_text(hyp_file)
    if set(ref_rows) != set(hyp_rows):
        raise ValueError("ref and hyp files must contain the same keys")
    return [
        {
            "key": key,
            "hyp": hyp_rows[key],
            "ref": ref_rows[key],
        }
        for key in ref_rows
    ]


def _load_key_text(path: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_number} is not key<TAB>text")
            key, text = parts
            rows[key] = text
    return rows


def _aggregate_segment_scores(segment_scores: list[SegmentScore]) -> dict[str, object]:
    if not segment_scores:
        raise ValueError("BLEURT-20 scoring received no segments")
    scores = [float(item.score) for item in segment_scores]
    return {
        "bleurt_20": sum(scores) / len(scores),
        "score": sum(scores) / len(scores),
        "segment_scores": scores,
        "segment_keys": [item.key for item in segment_scores],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score S2TT outputs with BLEURT-20.")
    parser.add_argument("--ref-file", required=True)
    parser.add_argument("--hyp-file", required=True)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    _, report = score_bleurt_20(
        KeyTextFiles(ref_file=args.ref_file, hyp_file=args.hyp_file),
        language=args.language,
    )
    Path(args.output).write_text(
        json.dumps(report.details["result"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
