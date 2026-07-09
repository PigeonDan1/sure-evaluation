"""XCOMET-XL scoring wrapper for S2TT."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sure_eval.evaluation.core.types import PipelineNodeResult

NODE_ID = "scoring/xcomet_xl"
NODE_VERSION = "v1"
MODEL_ID = "Unbabel/XCOMET-XL"
NODE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DIR = NODE_DIR / "checkpoints" / "xcomet_xl" / "huggingface"
DEFAULT_ENCODER_DIR = NODE_DIR / "checkpoints" / "xlm_roberta_xl" / "huggingface"
DEFAULT_MODELSCOPE_DIR = (
    NODE_DIR / "checkpoints" / "xcomet_xl" / "modelscope" / "evalscope" / "XCOMET-XL"
)


@dataclass(frozen=True)
class SegmentScore:
    """One segment-level semantic score."""

    key: str
    score: float


XCOMETRunner = Callable[[list[dict[str, str]]], list[SegmentScore]]


def score_xcomet_xl(
    *,
    src_file: str,
    ref_file: str,
    hyp_file: str,
    language: str,
    runner: XCOMETRunner | None = None,
) -> PipelineNodeResult:
    """Score aligned src/hyp/ref key-text files with XCOMET-XL."""

    rows = _load_src_hyp_ref(src_file=src_file, ref_file=ref_file, hyp_file=hyp_file)
    scorer = runner or run_xcomet_xl_model
    segment_scores = scorer(rows)
    result = _aggregate_segment_scores(segment_scores)
    return PipelineNodeResult(
        stage="scoring",
        node_id=NODE_ID,
        version=NODE_VERSION,
        details={
            "backend": "comet",
            "metric": "xcomet_xl",
            "model": MODEL_ID,
            "language": language,
            "result": result,
            "num_samples": len(rows),
        },
        internal_stages=("aligned_input_loading", "xcomet_xl_inference", "segment_mean"),
    )


def run_xcomet_xl_model(rows: list[dict[str, str]]) -> list[SegmentScore]:
    """Run Unbabel/XCOMET-XL through COMET.

    This imports COMET lazily so the main SURE-EVAL environment does not need the
    heavy semantic-metric dependencies unless this node is selected.
    """

    checkpoint_path = os.environ.get("XCOMET_XL_CHECKPOINT_PATH")
    if checkpoint_path is not None:
        model_path = checkpoint_path
    else:
        model_path = _find_local_xcomet_checkpoint(DEFAULT_MODELSCOPE_DIR)
    _configure_transformer_cache(offline=model_path is not None)

    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as exc:
        raise RuntimeError(
            "scoring/xcomet_xl requires the COMET environment. "
            "Run it from src/sure_eval/evaluation/nodes/scoring/xcomet_xl with uv."
        ) from exc

    if model_path is None:
        checkpoint_dir = Path(os.environ.get("XCOMET_XL_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_path = download_model(MODEL_ID, saving_directory=checkpoint_dir)
    model = load_from_checkpoint(model_path)
    model_inputs = [
        {
            "src": row["src"],
            "mt": row["hyp"],
            "ref": row["ref"],
        }
        for row in rows
    ]
    prediction = model.predict(model_inputs, batch_size=8, gpus=0)
    raw_scores = _extract_scores(prediction)
    return [
        SegmentScore(key=row["key"], score=float(score))
        for row, score in zip(rows, raw_scores, strict=True)
    ]


def _load_src_hyp_ref(*, src_file: str, ref_file: str, hyp_file: str) -> list[dict[str, str]]:
    src_rows = _load_key_text(src_file)
    ref_rows = _load_key_text(ref_file)
    hyp_rows = _load_key_text(hyp_file)
    if set(src_rows) != set(ref_rows) or set(ref_rows) != set(hyp_rows):
        raise ValueError("src, ref, and hyp files must contain the same keys")
    return [
        {
            "key": key,
            "src": src_rows[key],
            "hyp": hyp_rows[key],
            "ref": ref_rows[key],
        }
        for key in src_rows
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
        raise ValueError("XCOMET-XL scoring received no segments")
    scores = [float(item.score) for item in segment_scores]
    return {
        "xcomet_xl": sum(scores) / len(scores),
        "score": sum(scores) / len(scores),
        "segment_scores": scores,
        "segment_keys": [item.key for item in segment_scores],
    }


def _extract_scores(prediction: object) -> list[float]:
    if isinstance(prediction, dict):
        if "scores" in prediction:
            return [float(score) for score in prediction["scores"]]
        if "metadata" in prediction and isinstance(prediction["metadata"], dict):
            scores = prediction["metadata"].get("scores")
            if scores is not None:
                return [float(score) for score in scores]
    scores = getattr(prediction, "scores", None)
    if scores is not None:
        return [float(score) for score in scores]
    raise RuntimeError("COMET prediction did not expose segment scores")


def _find_local_xcomet_checkpoint(root: Path) -> str | None:
    if not root.exists():
        return None
    for pattern in ("*.ckpt", "**/*.ckpt"):
        matches = sorted(root.glob(pattern))
        if matches:
            return str(matches[0])
    return None


def _configure_transformer_cache(*, offline: bool) -> None:
    os.environ.setdefault("HF_HOME", str(DEFAULT_ENCODER_DIR))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DEFAULT_ENCODER_DIR / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_ENCODER_DIR / "transformers"))
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score S2TT outputs with Unbabel/XCOMET-XL.")
    parser.add_argument("--src-file", required=True)
    parser.add_argument("--ref-file", required=True)
    parser.add_argument("--hyp-file", required=True)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    report = score_xcomet_xl(
        src_file=args.src_file,
        ref_file=args.ref_file,
        hyp_file=args.hyp_file,
        language=args.language,
    )
    Path(args.output).write_text(
        json.dumps(report.details["result"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
