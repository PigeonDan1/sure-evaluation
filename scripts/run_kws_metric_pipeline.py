#!/usr/bin/env python3
"""Run the SURE KWS metric pipeline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sure_eval.evaluation.tasks.kws.pipeline import evaluate_kws_files


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _split_thresholds(value: str | None) -> list[float] | None:
    if not value:
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _metric_report_from_route(report) -> dict[str, Any]:
    return {
        "metrics": report.details["results"],
        "rows": report.details["rows"],
        "summary": report.details["summary"],
    }


def _pipeline_trace_to_dict(report) -> list[dict[str, Any]]:
    return [
        {
            "stage": node.stage,
            "node_id": node.node_id,
            "version": node.version,
            "internal_stages": list(node.internal_stages),
        }
        for node in report.pipeline_trace
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-jsonl", type=Path, help="SURE KWS gt.jsonl file.")
    parser.add_argument("--sample-output", type=Path, help="Wrapper sample_output.json file.")
    parser.add_argument("--wekws-label-file", type=Path, help="WekWS label jsonl file.")
    parser.add_argument("--wekws-score-file", type=Path, help="WekWS score_ctc.py output file.")
    parser.add_argument("--wekws-frame-score-file", type=Path, help="WekWS score.py frame-level output file.")
    parser.add_argument("--keyword", help="Keyword used with WekWS score files.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", help="Comma-separated thresholds. Defaults to 0.00..1.00.")
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.reference_jsonl and args.sample_output:
        input_mode = "sure_json"
        report = evaluate_kws_files(
            reference_jsonl=args.reference_jsonl,
            sample_output=args.sample_output,
            threshold=args.threshold,
            thresholds=_split_thresholds(args.thresholds),
            threshold_step=args.threshold_step,
        )
    elif args.wekws_label_file and args.wekws_score_file and args.keyword:
        input_mode = "wekws_score_ctc"
        report = evaluate_kws_files(
            wekws_label_file=args.wekws_label_file,
            wekws_score_file=args.wekws_score_file,
            keyword=args.keyword,
            threshold=args.threshold,
            thresholds=_split_thresholds(args.thresholds),
            threshold_step=args.threshold_step,
        )
    elif args.wekws_label_file and args.wekws_frame_score_file and args.keyword:
        input_mode = "wekws_frame_score"
        report = evaluate_kws_files(
            wekws_label_file=args.wekws_label_file,
            wekws_frame_score_file=args.wekws_frame_score_file,
            keyword=args.keyword,
            threshold=args.threshold,
            thresholds=_split_thresholds(args.thresholds),
            threshold_step=args.threshold_step,
        )
    else:
        raise SystemExit(
            "Provide either --reference-jsonl + --sample-output, or "
            "--wekws-label-file + --wekws-score-file + --keyword, or "
            "--wekws-label-file + --wekws-frame-score-file + --keyword."
        )

    payload = {
        "ok": True,
        "input_mode": input_mode,
        "threshold": args.threshold,
        "pipeline_id": report.pipeline_id,
        "input_contract": report.details["input_contract"],
        "input_files": report.details["input_files"],
        "pipeline_trace": _pipeline_trace_to_dict(report),
        **_metric_report_from_route(report),
    }
    rendered = json.dumps(_jsonable(payload), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
