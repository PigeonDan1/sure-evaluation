#!/usr/bin/env python3
"""Run KWS scoring through an exact pipeline JSON (same contract as metric run)."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from sure_eval.evaluation.cli_adapters import read_json, run_pipeline_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-jsonl")
    parser.add_argument("--sample-output")
    parser.add_argument("--wekws-label-file")
    parser.add_argument("--wekws-score-file")
    parser.add_argument("--wekws-frame-score-file")
    parser.add_argument("--keyword")
    parser.add_argument("--target-fa-per-hour", type=float)
    parser.add_argument("--include-roc", action="store_true", default=None)
    parser.add_argument("--match-tolerance-sec", type=float)
    args = vars(parser.parse_args(argv))
    pipeline = read_json(args.pop("pipeline"))
    if str(pipeline.get("task", "")).lower() != "kws":
        parser.error("--pipeline must describe a KWS pipeline")
    print(json.dumps(run_pipeline_spec(pipeline, **args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
