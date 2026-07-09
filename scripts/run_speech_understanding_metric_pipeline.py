#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sure_eval.evaluation.base import MetricResult
from sure_eval.evaluation.sure_evaluator import SUREEvaluator
from sure_eval.evaluation.tasks.asr.metrics import CERMetric, WERMetric
from sure_eval.evaluation.tasks.classification.pipeline import evaluate_classification_files
from sure_eval.evaluation.tasks.s2tt.metrics import BLEUMetric
from sure_eval.evaluation.tasks.slu.pipeline import evaluate_slu_files


DEFAULT_TASKS = ("ASR", "S2TT", "SER", "SLU", "GR")


def read_key_text(path: Path) -> tuple[list[str], list[str]]:
    keys: list[str] = []
    texts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" not in line:
                raise ValueError(f"{path} contains a non key-tab-text row: {line!r}")
            key, text = line.split("\t", 1)
            keys.append(key)
            texts.append(text)
    return keys, texts


def metric_to_dict(result: MetricResult, backend: str) -> dict[str, Any]:
    payload = asdict(result)
    payload["backend"] = backend
    return payload


def evaluate_asr(ref_path: Path, hyp_path: Path, language: str) -> dict[str, Any]:
    ref_keys, references = read_key_text(ref_path)
    hyp_keys, predictions = read_key_text(hyp_path)
    ensure_aligned(ref_path, hyp_path, ref_keys, hyp_keys)
    if language == "zh":
        metric = CERMetric()
        backend = "sure_eval.evaluation.tasks.asr.metrics.CERMetric"
    else:
        metric = WERMetric()
        backend = "sure_eval.evaluation.tasks.asr.metrics.WERMetric"
    result = metric.calculate_batch(predictions, references, language=language)
    return metric_to_dict(result, backend)


def evaluate_s2tt(ref_path: Path, hyp_path: Path, language: str) -> dict[str, Any]:
    ref_keys, references = read_key_text(ref_path)
    hyp_keys, predictions = read_key_text(hyp_path)
    ensure_aligned(ref_path, hyp_path, ref_keys, hyp_keys)
    result = BLEUMetric(language=language).calculate_batch(predictions, references)
    payload = metric_to_dict(result, "sure_eval.evaluation.tasks.s2tt.metrics.BLEUMetric")
    sure_result = SUREEvaluator(language=language).evaluate("s2tt", str(ref_path), str(hyp_path))
    payload["details"]["sure_evaluator"] = sure_result
    return payload


def evaluate_classification(ref_path: Path, hyp_path: Path, task: str, prompt_path: Path | None = None) -> dict[str, Any]:
    ref_keys, references = read_key_text(ref_path)
    hyp_keys, predictions = read_key_text(hyp_path)
    ensure_aligned(ref_path, hyp_path, ref_keys, hyp_keys)
    if task == "SLU":
        if prompt_path is None or not prompt_path.exists():
            raise ValueError(f"SLU requires prompt_jsonl: {prompt_path}")
        report = evaluate_slu_files(str(ref_path), str(hyp_path), prompt_jsonl=str(prompt_path))
        backend = "sure_eval.evaluation.tasks.slu.pipeline.evaluate_slu_files"
    else:
        report = evaluate_classification_files(str(ref_path), str(hyp_path), task=task)
        backend = "sure_eval.evaluation.tasks.classification.pipeline.evaluate_classification_files"
    return {
        "metric_name": "accuracy",
        "score": report.score,
        "details": {
            "correct": report.details["scoring_result"]["correct"],
            "total": report.details["scoring_result"]["total"],
            "pipeline_id": report.pipeline_id,
            "input_contract": report.details["input_contract"],
            "input_files": report.details["input_files"],
            "pipeline_trace": [
                {
                    "stage": node.stage,
                    "node_id": node.node_id,
                    "version": node.version,
                    "internal_stages": list(node.internal_stages),
                }
                for node in report.pipeline_trace
            ],
        },
        "backend": backend,
    }


def ensure_aligned(ref_path: Path, hyp_path: Path, ref_keys: list[str], hyp_keys: list[str]) -> None:
    if ref_keys != hyp_keys:
        raise ValueError(
            f"ref/hyp key mismatch for {ref_path.name} and {hyp_path.name}: "
            f"ref_keys={ref_keys}, hyp_keys={hyp_keys}"
        )


def evaluate_task(task: str, artifacts_dir: Path, asr_language: str, s2tt_language: str) -> dict[str, Any]:
    task_lower = task.lower()
    ref_path = artifacts_dir / f"ref_{task_lower}.txt"
    hyp_path = artifacts_dir / f"hyp_{task_lower}.txt"
    prompt_path = artifacts_dir / f"prompt_{task_lower}.jsonl"
    if not ref_path.exists() or not hyp_path.exists():
        missing = [str(path) for path in (ref_path, hyp_path) if not path.exists()]
        return {
            "status": "missing",
            "task": task,
            "missing": missing,
        }

    if task == "ASR":
        metric = evaluate_asr(ref_path, hyp_path, asr_language)
    elif task == "S2TT":
        metric = evaluate_s2tt(ref_path, hyp_path, s2tt_language)
    elif task in {"SER", "SLU", "GR"}:
        metric = evaluate_classification(
            ref_path,
            hyp_path,
            task=task,
            prompt_path=prompt_path if task == "SLU" else None,
        )
    else:
        raise ValueError(f"Unsupported speech understanding task: {task}")

    ref_keys, _ = read_key_text(ref_path)
    return {
        "status": "complete",
        "task": task,
        "num_samples": len(ref_keys),
        "ref_file": str(ref_path),
        "hyp_file": str(hyp_path),
        "metric": metric,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SURE speech-understanding metrics from existing ref/hyp files.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Directory containing ref_<task>.txt and hyp_<task>.txt files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the combined metric report JSON.",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated tasks to evaluate. Defaults to ASR,S2TT,SER,SLU,GR.",
    )
    parser.add_argument("--asr-language", default="zh")
    parser.add_argument("--s2tt-language", default="zh")
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = [task.strip().upper() for task in args.tasks.split(",") if task.strip()]
    errors: list[dict[str, Any]] = []
    task_results: dict[str, Any] = {}

    for task in tasks:
        try:
            task_results[task] = evaluate_task(task, args.artifacts_dir, args.asr_language, args.s2tt_language)
        except Exception as exc:  # noqa: BLE001 - report all task failures in one artifact.
            errors.append({"task": task, "error": str(exc)})
            task_results[task] = {"status": "error", "task": task, "error": str(exc)}

    complete = [task for task, result in task_results.items() if result.get("status") == "complete"]
    report = {
        "ok": not errors and len(complete) == len(tasks),
        "model": args.model,
        "artifacts_dir": str(args.artifacts_dir),
        "tasks": tasks,
        "complete_tasks": complete,
        "metric_namespaces": {
            "ASR": "sure_eval.evaluation.tasks.asr.metrics",
            "S2TT": "sure_eval.evaluation.tasks.s2tt.metrics",
            "SER": "sure_eval.evaluation.tasks.classification",
            "SLU": "sure_eval.evaluation.tasks.slu",
            "GR": "sure_eval.evaluation.tasks.classification",
        },
        "task_results": task_results,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
