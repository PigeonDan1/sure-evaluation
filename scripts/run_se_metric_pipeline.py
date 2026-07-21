#!/usr/bin/env python3
"""Run the connected SE metric pipeline for one enhanced-audio sample."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sure_eval.evaluation.base import MetricResult
from sure_eval.evaluation.cache import get_cache_dir
from sure_eval.evaluation.nodes.scoring._full_reference_audio import PESQProvider, SISDRProvider, STOIProvider
from sure_eval.evaluation.tasks.se.pipeline import evaluate_se_samples
from sure_eval.evaluation.tasks.se.types import SESample

DEFAULT_CACHE_DIR = get_cache_dir("se-metrics")
DEFAULT_METRICS = ("si-sdr", "stoi", "pesq", "dnsmos", "wv-mos", "utmos")


def _jsonable(value: Any) -> Any:
    if isinstance(value, MetricResult):
        return {
            "metric_name": value.metric_name,
            "score": _jsonable(value.score),
            "details": _jsonable(value.details),
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _build_stub_providers() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {
            "dnsmos": lambda prediction, reference="", **kwargs: {"OVRL": 3.1, "SIG": 3.3},
            "wv-mos": lambda prediction, reference="", **kwargs: {"mos": 3.4},
            "utmos": lambda prediction, reference="", **kwargs: {"utmos": 3.2},
        },
        {
            "si-sdr": lambda prediction, reference, **kwargs: {"si_sdr": 8.0},
            "stoi": lambda prediction, reference, **kwargs: {"stoi": 0.91},
            "pesq": lambda prediction, reference, **kwargs: {"pesq": 2.8},
        },
    )


def _build_real_providers(*, device: str, cache_dir: Path, metrics: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, Any]]:
    mos_providers: dict[str, Any] = {}
    reference_providers: dict[str, Any] = {}
    if set(metrics) & {"dnsmos", "wv-mos", "utmos"}:
        mos_cache = cache_dir / "mos"
        if "dnsmos" in metrics:
            from sure_eval.evaluation.nodes.scoring.common.mos_providers import DNSMOSProvider

            mos_providers["dnsmos"] = DNSMOSProvider(cache_dir=mos_cache)
        if "wv-mos" in metrics:
            from sure_eval.evaluation.nodes.scoring.common.mos_providers import WVMOSProvider

            mos_providers["wv-mos"] = WVMOSProvider(cache_dir=mos_cache, device=device)
        if "utmos" in metrics:
            from sure_eval.evaluation.nodes.scoring.common.mos_providers import UTMOSProvider

            mos_providers["utmos"] = UTMOSProvider(cache_dir=mos_cache, device=device)
    if "si-sdr" in metrics:
        reference_providers["si-sdr"] = SISDRProvider()
    if "stoi" in metrics:
        reference_providers["stoi"] = STOIProvider()
    if "pesq" in metrics:
        reference_providers["pesq"] = PESQProvider()
    return mos_providers, reference_providers


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enhanced-audio", required=True, type=Path)
    parser.add_argument("--noisy-audio", default="", type=Path)
    parser.add_argument("--reference-audio", default="", type=Path)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--sample-id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--stub", action="store_true", help="Use deterministic stub providers instead of models/packages.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _split_metrics(value: str) -> tuple[str, ...]:
    return tuple(_normalize_metric(item) for item in value.split(",") if item.strip())


def _normalize_metric(metric: str) -> str:
    normalized = metric.strip().lower().replace("_", "-")
    return {
        "sisdr": "si-sdr",
        "si-sdr": "si-sdr",
        "wvmos": "wv-mos",
        "wv-mos": "wv-mos",
    }.get(normalized, normalized)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = _split_metrics(args.metrics)
    sample = SESample(
        enhanced_audio=str(args.enhanced_audio),
        noisy_audio="" if str(args.noisy_audio) == "." else str(args.noisy_audio or ""),
        reference_audio="" if str(args.reference_audio) == "." else str(args.reference_audio or ""),
        sample_id=args.sample_id,
    )
    mos_providers, reference_providers = (
        _build_stub_providers()
        if args.stub
        else _build_real_providers(device=args.device, cache_dir=args.cache_dir, metrics=metrics)
    )
    payload: dict[str, Any] = {
        "sample": {
            "enhanced_audio": sample.enhanced_audio,
            "noisy_audio": sample.noisy_audio,
            "reference_audio": sample.reference_audio,
            "sample_id": sample.sample_id,
        },
        "device": args.device,
        "cache_dir": str(args.cache_dir),
        "metrics": list(metrics),
        "stub": args.stub,
    }
    try:
        report = evaluate_se_samples(
            [sample],
            metrics=metrics,
            mos_providers=mos_providers,
            reference_providers=reference_providers,
        )
    except Exception as exc:
        if args.fail_fast:
            raise
        payload.update(
            {
                "ok": False,
                "metrics": {},
                "rows": [],
                "errors": [
                    {
                        "stage": "se",
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                ],
            }
        )
    else:
        payload.update(
            {
                "ok": True,
                "report": {
                    "task": report.task,
                    "metric": report.metric,
                    "score": report.score,
                    "pipeline_id": report.pipeline_id,
                },
                "metrics": report.details["results"],
                "rows": report.details["rows"],
                "errors": [],
            }
        )

    rendered = json.dumps(_jsonable(payload), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
