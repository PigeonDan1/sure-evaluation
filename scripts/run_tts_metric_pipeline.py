#!/usr/bin/env python3
"""Run the connected TTS metric pipeline for one synthesized-audio sample."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from statistics import fmean
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sure_eval.evaluation.cache import get_cache_dir
from sure_eval.evaluation.base import MetricResult
from sure_eval.evaluation.tasks.tts.compat import (
    TTSMetricPipeline,
    TTSSample,
)
from sure_eval.evaluation.nodes.transcription import StaticTranscriber

DEFAULT_CACHE_DIR = get_cache_dir("tts-metrics")


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


def _build_stub_pipeline(language: str, reference_text: str) -> TTSMetricPipeline:
    return TTSMetricPipeline(
        semantic_transcribers={
            language: StaticTranscriber(reference_text),
            "zh": StaticTranscriber(reference_text),
            "en": StaticTranscriber(reference_text),
        },
        speaker_providers={
            "wavlm-large": lambda prediction, reference, **kwargs: {"ASV": 0.39},
            "ecapa-tdnn": lambda prediction, reference, **kwargs: {"ASV": 0.31},
            "eres2net": lambda prediction, reference, **kwargs: {"ASV": 0.56},
        },
        mos_providers={
            "dnsmos": lambda prediction, reference="", **kwargs: {"OVRL": 3.0, "SIG": 3.1},
            "wv-mos": lambda prediction, reference="", **kwargs: {"mos": 3.29},
            "utmos": lambda prediction, reference="", **kwargs: {"utmos": 2.75},
        },
    )


def _filter_pipeline(
    pipeline: TTSMetricPipeline,
    *,
    semantic: bool,
    speaker_backends: set[str],
    mos_backends: set[str],
) -> TTSMetricPipeline:
    if not semantic:
        pipeline.semantic_transcribers.clear()
    if speaker_backends:
        pipeline.speaker_providers = {
            name: provider
            for name, provider in pipeline.speaker_providers.items()
            if name in speaker_backends
        }
    else:
        pipeline.speaker_providers.clear()
    if mos_backends:
        pipeline.mos_providers = {
            name: provider
            for name, provider in pipeline.mos_providers.items()
            if name in mos_backends
        }
    else:
        pipeline.mos_providers.clear()
    return pipeline


def _build_real_pipeline(
    *,
    language: str,
    device: str,
    cache_dir: Path,
    semantic: bool,
    speaker_backends: set[str],
    mos_backends: set[str],
) -> TTSMetricPipeline:
    """Build only the providers needed by this runner segment."""
    semantic_transcribers: dict[str, Any] = {}
    speaker_providers: dict[str, Any] = {}
    mos_providers: dict[str, Any] = {}

    if semantic:
        semantic_cache = cache_dir / "semantic"
        if language.lower().startswith(("zh", "cmn", "yue")):
            from sure_eval.evaluation.nodes.transcription import ParaformerZHTranscriber

            semantic_transcribers["zh"] = ParaformerZHTranscriber(device=device, cache_dir=semantic_cache)
        else:
            from sure_eval.evaluation.nodes.transcription import WhisperLargeV3Transcriber

            semantic_transcribers["en"] = WhisperLargeV3Transcriber(device=device, cache_dir=semantic_cache)

    if speaker_backends:
        from sure_eval.evaluation.nodes.scoring.common.speaker_providers import EmbeddingSpeakerSimilarityProvider

        speaker_cache = cache_dir / "speaker"
        if "wavlm-large" in speaker_backends:
            from sure_eval.evaluation.nodes.scoring.common.speaker_providers import WavLMSpeakerEmbeddingProvider

            speaker_providers["wavlm-large"] = EmbeddingSpeakerSimilarityProvider(
                WavLMSpeakerEmbeddingProvider(device=device, cache_dir=speaker_cache),
                backend="wavlm-large-cosine",
            )
        if "ecapa-tdnn" in speaker_backends:
            from sure_eval.evaluation.nodes.scoring.common.speaker_providers import ECAPATDNNEmbeddingProvider

            speaker_providers["ecapa-tdnn"] = EmbeddingSpeakerSimilarityProvider(
                ECAPATDNNEmbeddingProvider(device=device, cache_dir=speaker_cache),
                backend="speechbrain-ecapa-tdnn-cosine",
            )
        if "eres2net" in speaker_backends:
            from sure_eval.evaluation.nodes.scoring.common.speaker_providers import ERes2NetEmbeddingProvider, ERes2NetSimilarityProvider

            speaker_providers["eres2net"] = ERes2NetSimilarityProvider(
                device=device,
                cache_dir=speaker_cache,
                embedding_provider=ERes2NetEmbeddingProvider(device=device, cache_dir=speaker_cache),
            )

    if mos_backends:
        mos_cache = cache_dir / "mos"
        if "dnsmos" in mos_backends:
            from sure_eval.evaluation.nodes.scoring.common.mos_providers import DNSMOSProvider

            mos_providers["dnsmos"] = DNSMOSProvider(cache_dir=mos_cache)
        if "wv-mos" in mos_backends:
            from sure_eval.evaluation.nodes.scoring.common.mos_providers import WVMOSProvider

            mos_providers["wv-mos"] = WVMOSProvider(cache_dir=mos_cache, device=device)
        if "utmos" in mos_backends:
            from sure_eval.evaluation.nodes.scoring.common.mos_providers import UTMOSProvider

            mos_providers["utmos"] = UTMOSProvider(cache_dir=mos_cache, device=device)

    return TTSMetricPipeline(
        semantic_transcribers=semantic_transcribers,
        speaker_providers=speaker_providers,
        mos_providers=mos_providers,
    )


def _run_one(
    pipeline: TTSMetricPipeline,
    sample: TTSSample,
    *,
    fail_fast: bool,
    semantic_normalizer: str | None = None,
) -> dict[str, Any]:
    effective_semantic_normalizer = (
        semantic_normalizer if semantic_normalizer is not None else pipeline.semantic_normalizer
    )
    if semantic_normalizer is not None:
        pipeline.semantic_normalizer = semantic_normalizer
    if fail_fast:
        report = pipeline.evaluate([sample])
        return {
            "ok": True,
            "metrics": _jsonable(report.results),
            "rows": _jsonable(report.rows),
            "errors": [],
        }

    metrics: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    stages: list[tuple[str, TTSMetricPipeline]] = []
    if pipeline.semantic_transcribers:
        stages.append(
            (
                "semantic",
                TTSMetricPipeline(
                    semantic_transcribers=pipeline.semantic_transcribers,
                    semantic_normalizer=effective_semantic_normalizer,
                ),
            )
        )
    for backend_name, provider in pipeline.speaker_providers.items():
        stages.append((f"speaker/{backend_name}", TTSMetricPipeline(speaker_providers={backend_name: provider})))
    for backend_name, provider in pipeline.mos_providers.items():
        stages.append((f"mos/{backend_name}", TTSMetricPipeline(mos_providers={backend_name: provider})))

    for stage_name, stage_pipeline in stages:
        try:
            report = stage_pipeline.evaluate([sample])
            stage_metrics = _jsonable(report.results)
            if stage_name.startswith("speaker/"):
                stage_metrics.pop("sim", None)
            metrics.update(stage_metrics)
            rows.extend(_jsonable(report.rows))
        except Exception as exc:
            errors.append(
                {
                    "stage": stage_name,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    speaker_scores = [
        metric["score"]
        for name, metric in metrics.items()
        if name.startswith("sim/") and isinstance(metric, dict) and "score" in metric
    ]
    if speaker_scores and "sim" not in metrics:
        metrics["sim"] = {
            "metric_name": "sim",
            "score": fmean(float(score) for score in speaker_scores),
            "details": {
                "num_backends": len(speaker_scores),
                "backend_metrics": {
                    name: metric["score"]
                    for name, metric in metrics.items()
                    if name.startswith("sim/") and isinstance(metric, dict)
                },
            },
        }

    return {
        "ok": not errors,
        "metrics": metrics,
        "rows": rows,
        "errors": errors,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-audio", required=True, type=Path)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--reference-audio", required=True, type=Path)
    parser.add_argument("--language", default="en")
    parser.add_argument("--sample-id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--speaker-backends", default="wavlm-large,ecapa-tdnn,eres2net")
    parser.add_argument("--mos-backends", default="dnsmos,wv-mos,utmos")
    parser.add_argument("--no-semantic", action="store_true")
    parser.add_argument(
        "--semantic-normalizer",
        help='Optional semantic ASR normalizer, for example "wetext:zh_tn".',
    )
    parser.add_argument("--stub", action="store_true", help="Use deterministic stub providers instead of heavy models.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _split_csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sample = TTSSample(
        prediction_audio=str(args.prediction_audio),
        reference_text=args.reference_text,
        reference_audio=str(args.reference_audio),
        language=args.language,
        sample_id=args.sample_id,
    )

    speaker_backends = _split_csv(args.speaker_backends)
    mos_backends = _split_csv(args.mos_backends)
    pipeline = (
        _filter_pipeline(
            _build_stub_pipeline(args.language, args.reference_text),
            semantic=not args.no_semantic,
            speaker_backends=speaker_backends,
            mos_backends=mos_backends,
        )
        if args.stub
        else _build_real_pipeline(
            language=args.language,
            device=args.device,
            cache_dir=args.cache_dir,
            semantic=not args.no_semantic,
            speaker_backends=speaker_backends,
            mos_backends=mos_backends,
        )
    )

    payload = {
        "sample": {
            "prediction_audio": str(args.prediction_audio),
            "reference_text": args.reference_text,
            "reference_audio": str(args.reference_audio),
            "language": args.language,
            "sample_id": args.sample_id,
        },
        "device": args.device,
        "cache_dir": str(args.cache_dir),
        "stub": args.stub,
        **_run_one(
            pipeline,
            sample,
            fail_fast=args.fail_fast,
            semantic_normalizer=args.semantic_normalizer,
        ),
        "semantic_normalizer": args.semantic_normalizer,
    }

    rendered = json.dumps(_jsonable(payload), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
