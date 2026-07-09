"""TTS task-level evaluation routes.

The package keeps heavy compatibility metrics lazy so metric-only evaluation
images can import lightweight route modules without ASR normalization deps.
"""

from __future__ import annotations

from typing import Any

from sure_eval.evaluation.tasks.tts.pipeline import evaluate_tts_samples
from sure_eval.evaluation.tasks.tts.types import TTSMetricReport, TTSSample

__all__ = [
    "CERMetric",
    "DNSMOSMetric",
    "MetricSource",
    "SIMMetric",
    "TTSMetricPipeline",
    "TTSMetricReport",
    "TTSSample",
    "UTMOSMetric",
    "WERMetric",
    "WVMOSMetric",
    "build_default_tts_metric_pipeline",
    "evaluate_tts_samples",
]


def __getattr__(name: str) -> Any:
    if name in {"TTSMetricPipeline", "build_default_tts_metric_pipeline"}:
        from sure_eval.evaluation.tasks.tts import compat

        return getattr(compat, name)
    if name in {"CERMetric", "DNSMOSMetric", "MetricSource", "SIMMetric", "UTMOSMetric", "WERMetric", "WVMOSMetric"}:
        from sure_eval.evaluation.tasks.tts import metrics

        return getattr(metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
