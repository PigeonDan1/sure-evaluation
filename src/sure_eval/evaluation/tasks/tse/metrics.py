"""TSE evaluation metric definitions.

Signal-level metrics (SI-SDR) are lightweight and computed directly.
Speaker-similarity and MOS metrics reuse the provider-backed definitions from TTS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sure_eval.evaluation.base import MetricResult
from sure_eval.evaluation.tasks.tts.metrics import (
    DNSMOSMetric as _TTSDNSMOSMetric,
)
from sure_eval.evaluation.tasks.tts.metrics import (
    SIMMetric as _TTSSIMMetric,
)
from sure_eval.evaluation.tasks.tts.metrics import (
    UTMOSMetric as _TTSUTMOSMetric,
)
from sure_eval.evaluation.tasks.tts.metrics import (
    WVMOSMetric as _TTSWVMOSMetric,
)
from sure_eval.evaluation.tasks.tts.metrics import MetricSource


@dataclass(frozen=True)
class SISDRMetric:
    """Scale-Invariant Signal-to-Distortion Ratio for TSE signal quality."""

    metric_name: str = "si_sdr"
    source: MetricSource = MetricSource(
        primary_reference="Le Roux et al., 2019 — SDM: SDM",
        method="SI-SDR between extracted and clean reference audio; scale-invariant.",
        score_key="si_sdr",
        higher_is_better=True,
        dependencies=("numpy", "soundfile", "scipy"),
    )
    score_provider: Any = None

    def calculate(
        self,
        prediction: str,
        reference: str,
        *,
        mixed: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        from sure_eval.evaluation.nodes.scoring.si_sdr.node import score_si_sdr

        mixed_arg = [mixed] if mixed else None
        trace = score_si_sdr([("single", prediction, reference)], mixed_paths=mixed_arg)
        result = trace.details["result"]
        details: dict[str, Any] = {
            "num_samples": 1,
            "score_key": "si_sdr",
            "per_sample": result["per_sample"],
        }
        if "si_sdri" in result:
            details["si_sdri"] = result["si_sdri"]
        return MetricResult(
            metric_name=self.metric_name,
            score=float(result["per_sample"][0]["si_sdr"]),
            details=details,
        )

    def calculate_batch(
        self,
        predictions: list[str],
        references: list[str],
        *,
        mixed: list[str] | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        from sure_eval.evaluation.nodes.scoring.si_sdr.node import score_si_sdr

        rows = [
            (f"utt{i + 1}", pred, ref)
            for i, (pred, ref) in enumerate(zip(predictions, references, strict=True))
        ]
        trace = score_si_sdr(rows, mixed_paths=mixed)
        result = trace.details["result"]
        details: dict[str, Any] = {
            "num_samples": len(rows),
            "score_key": "si_sdr",
            "per_sample": result["per_sample"],
        }
        if "si_sdri" in result:
            details["si_sdri"] = result["si_sdri"]
        return MetricResult(
            metric_name=self.metric_name,
            score=float(result["score"]),
            details=details,
        )


class SIMMetric(_TTSSIMMetric):
    """TSE speaker similarity over extracted and reference speaker audio."""


class DNSMOSMetric(_TTSDNSMOSMetric):
    """TSE no-reference DNSMOS audio quality."""


class WVMOSMetric(_TTSWVMOSMetric):
    """TSE Wav2Vec2 MOS audio quality."""


class UTMOSMetric(_TTSUTMOSMetric):
    """TSE UTMOS audio quality."""


__all__ = [
    "DNSMOSMetric",
    "SISDRMetric",
    "SIMMetric",
    "UTMOSMetric",
    "WVMOSMetric",
]