"""TTS evaluation metric definitions.

The audio-quality and speaker-similarity metrics are intentionally lightweight
definitions. Heavy model loading is delegated to an injected score provider or
task runner so importing this module does not download checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Callable, Dict, Union

from sure_eval.evaluation.base import MetricResult
from sure_eval.evaluation.tasks.asr.metrics import CERMetric as _ASRCERMetric
from sure_eval.evaluation.tasks.asr.metrics import WERMetric as _ASRWERMetric

ScoreProvider = Callable[..., Union[float, Dict[str, Any]]]

_RUNTIME_DETAIL_KEYS = {
    "audio_path",
    "decode_time",
    "decode_time_ms",
    "duration",
    "duration_ms",
    "duration_sec",
    "duration_seconds",
    "elapsed",
    "elapsed_ms",
    "elapsed_sec",
    "elapsed_seconds",
    "filename",
    "latency",
    "latency_ms",
    "len_in_sec",
    "num_hops",
    "prediction_audio",
    "reference_audio",
    "reference_text",
    "realtime_factor",
    "real_time_factor",
    "rtf",
    "speed",
    "sr",
    "throughput",
}


@dataclass(frozen=True)
class MetricSource:
    """Reference implementation metadata for a metric definition."""

    primary_reference: str
    method: str
    score_key: str
    higher_is_better: bool
    dependencies: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary_reference": self.primary_reference,
            "method": self.method,
            "score_key": self.score_key,
            "higher_is_better": self.higher_is_better,
            "dependencies": list(self.dependencies),
        }


class WERMetric(_ASRWERMetric):
    """TTS intelligibility WER over ASR transcripts."""

    metric_name = "tts_wer"
    source = MetricSource(
        primary_reference="BytedanceSpeech/seed-tts-eval",
        method=(
            "Transcribe generated speech with Whisper-large-v3 for English or "
            "Paraformer-zh for Mandarin, then compute word error rate against "
            "the target synthesis text with punctuation normalization."
        ),
        score_key="wer",
        higher_is_better=False,
        dependencies=("sure_eval.evaluation.tasks.asr", "jiwer-compatible-error-rate"),
    )


class CERMetric(_ASRCERMetric):
    """TTS intelligibility CER over ASR transcripts."""

    metric_name = "tts_cer"
    source = MetricSource(
        primary_reference="BytedanceSpeech/seed-tts-eval",
        method=(
            "Character-level text error rate for Mandarin TTS intelligibility; "
            "Seed-TTS-Eval obtains transcripts with Paraformer-zh before "
            "computing the same edit-distance error-rate family."
        ),
        score_key="cer",
        higher_is_better=False,
        dependencies=("sure_eval.evaluation.tasks.asr", "jiwer-compatible-error-rate"),
    )


class _ProviderBackedAudioMetric:
    metric_name = "audio_metric"
    source = MetricSource(
        primary_reference="",
        method="",
        score_key="score",
        higher_is_better=True,
    )

    def __init__(self, score_provider: ScoreProvider | None = None) -> None:
        self.score_provider = score_provider

    def calculate(
        self,
        prediction: str,
        reference: str,
        **kwargs: Any,
    ) -> MetricResult:
        return self.calculate_batch([prediction], [reference], **kwargs)

    def calculate_batch(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs: Any,
    ) -> MetricResult:
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length")
        if not predictions:
            raise ValueError("at least one prediction is required")

        provider = self._require_provider()
        rows = [
            self._normalize_provider_result(provider(prediction, reference, **kwargs))
            for prediction, reference in zip(predictions, references, strict=True)
        ]
        scores = [row[self.source.score_key] for row in rows]
        details: dict[str, Any] = {
            "num_samples": len(predictions),
            "score_key": self.source.score_key,
            "source": self.source.as_dict(),
            "per_sample": rows,
        }
        details.update(self._aggregate_extra(rows))
        return MetricResult(
            metric_name=self.metric_name,
            score=fmean(scores),
            details=details,
        )

    def _require_provider(self) -> ScoreProvider:
        if self.score_provider is None:
            raise RuntimeError(
                f"{self.__class__.__name__} requires a score_provider or task-specific runner. "
                "This module only defines metric semantics and does not load heavy evaluation "
                "models at import time."
            )
        return self.score_provider

    def _normalize_provider_result(self, raw_result: float | dict[str, Any]) -> dict[str, Any]:
        if isinstance(raw_result, (float, int)):
            return {self.source.score_key: float(raw_result)}
        if not isinstance(raw_result, dict):
            raise TypeError("score_provider must return a float or a dict")

        normalized = {
            key: value
            for key, value in raw_result.items()
            if str(key).lower() not in _RUNTIME_DETAIL_KEYS
        }
        if self.source.score_key not in normalized:
            fallback_key = self._find_score_key(normalized)
            if fallback_key is None:
                raise KeyError(
                    f"score_provider result must contain '{self.source.score_key}' "
                    "or a recognized metric score key"
                )
            normalized[self.source.score_key] = normalized[fallback_key]
        normalized[self.source.score_key] = float(normalized[self.source.score_key])
        return normalized

    def _find_score_key(self, row: dict[str, Any]) -> str | None:
        return None

    def _aggregate_extra(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {}


class SIMMetric(_ProviderBackedAudioMetric):
    """Speaker similarity using speaker-verification embeddings and cosine score."""

    metric_name = "sim"
    source = MetricSource(
        primary_reference="BytedanceSpeech/seed-tts-eval",
        method=(
            "Cosine similarity between generated speech and prompt/reference speaker "
            "embeddings; Seed-TTS-Eval reports ASV from WavLM-large speaker verification, and "
            "CV3-Eval uses ERes2Net/3D-Speaker."
        ),
        score_key="ASV",
        higher_is_better=True,
        dependencies=("speaker-verification-model", "torch"),
    )

    def _find_score_key(self, row: dict[str, Any]) -> str | None:
        for key in (
            "score",
            "avg_score",
            "avg score",
            "hyp_score",
            "similarity",
            "sim",
            "cosine_similarity",
        ):
            if key in row:
                return key
        return None

    def _aggregate_extra(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if all("ref_score" in row for row in rows):
            extra["mean_ref_similarity"] = fmean(float(row["ref_score"]) for row in rows)
        variance_values = [
            float(row[key])
            for row in rows
            for key in ("ASV-var", "ASV_var", "score_var", "var")
            if key in row
        ]
        if variance_values:
            extra["mean_ASV_var"] = fmean(variance_values)
        return extra


class DNSMOSMetric(_ProviderBackedAudioMetric):
    """No-reference speech quality using DNSMOS P.835-style scores."""

    metric_name = "dnsmos"
    source = MetricSource(
        primary_reference="funaudiollm/cv3-eval",
        method="DNSMOS P.835 ONNX scoring over generated audio clips; primary score is OVRL.",
        score_key="OVRL",
        higher_is_better=True,
        dependencies=("onnxruntime", "librosa", "soundfile"),
    )

    def _find_score_key(self, row: dict[str, Any]) -> str | None:
        for key in ("ovrl", "overall", "P808_MOS", "p808_mos", "mos", "score"):
            if key in row:
                return key
        return None

    def _aggregate_extra(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        extras: dict[str, Any] = {}
        for output_key, aliases in {
            "SIG": ("SIG", "sig"),
            "BAK": ("BAK", "bak"),
            "P808_MOS": ("P808_MOS", "p808_mos", "P808MOS"),
            "OVRL_raw": ("OVRL_raw", "ovrl_raw"),
            "SIG_raw": ("SIG_raw", "sig_raw"),
            "BAK_raw": ("BAK_raw", "bak_raw"),
        }.items():
            values = [
                float(row[key])
                for row in rows
                for key in aliases
                if key in row
            ]
            if values:
                extras[f"mean_{output_key}"] = fmean(values)
        return extras


class WVMOSMetric(_ProviderBackedAudioMetric):
    """Wav2Vec2-based MOS used by EmergentTTS-Eval."""

    metric_name = "wv-mos"
    source = MetricSource(
        primary_reference="boson-ai/emergenttts-eval-public",
        method="Wav2Vec2MOS predicts a MOS value from generated audio.",
        score_key="mos",
        higher_is_better=True,
        dependencies=("torch", "transformers", "librosa"),
    )

    def _find_score_key(self, row: dict[str, Any]) -> str | None:
        for key in ("MOS", "wv_mos", "wv-mos", "wvmos", "score"):
            if key in row:
                return key
        return None


class UTMOSMetric(_ProviderBackedAudioMetric):
    """UTMOS no-reference speech naturalness MOS definition."""

    metric_name = "utmos"
    source = MetricSource(
        primary_reference="sarulab-speech/UTMOS22",
        method="UTMOS22 no-reference MOS prediction from generated audio.",
        score_key="utmos",
        higher_is_better=True,
        dependencies=("utmos", "torch"),
    )

    def _find_score_key(self, row: dict[str, Any]) -> str | None:
        for key in ("UTMOS", "predicted_mos", "mos", "score"):
            if key in row:
                return key
        return None


__all__ = [
    "CERMetric",
    "DNSMOSMetric",
    "MetricSource",
    "SIMMetric",
    "ScoreProvider",
    "UTMOSMetric",
    "WERMetric",
    "WVMOSMetric",
]
