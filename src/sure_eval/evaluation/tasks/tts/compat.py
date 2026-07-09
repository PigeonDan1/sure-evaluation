"""End-to-end orchestration for TTS metric evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from statistics import fmean
from typing import Any

from sure_eval.evaluation.base import MetricResult
from sure_eval.evaluation.tasks.tts.metrics import ScoreProvider
from sure_eval.evaluation.nodes.transcription.common.providers import Transcriber
from sure_eval.evaluation.tasks.tts.types import TTSMetricReport, TTSSample


class TTSMetricPipeline:
    """Connect semantic, speaker-similarity, and MOS providers for TTS evaluation.

    Heavy model inference stays inside the injected providers/transcribers. This
    class only coordinates inputs and applies the existing metric definitions.
    """

    def __init__(
        self,
        *,
        semantic_transcribers: Mapping[str, Transcriber] | None = None,
        semantic_normalizer: str | None = None,
        speaker_provider: ScoreProvider | None = None,
        speaker_providers: Mapping[str, ScoreProvider] | None = None,
        mos_providers: Mapping[str, ScoreProvider] | None = None,
    ) -> None:
        self.semantic_transcribers = dict(semantic_transcribers or {})
        self.semantic_normalizer = semantic_normalizer
        self.speaker_providers = dict(speaker_providers or {})
        if speaker_provider is not None:
            self.speaker_providers.setdefault("wavlm-large", speaker_provider)
        self.mos_providers = dict(mos_providers or {})

    def evaluate(self, samples: list[TTSSample]) -> TTSMetricReport:
        """Evaluate all configured metric families for a batch of TTS samples."""
        if not samples:
            raise ValueError("at least one TTS sample is required")

        rows = [self._base_row(sample) for sample in samples]
        results: dict[str, MetricResult] = {}

        semantic_results = self._evaluate_semantic(samples, rows)
        results.update(semantic_results)

        speaker_results = self._evaluate_speakers(samples, rows)
        results.update(speaker_results)

        for metric_name in ("dnsmos", "wv-mos", "utmos"):
            provider = self.mos_providers.get(metric_name)
            if provider is not None:
                results[metric_name] = self._evaluate_mos(metric_name, provider, samples, rows)

        return TTSMetricReport(results=results, rows=rows)

    @staticmethod
    def _base_row(sample: TTSSample) -> dict[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "prediction_audio": sample.prediction_audio,
            "reference_text": sample.reference_text,
            "reference_audio": sample.reference_audio,
            "language": sample.language,
            "metadata": dict(sample.metadata),
        }

    def _evaluate_semantic(
        self,
        samples: list[TTSSample],
        rows: list[dict[str, Any]],
    ) -> dict[str, MetricResult]:
        grouped: dict[str, list[tuple[int, TTSSample]]] = {}
        for index, sample in enumerate(samples):
            if self._semantic_transcriber_for(sample.language) is None:
                continue
            metric_name = "tts_cer" if self._uses_cer(sample.language) else "tts_wer"
            grouped.setdefault(metric_name, []).append((index, sample))

        results: dict[str, MetricResult] = {}
        for metric_name, indexed_samples in grouped.items():
            from sure_eval.evaluation.tasks.tts.pipeline import evaluate_tts_samples

            route_report = evaluate_tts_samples(
                [sample for _, sample in indexed_samples],
                metrics=(metric_name,),
                semantic_normalizer=self.semantic_normalizer,
                transcribers=self.semantic_transcribers,
            )
            for routed_row, (row_index, _) in zip(route_report.details["rows"], indexed_samples, strict=True):
                rows[row_index]["semantic"] = routed_row["semantic"]
            result_payload = route_report.details["results"][metric_name]
            results[metric_name] = MetricResult(
                metric_name=metric_name,
                score=route_report.score,
                details={
                    "num_samples": result_payload["num_samples"],
                    "score_key": "cer" if metric_name == "tts_cer" else "wer",
                    "sure_result": result_payload["asr_result"],
                    "pipeline_id": route_report.pipeline_id,
                    "input_contract": route_report.details["input_contract"],
                    "input_roles": list(route_report.details["input_files"].keys()),
                    "pipeline_trace": [
                        {
                            "stage": node.stage,
                            "node_id": node.node_id,
                            "version": node.version,
                            "profile": node.details.get("profile"),
                            "role": node.details.get("role"),
                            "internal_stages": list(node.internal_stages),
                        }
                        for node in route_report.pipeline_trace
                    ],
                },
            )
        return results

    def _semantic_transcriber_for(self, language: str) -> Transcriber | None:
        if language in self.semantic_transcribers:
            return self.semantic_transcribers[language]
        family = "zh" if self._uses_cer(language) else "en"
        return self.semantic_transcribers.get(family)

    @staticmethod
    def _uses_cer(language: str) -> bool:
        return language.lower().startswith(("zh", "cmn", "yue"))

    @staticmethod
    def _common_language(languages: list[str], *, default: str) -> str:
        unique = {language for language in languages if language}
        return unique.pop() if len(unique) == 1 else default

    def _evaluate_speakers(
        self,
        samples: list[TTSSample],
        rows: list[dict[str, Any]],
    ) -> dict[str, MetricResult]:
        if not self.speaker_providers:
            return {}

        results: dict[str, MetricResult] = {}
        backend_scores: list[float] = []
        for backend_name, provider in self.speaker_providers.items():
            from sure_eval.evaluation.nodes.scoring._audio_quality_dispatch import score_speaker_metric

            node_rows = []
            row_indexes = []
            for index, sample in enumerate(samples):
                if not sample.reference_audio:
                    continue
                node_rows.append(
                    (
                        sample.sample_id or f"utt{index + 1}",
                        sample.prediction_audio,
                        sample.reference_audio,
                    )
                )
                row_indexes.append(index)
            if not node_rows:
                continue

            result_name = f"sim/{backend_name}"
            node_result = score_speaker_metric(
                node_rows,
                backend_name=backend_name,
                provider=provider,
            )
            node_payload = node_result.details["result"]
            for row_index, raw in zip(row_indexes, node_payload["per_sample"], strict=True):
                rows[row_index].setdefault("speaker", {})[backend_name] = raw
            result = self._metric_result_from_node(result_name, node_payload, node_result)
            results[result_name] = result
            backend_scores.append(result.score)

        if not results:
            raise ValueError("speaker similarity requires reference_audio for at least one sample")
        if "sim" not in results:
            results["sim"] = MetricResult(
                metric_name="sim",
                score=fmean(backend_scores),
                details={
                    "num_backends": len(backend_scores),
                    "backend_metrics": {
                        name: result.score
                        for name, result in results.items()
                        if name.startswith("sim/")
                    },
                },
            )
        return results

    def _evaluate_mos(
        self,
        metric_name: str,
        provider: ScoreProvider,
        samples: list[TTSSample],
        rows: list[dict[str, Any]],
    ) -> MetricResult:
        from sure_eval.evaluation.nodes.scoring._audio_quality_dispatch import score_mos_metric

        node_rows = [
            (sample.sample_id or f"utt{index + 1}", sample.prediction_audio)
            for index, sample in enumerate(samples)
        ]
        node_result = score_mos_metric(node_rows, metric_name=metric_name, provider=provider)
        node_payload = node_result.details["result"]
        for row, raw in zip(rows, node_payload["per_sample"], strict=True):
            row.setdefault("mos", {})[metric_name] = raw
        return self._metric_result_from_node(metric_name, node_payload, node_result)

    @staticmethod
    def _metric_result_from_node(
        metric_name: str,
        node_payload: dict[str, Any],
        node_result,
    ) -> MetricResult:
        details = dict(node_payload)
        details.pop("metric_name", None)
        details["pipeline_trace"] = [
            {
                "stage": node_result.stage,
                "node_id": node_result.node_id,
                "version": node_result.version,
                "internal_stages": list(node_result.internal_stages),
            }
        ]
        return MetricResult(
            metric_name=metric_name,
            score=float(node_payload["score"]),
            details=details,
        )

    @staticmethod
    def _result_from_rows(
        metric_name: str,
        score_key: str,
        rows: list[dict[str, Any]],
        *,
        source: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> MetricResult:
        scores = [float(row[score_key]) for row in rows]
        details: dict[str, Any] = {
            "num_samples": len(rows),
            "score_key": score_key,
            "per_sample": rows,
        }
        if source is not None:
            details["source"] = source
        if extra:
            details.update(extra)
        return MetricResult(
            metric_name=metric_name,
            score=fmean(scores),
            details=details,
        )


def build_default_tts_metric_pipeline(
    *,
    device: str = "cuda",
    cache_dir: str | Path | None = None,
) -> TTSMetricPipeline:
    """Build the validated local-model TTS metric pipeline.

    The returned pipeline wires all three metric families. Model weights are
    loaded lazily by each provider when `evaluate` is called.
    """
    from sure_eval.evaluation.nodes.scoring.common.mos_providers import DNSMOSProvider, UTMOSProvider, WVMOSProvider
    from sure_eval.evaluation.nodes.transcription.common.providers import ParaformerZHTranscriber, WhisperLargeV3Transcriber
    from sure_eval.evaluation.nodes.scoring.common.speaker_providers import (
        ECAPATDNNEmbeddingProvider,
        ERes2NetEmbeddingProvider,
        ERes2NetSimilarityProvider,
        EmbeddingSpeakerSimilarityProvider,
        WavLMSpeakerEmbeddingProvider,
    )

    cache_path = Path(cache_dir) if cache_dir is not None else None
    semantic_cache = cache_path / "semantic" if cache_path is not None else None
    speaker_cache = cache_path / "speaker" if cache_path is not None else None
    mos_cache = cache_path / "mos" if cache_path is not None else None

    return TTSMetricPipeline(
        semantic_transcribers={
            "en": WhisperLargeV3Transcriber(device=device, cache_dir=semantic_cache),
            "zh": ParaformerZHTranscriber(device=device, cache_dir=semantic_cache),
        },
        speaker_providers={
            "wavlm-large": EmbeddingSpeakerSimilarityProvider(
                WavLMSpeakerEmbeddingProvider(device=device, cache_dir=speaker_cache),
                backend="wavlm-large-cosine",
            ),
            "ecapa-tdnn": EmbeddingSpeakerSimilarityProvider(
                ECAPATDNNEmbeddingProvider(device=device, cache_dir=speaker_cache),
                backend="speechbrain-ecapa-tdnn-cosine",
            ),
            "eres2net": ERes2NetSimilarityProvider(
                device=device,
                cache_dir=speaker_cache,
                embedding_provider=ERes2NetEmbeddingProvider(device=device, cache_dir=speaker_cache),
            ),
        },
        mos_providers={
            "dnsmos": DNSMOSProvider(cache_dir=mos_cache),
            "wv-mos": WVMOSProvider(cache_dir=mos_cache, device=device),
            "utmos": UTMOSProvider(cache_dir=mos_cache, device=device),
        },
    )


__all__ = [
    "TTSMetricPipeline",
    "TTSMetricReport",
    "TTSSample",
    "build_default_tts_metric_pipeline",
]
