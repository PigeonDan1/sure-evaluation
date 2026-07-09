"""End-to-end orchestration for VC metric evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sure_eval.evaluation.tasks.tts.compat import TTSMetricPipeline, TTSMetricReport, TTSSample
from sure_eval.evaluation.tasks.tts.metrics import ScoreProvider
from sure_eval.evaluation.nodes.transcription.common.providers import Transcriber
from sure_eval.evaluation.tasks.vc.types import VCSample


class VCMetricPipeline(TTSMetricPipeline):
    """Connect semantic, speaker-similarity, and MOS providers for VC evaluation.

    VC uses the same validated provider family as TTS. The main input rename is
    `converted_audio`; semantic WER/CER compares converted speech against
    `reference_text` when present, otherwise against a transcript of
    `reference_audio`.
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
        super().__init__(
            semantic_transcribers=semantic_transcribers,
            semantic_normalizer=semantic_normalizer,
            speaker_provider=speaker_provider,
            speaker_providers=speaker_providers,
            mos_providers=mos_providers,
        )

    def evaluate(self, samples: list[VCSample]) -> TTSMetricReport:  # type: ignore[override]
        tts_samples = [self._to_tts_sample(sample) for sample in samples]
        report = super().evaluate(tts_samples)
        for row, sample in zip(report.rows, samples, strict=True):
            row["converted_audio"] = row.pop("prediction_audio")
            row["source_audio"] = sample.source_audio
            if not sample.reference_text:
                row.pop("reference_text", None)
        for old_name, new_name in (("tts_wer", "vc_wer"), ("tts_cer", "vc_cer")):
            if old_name in report.results:
                report.results[new_name] = report.results.pop(old_name)
                report.results[new_name].metric_name = new_name
                for row in report.rows:
                    semantic = row.get("semantic")
                    if isinstance(semantic, dict) and semantic.get("metric") == old_name:
                        semantic["metric"] = new_name
        return report

    def _evaluate_semantic(self, samples: list[TTSSample], rows: list[dict[str, Any]]):  # type: ignore[override]
        grouped: dict[str, list[tuple[int, TTSSample]]] = {}
        for index, sample in enumerate(samples):
            if self._semantic_transcriber_for(sample.language) is None:
                continue
            metric_name = "tts_cer" if self._uses_cer(sample.language) else "tts_wer"
            grouped.setdefault(metric_name, []).append((index, sample))

        results = {}
        for metric_name, indexed_samples in grouped.items():
            from sure_eval.evaluation.base import MetricResult
            from sure_eval.evaluation.tasks.vc.pipeline import evaluate_vc_samples

            vc_metric_name = "vc_cer" if metric_name == "tts_cer" else "vc_wer"
            route_report = evaluate_vc_samples(
                [self._to_vc_sample(sample) for _, sample in indexed_samples],
                metrics=(vc_metric_name,),
                semantic_normalizer=self.semantic_normalizer,
                transcribers=self.semantic_transcribers,
            )
            for routed_row, (row_index, _) in zip(route_report.details["rows"], indexed_samples, strict=True):
                rows[row_index]["semantic"] = routed_row["semantic"]
                rows[row_index]["semantic"]["metric"] = metric_name
            result_payload = route_report.details["results"][vc_metric_name]
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

    @staticmethod
    def _to_tts_sample(sample: VCSample) -> TTSSample:
        metadata = dict(sample.metadata)
        if sample.source_audio:
            metadata.setdefault("source_audio", sample.source_audio)
        return TTSSample(
            prediction_audio=sample.converted_audio,
            reference_text=sample.reference_text,
            reference_audio=sample.reference_audio,
            language=sample.language,
            sample_id=sample.sample_id,
            metadata=metadata,
        )

    @staticmethod
    def _to_vc_sample(sample: TTSSample) -> VCSample:
        metadata = dict(sample.metadata)
        source_audio = str(metadata.pop("source_audio", ""))
        return VCSample(
            converted_audio=sample.prediction_audio,
            reference_audio=sample.reference_audio,
            source_audio=source_audio,
            reference_text=sample.reference_text,
            language=sample.language,
            sample_id=sample.sample_id,
            metadata=metadata,
        )


def build_default_vc_metric_pipeline(
    *,
    device: str = "cuda",
    cache_dir: str | Path | None = None,
) -> VCMetricPipeline:
    """Build the validated local-model VC metric pipeline."""
    from sure_eval.evaluation.tasks.tts.compat import build_default_tts_metric_pipeline

    tts_pipeline = build_default_tts_metric_pipeline(device=device, cache_dir=cache_dir)
    return VCMetricPipeline(
        semantic_transcribers=tts_pipeline.semantic_transcribers,
        speaker_providers=tts_pipeline.speaker_providers,
        mos_providers=tts_pipeline.mos_providers,
    )


__all__ = [
    "VCMetricPipeline",
    "TTSMetricReport",
    "VCSample",
    "build_default_vc_metric_pipeline",
]
