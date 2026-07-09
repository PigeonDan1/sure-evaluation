"""S2TT evaluation metrics backed by the canonical S2TT task pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

from sure_eval.evaluation.base import MetricResult


class BLEUMetric:
    """BLEU metric for translation."""

    def __init__(self, language: str = "zh") -> None:
        self.language = language

    def calculate(
        self,
        prediction: str,
        reference: str,
        **kwargs,
    ) -> MetricResult:
        """Calculate BLEU for a single sample."""
        return self.calculate_batch([prediction], [reference], **kwargs)

    def calculate_batch(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs,
    ) -> MetricResult:
        """Calculate corpus BLEU through the canonical S2TT pipeline."""
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length")

        ref_handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        hyp_handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        ref_path = ref_handle.name
        hyp_path = hyp_handle.name
        ref_handle.close()
        hyp_handle.close()
        try:
            _write_key_text(ref_path, references)
            _write_key_text(hyp_path, predictions)
            from sure_eval.evaluation.tasks.s2tt.pipeline import evaluate_s2tt_files

            report = evaluate_s2tt_files(
                ref_path,
                hyp_path,
                language=kwargs.get("language", self.language),
                metric="bleu",
            )
            result = report.details["scoring_result"]
            return MetricResult(
                metric_name="bleu",
                score=report.score,
                details={
                    "bleu": result["bleu"],
                    "bleu_char": result["bleu_char"],
                    "chrf": result["chrf"],
                    "score": result["score"],
                    "pipeline_id": report.pipeline_id,
                    "input_contract": report.details["input_contract"],
                    "input_roles": list(report.details["input_files"].keys()),
                    "pipeline_trace": [
                        {
                            "stage": node.stage,
                            "node_id": node.node_id,
                            "version": node.version,
                            "tokenizer_profile": node.details.get("tokenizer_profile"),
                            "tokenizer": node.details.get("tokenizer"),
                            "internal_stages": list(node.internal_stages),
                        }
                        for node in report.pipeline_trace
                    ],
                },
            )
        finally:
            Path(ref_path).unlink(missing_ok=True)
            Path(hyp_path).unlink(missing_ok=True)


class XCOMETXLMetric:
    """XCOMET-XL semantic metric for speech translation."""

    def __init__(self, language: str = "zh") -> None:
        self.language = language

    def calculate(
        self,
        prediction: str,
        reference: str,
        **kwargs,
    ) -> MetricResult:
        source = kwargs.get("source", "")
        return self.calculate_batch([prediction], [reference], sources=[source], **kwargs)

    def calculate_batch(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs,
    ) -> MetricResult:
        """Calculate segment-mean XCOMET-XL through the canonical S2TT pipeline."""
        sources = kwargs.get("sources")
        if sources is None:
            raise ValueError("sources is required for XCOMETXLMetric")
        if len(predictions) != len(references) or len(predictions) != len(sources):
            raise ValueError("predictions, references, and sources must have the same length")

        src_handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        ref_handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        hyp_handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        src_path = src_handle.name
        ref_path = ref_handle.name
        hyp_path = hyp_handle.name
        src_handle.close()
        ref_handle.close()
        hyp_handle.close()
        try:
            _write_key_text(src_path, sources)
            _write_key_text(ref_path, references)
            _write_key_text(hyp_path, predictions)
            from sure_eval.evaluation.tasks.s2tt.pipeline import evaluate_s2tt_files

            report = evaluate_s2tt_files(
                ref_path,
                hyp_path,
                language=kwargs.get("language", self.language),
                metric="xcomet_xl",
                src_file=src_path,
                xcomet_runner=kwargs.get("xcomet_runner"),
            )
            return _metric_result_from_report("xcomet_xl", report)
        finally:
            Path(src_path).unlink(missing_ok=True)
            Path(ref_path).unlink(missing_ok=True)
            Path(hyp_path).unlink(missing_ok=True)


class BLEURT20Metric:
    """BLEURT-20 semantic metric for speech translation."""

    def __init__(self, language: str = "zh") -> None:
        self.language = language

    def calculate(
        self,
        prediction: str,
        reference: str,
        **kwargs,
    ) -> MetricResult:
        return self.calculate_batch([prediction], [reference], **kwargs)

    def calculate_batch(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs,
    ) -> MetricResult:
        """Calculate segment-mean BLEURT-20 through the canonical S2TT pipeline."""
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length")

        ref_handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        hyp_handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        ref_path = ref_handle.name
        hyp_path = hyp_handle.name
        ref_handle.close()
        hyp_handle.close()
        try:
            _write_key_text(ref_path, references)
            _write_key_text(hyp_path, predictions)
            from sure_eval.evaluation.tasks.s2tt.pipeline import evaluate_s2tt_files

            report = evaluate_s2tt_files(
                ref_path,
                hyp_path,
                language=kwargs.get("language", self.language),
                metric="bleurt_20",
                bleurt_runner=kwargs.get("bleurt_runner"),
            )
            return _metric_result_from_report("bleurt_20", report)
        finally:
            Path(ref_path).unlink(missing_ok=True)
            Path(hyp_path).unlink(missing_ok=True)


def _write_key_text(path: str, texts: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for index, text in enumerate(texts, start=1):
            handle.write(f"utt{index}\t{text}\n")


def _metric_result_from_report(metric_name: str, report) -> MetricResult:
    result = report.details["scoring_result"]
    return MetricResult(
        metric_name=metric_name,
        score=report.score,
        details={
            **result,
            "pipeline_id": report.pipeline_id,
            "input_contract": report.details["input_contract"],
            "input_roles": list(report.details["input_files"].keys()),
            "pipeline_trace": [
                {
                    "stage": node.stage,
                    "node_id": node.node_id,
                    "version": node.version,
                    "backend": node.details.get("backend"),
                    "model": node.details.get("model"),
                    "internal_stages": list(node.internal_stages),
                }
                for node in report.pipeline_trace
            ],
        },
    )


__all__ = [
    "BLEUMetric",
    "BLEURT20Metric",
    "XCOMETXLMetric",
]
