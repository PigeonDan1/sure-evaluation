"""Classification compatibility metrics backed by the task route."""

from __future__ import annotations

import tempfile
from pathlib import Path

from sure_eval.evaluation.base import MetricResult


class AccuracyMetric:
    """Accuracy metric for classification tasks."""

    def calculate(
        self,
        prediction: str,
        reference: str,
        **kwargs,
    ) -> MetricResult:
        """Calculate accuracy for single sample."""
        return self.calculate_batch([prediction], [reference], **kwargs)

    def calculate_batch(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs,
    ) -> MetricResult:
        """Calculate accuracy for batch."""
        ref_handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        hyp_handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        ref_path = ref_handle.name
        hyp_path = hyp_handle.name
        ref_handle.close()
        hyp_handle.close()
        try:
            _write_key_text(ref_path, references)
            _write_key_text(hyp_path, predictions)
            from sure_eval.evaluation.tasks.classification.pipeline import evaluate_classification_files

            report = evaluate_classification_files(
                ref_path,
                hyp_path,
                task=kwargs.get("task", "classification"),
                label_spec=kwargs.get("label_spec"),
            )
            result = report.details["scoring_result"]
            return MetricResult(
                metric_name="accuracy",
                score=report.score,
                details={
                    "correct": result["correct"],
                    "total": result["total"],
                    "valid": result["valid"],
                    "invalid": result["invalid"],
                    "pipeline_id": report.pipeline_id,
                    "input_contract": report.details["input_contract"],
                    "input_roles": list(report.details["input_files"].keys()),
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
            )
        finally:
            Path(ref_path).unlink(missing_ok=True)
            Path(hyp_path).unlink(missing_ok=True)


def _write_key_text(path: str, texts: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for index, text in enumerate(texts, start=1):
            handle.write(f"utt{index}\t{text}\n")


__all__ = [
    "AccuracyMetric",
]
