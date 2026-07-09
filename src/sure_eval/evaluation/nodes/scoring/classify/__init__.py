"""Generic classification accuracy scoring node."""

from sure_eval.evaluation.nodes.scoring.classify.node import (
    LabelSpec,
    default_label_spec,
    load_label_spec,
    score_classification_files,
    score_classification_rows,
)

__all__ = [
    "LabelSpec",
    "default_label_spec",
    "load_label_spec",
    "score_classification_files",
    "score_classification_rows",
]
