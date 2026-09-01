"""Speaker verification task route."""

from sure_eval.evaluation.tasks.sv.pipeline import evaluate_sv_files

__all__ = ["evaluate_sv_files"]
