"""Equal error rate scoring node."""

from __future__ import annotations

import numpy as np

from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.scoring.common.sv_metrics import compute_eer

NODE_ID = "scoring/det_eer"
NODE_VERSION = "v1"


def score_eer(scores: np.ndarray, labels: np.ndarray) -> PipelineNodeResult:
    eer, threshold = compute_eer(scores, labels)
    return PipelineNodeResult(
        stage="scoring",
        node_id=NODE_ID,
        version=NODE_VERSION,
        details={
            "metric": "eer",
            "result": {
                "metric_name": "eer",
                "score": eer,
                "eer_percent": eer,
                "threshold": threshold,
                "interpolation": "linear_det_crossing",
            },
        },
        internal_stages=("det_curve", "crossing_interpolation"),
    )
