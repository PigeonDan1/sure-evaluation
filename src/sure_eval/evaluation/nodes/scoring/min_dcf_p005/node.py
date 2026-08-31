"""Normalized minimum detection cost scoring node."""

from __future__ import annotations

import numpy as np

from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.scoring.common.sv_metrics import compute_min_dcf

NODE_ID = "scoring/min_dcf_p005"
NODE_VERSION = "v1"


def score_min_dcf(scores: np.ndarray, labels: np.ndarray) -> PipelineNodeResult:
    min_dcf, threshold = compute_min_dcf(scores, labels, p_target=0.05, c_miss=1.0, c_fa=1.0)
    return PipelineNodeResult(
        stage="scoring",
        node_id=NODE_ID,
        version=NODE_VERSION,
        details={
            "metric": "min_dcf",
            "result": {
                "metric_name": "min_dcf",
                "score": min_dcf,
                "min_dcf": min_dcf,
                "threshold": threshold,
                "normalized": True,
                "p_target": 0.05,
                "c_miss": 1.0,
                "c_fa": 1.0,
            },
        },
        internal_stages=("det_curve", "normalized_detection_cost"),
    )
