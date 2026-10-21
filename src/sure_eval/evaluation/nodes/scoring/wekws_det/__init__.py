"""KWS scoring at a fixed false-alarm-per-hour budget."""

from sure_eval.evaluation.nodes.scoring.wekws_det.node import NODE_ID, NODE_VERSION, score_wekws_det
from sure_eval.evaluation.nodes.scoring.wekws_det.metrics import (
    DEFAULT_TARGET_FA_PER_HOUR,
    KWSEvent,
    KWSMetric,
    KWSMacroRecallMetric,
    KWSSample,
    KWSTrigger,
    normalize_keyword,
)

__all__ = [
    "NODE_ID",
    "NODE_VERSION",
    "DEFAULT_TARGET_FA_PER_HOUR",
    "KWSEvent",
    "KWSMetric",
    "KWSMacroRecallMetric",
    "KWSSample",
    "KWSTrigger",
    "normalize_keyword",
    "score_wekws_det",
]
