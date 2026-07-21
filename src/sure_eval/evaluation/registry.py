"""Task-scoped evaluation metric registry."""

from __future__ import annotations

from sure_eval.evaluation.base import Metric
from sure_eval.evaluation.nodes.scoring.wekws_det.metrics import KWSMetric
from sure_eval.evaluation.tasks.asr.metrics import CERMetric, WERMetric
from sure_eval.evaluation.tasks.classification.metrics import AccuracyMetric
from sure_eval.evaluation.tasks.s2tt.metrics import BLEUMetric, BLEURT20Metric, XCOMETXLMetric
from sure_eval.evaluation.tasks.se.metrics import PESQMetric, SISDRMetric, STOIMetric
from sure_eval.evaluation.tasks.tts.metrics import (
    CERMetric as TTSCERMetric,
    DNSMOSMetric,
    SIMMetric,
    UTMOSMetric,
    WERMetric as TTSWERMetric,
    WVMOSMetric,
)
from sure_eval.evaluation.tasks.vc.metrics import CERMetric as VCCERMetric
from sure_eval.evaluation.tasks.vc.metrics import WERMetric as VCWERMetric


class MetricRegistry:
    """Registry for evaluation metrics."""

    _METRICS = {
        "cer": CERMetric,
        "wer": WERMetric,
        "tts_cer": TTSCERMetric,
        "tts_wer": TTSWERMetric,
        "vc_cer": VCCERMetric,
        "vc_wer": VCWERMetric,
        "sim": SIMMetric,
        "dnsmos": DNSMOSMetric,
        "wv-mos": WVMOSMetric,
        "utmos": UTMOSMetric,
        "si-sdr": SISDRMetric,
        "sisdr": SISDRMetric,
        "stoi": STOIMetric,
        "pesq": PESQMetric,
        "accuracy": AccuracyMetric,
        "bleu": BLEUMetric,
        "xcomet_xl": XCOMETXLMetric,
        "bleurt_20": BLEURT20Metric,
        "kws": KWSMetric,
        "kws_accuracy": KWSMetric,
    }

    @classmethod
    def get_metric(cls, name: str, **kwargs) -> Metric:
        """Get a metric instance."""
        metric_class = cls._METRICS.get(name.lower())
        if not metric_class:
            raise ValueError(f"Unknown metric: {name}")
        return metric_class(**kwargs)

    @classmethod
    def list_metrics(cls) -> list[str]:
        """List available metrics."""
        return list(cls._METRICS.keys())
