"""Voice conversion metric definitions."""

from __future__ import annotations

from sure_eval.evaluation.tasks.tts.metrics import CERMetric as _TTSCERMetric
from sure_eval.evaluation.tasks.tts.metrics import DNSMOSMetric as _TTSDNSMOSMetric
from sure_eval.evaluation.tasks.tts.metrics import SIMMetric as _TTSSIMMetric
from sure_eval.evaluation.tasks.tts.metrics import UTMOSMetric as _TTSUTMOSMetric
from sure_eval.evaluation.tasks.tts.metrics import WERMetric as _TTSWERMetric
from sure_eval.evaluation.tasks.tts.metrics import WVMOSMetric as _TTSWVMOSMetric


class WERMetric(_TTSWERMetric):
    """VC semantic WER over converted speech and same-content reference text/audio."""

    metric_name = "vc_wer"


class CERMetric(_TTSCERMetric):
    """VC semantic CER over converted speech and same-content reference text/audio."""

    metric_name = "vc_cer"


class SIMMetric(_TTSSIMMetric):
    """VC speaker similarity over converted and target/reference audio."""


class DNSMOSMetric(_TTSDNSMOSMetric):
    """VC no-reference DNSMOS audio quality."""


class WVMOSMetric(_TTSWVMOSMetric):
    """VC Wav2Vec2 MOS audio quality."""


class UTMOSMetric(_TTSUTMOSMetric):
    """VC UTMOS audio quality."""


__all__ = [
    "CERMetric",
    "DNSMOSMetric",
    "SIMMetric",
    "UTMOSMetric",
    "WERMetric",
    "WVMOSMetric",
]
