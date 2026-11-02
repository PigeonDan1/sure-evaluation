"""Normalized speech-to-reverberation modulation energy ratio."""

from __future__ import annotations

import math
from typing import Any

from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.scoring._audio_quality import MOSProvider, MOSRow, score_mos_backend
from sure_eval.evaluation.nodes.scoring._full_reference_audio import read_audio_mono

NODE_ID = "scoring/nsrmr"
NODE_VERSION = "v1"


class NSRMRProvider:
    """Match the local leaderboard's `srmrpy` fast normalized 16 kHz mode."""

    def __call__(self, enhanced_audio: str, reference: str = "", **kwargs: Any) -> dict[str, Any]:
        try:
            from srmrpy import srmr
        except ModuleNotFoundError as exc:
            raise RuntimeError("nSRMR scoring requires `python -m pip install srmrpy`") from exc
        audio, _ = read_audio_mono(enhanced_audio, target_sample_rate=16000)
        if not len(audio):
            raise ValueError("nSRMR requires non-empty audio")
        value = float(srmr(audio, 16000, fast=True, norm=True)[0])
        if not math.isfinite(value):
            raise ValueError("nSRMR score is non-finite")
        return {"score": value, "nsrmr": value, "unit": "ratio"}


def score_nsrmr(rows: list[MOSRow], *, provider: MOSProvider | None = None) -> PipelineNodeResult:
    return score_mos_backend(
        rows, metric_name="nsrmr", node_id=NODE_ID,
        provider=provider or NSRMRProvider(), version=NODE_VERSION,
    )
