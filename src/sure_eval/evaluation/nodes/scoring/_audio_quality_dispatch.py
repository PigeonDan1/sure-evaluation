"""Dispatch helpers for concrete provider-backed audio scoring nodes."""

from __future__ import annotations

from typing import Any

from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.scoring._audio_quality import MOSRow, SpeakerRow
from sure_eval.evaluation.nodes.scoring._full_reference_audio import FullReferenceAudioRow
from sure_eval.evaluation.nodes.scoring.dnsmos import score_dnsmos
from sure_eval.evaluation.nodes.scoring.ecapa_tdnn_sim import score_ecapa_tdnn_sim
from sure_eval.evaluation.nodes.scoring.eres2net_sim import score_eres2net_sim
from sure_eval.evaluation.nodes.scoring.pesq import score_pesq
from sure_eval.evaluation.nodes.scoring.si_sdr import score_si_sdr
from sure_eval.evaluation.nodes.scoring.stoi import score_stoi
from sure_eval.evaluation.nodes.scoring.utmos import score_utmos
from sure_eval.evaluation.nodes.scoring.wavlm_large_sim import score_wavlm_large_sim
from sure_eval.evaluation.nodes.scoring.wv_mos import score_wv_mos


def score_speaker_metric(
    rows: list[SpeakerRow],
    *,
    backend_name: str,
    provider: Any,
) -> PipelineNodeResult:
    if backend_name == "wavlm-large":
        return score_wavlm_large_sim(rows, provider=provider)
    if backend_name == "ecapa-tdnn":
        return score_ecapa_tdnn_sim(rows, provider=provider)
    if backend_name == "eres2net":
        return score_eres2net_sim(rows, provider=provider)
    raise ValueError(f"unsupported speaker similarity backend: {backend_name}")


def score_mos_metric(
    rows: list[MOSRow],
    *,
    metric_name: str,
    provider: Any,
) -> PipelineNodeResult:
    if metric_name == "dnsmos":
        return score_dnsmos(rows, provider=provider)
    if metric_name == "wv-mos":
        return score_wv_mos(rows, provider=provider)
    if metric_name == "utmos":
        return score_utmos(rows, provider=provider)
    raise ValueError(f"unsupported MOS metric: {metric_name}")


def score_full_reference_metric(
    rows: list[FullReferenceAudioRow],
    *,
    metric_name: str,
    provider: Any,
) -> PipelineNodeResult:
    if metric_name == "si-sdr":
        return score_si_sdr(rows, provider=provider)
    if metric_name == "stoi":
        return score_stoi(rows, provider=provider)
    if metric_name == "pesq":
        return score_pesq(rows, provider=provider)
    raise ValueError(f"unsupported full-reference audio metric: {metric_name}")
