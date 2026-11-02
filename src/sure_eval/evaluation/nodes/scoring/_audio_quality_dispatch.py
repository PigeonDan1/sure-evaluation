"""Dispatch helpers for concrete provider-backed audio scoring nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.scoring._audio_quality import MOSRow, SpeakerRow

if TYPE_CHECKING:
    from sure_eval.evaluation.nodes.scoring._full_reference_audio import FullReferenceAudioRow


def score_speaker_metric(
    rows: list[SpeakerRow],
    *,
    backend_name: str,
    provider: Any,
) -> PipelineNodeResult:
    if backend_name == "wavlm-large":
        from sure_eval.evaluation.nodes.scoring.wavlm_large_sim import score_wavlm_large_sim

        return score_wavlm_large_sim(rows, provider=provider)
    if backend_name == "ecapa-tdnn":
        from sure_eval.evaluation.nodes.scoring.ecapa_tdnn_sim import score_ecapa_tdnn_sim

        return score_ecapa_tdnn_sim(rows, provider=provider)
    if backend_name == "eres2net":
        from sure_eval.evaluation.nodes.scoring.eres2net_sim import score_eres2net_sim

        return score_eres2net_sim(rows, provider=provider)
    raise ValueError(f"unsupported speaker similarity backend: {backend_name}")


def score_mos_metric(
    rows: list[MOSRow],
    *,
    metric_name: str,
    provider: Any,
) -> PipelineNodeResult:
    if metric_name == "dnsmos":
        from sure_eval.evaluation.nodes.scoring.dnsmos import score_dnsmos

        return score_dnsmos(rows, provider=provider)
    if metric_name == "dnsmos_p808":
        from sure_eval.evaluation.nodes.scoring.dnsmos_p808 import score_dnsmos_p808

        return score_dnsmos_p808(rows, provider=provider)
    if metric_name == "dnsmos_sig":
        from sure_eval.evaluation.nodes.scoring.dnsmos_sig import score_dnsmos_sig

        return score_dnsmos_sig(rows, provider=provider)
    if metric_name == "dnsmos_bak":
        from sure_eval.evaluation.nodes.scoring.dnsmos_bak import score_dnsmos_bak

        return score_dnsmos_bak(rows, provider=provider)
    if metric_name == "wv-mos":
        from sure_eval.evaluation.nodes.scoring.wv_mos import score_wv_mos

        return score_wv_mos(rows, provider=provider)
    if metric_name == "utmos":
        from sure_eval.evaluation.nodes.scoring.utmos import score_utmos

        return score_utmos(rows, provider=provider)
    if metric_name == "nsrmr":
        from sure_eval.evaluation.nodes.scoring.nsrmr import score_nsrmr

        return score_nsrmr(rows, provider=provider)
    if metric_name == "squim_stoi":
        from sure_eval.evaluation.nodes.scoring.squim_stoi import score_squim_stoi

        return score_squim_stoi(rows, provider=provider)
    if metric_name == "squim_pesq":
        from sure_eval.evaluation.nodes.scoring.squim_pesq import score_squim_pesq

        return score_squim_pesq(rows, provider=provider)
    if metric_name == "squim_si_sdr":
        from sure_eval.evaluation.nodes.scoring.squim_si_sdr import score_squim_si_sdr

        return score_squim_si_sdr(rows, provider=provider)
    if metric_name == "nisqa_mos":
        from sure_eval.evaluation.nodes.scoring.nisqa_mos import score_nisqa_mos

        return score_nisqa_mos(rows, provider=provider)
    if metric_name == "nisqa_noi":
        from sure_eval.evaluation.nodes.scoring.nisqa_noi import score_nisqa_noi

        return score_nisqa_noi(rows, provider=provider)
    if metric_name == "nisqa_dis":
        from sure_eval.evaluation.nodes.scoring.nisqa_dis import score_nisqa_dis

        return score_nisqa_dis(rows, provider=provider)
    if metric_name == "nisqa_col":
        from sure_eval.evaluation.nodes.scoring.nisqa_col import score_nisqa_col

        return score_nisqa_col(rows, provider=provider)
    if metric_name == "nisqa_loud":
        from sure_eval.evaluation.nodes.scoring.nisqa_loud import score_nisqa_loud

        return score_nisqa_loud(rows, provider=provider)
    if metric_name == "komos":
        from sure_eval.evaluation.nodes.scoring.komos import score_komos

        return score_komos(rows, provider=provider)
    raise ValueError(f"unsupported MOS metric: {metric_name}")


def score_full_reference_metric(
    rows: list[FullReferenceAudioRow],
    *,
    metric_name: str,
    provider: Any,
) -> PipelineNodeResult:
    if metric_name == "si-sdr":
        from sure_eval.evaluation.nodes.scoring.si_sdr import score_si_sdr

        return score_si_sdr(rows, provider=provider)
    if metric_name == "stoi":
        from sure_eval.evaluation.nodes.scoring.stoi import score_stoi

        return score_stoi(rows, provider=provider)
    if metric_name == "estoi":
        from sure_eval.evaluation.nodes.scoring.estoi import score_estoi

        return score_estoi(rows, provider=provider)
    if metric_name == "lsd":
        from sure_eval.evaluation.nodes.scoring.lsd import score_lsd

        return score_lsd(rows, provider=provider)
    if metric_name == "lps":
        from sure_eval.evaluation.nodes.scoring.lps import score_lps

        return score_lps(rows, provider=provider)
    if metric_name == "pesq":
        from sure_eval.evaluation.nodes.scoring.pesq import score_pesq

        return score_pesq(rows, provider=provider)
    raise ValueError(f"unsupported full-reference audio metric: {metric_name}")
