"""Runtime provider builders for TTS and VC audio-object metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sure_eval.evaluation.cache import get_cache_dir
from sure_eval.evaluation.scripts.contracts import NODES_ROOT


def build_tts_runtime(
    *,
    metrics: tuple[str, ...] | list[str],
    language: str,
    device: str = "cuda",
    cache_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the injected runtime objects needed by ``evaluate_tts_samples``."""

    return _build_audio_runtime(
        metrics=tuple(metrics),
        language=language,
        device=device,
        cache_dir=Path(cache_dir) if cache_dir else _default_cache_dir("tts"),
    )


def build_vc_runtime(
    *,
    metrics: tuple[str, ...] | list[str],
    language: str,
    device: str = "cuda",
    cache_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the injected runtime objects needed by ``evaluate_vc_samples``."""

    return _build_audio_runtime(
        metrics=tuple(metrics),
        language=language,
        device=device,
        cache_dir=Path(cache_dir) if cache_dir else _default_cache_dir("vc"),
    )


def build_se_runtime(
    *,
    metrics: tuple[str, ...] | list[str],
    device: str = "cuda",
    cache_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build injected runtime objects needed by ``evaluate_se_samples``."""

    requested = {_normalize_se_metric(metric) for metric in metrics}
    runtime = _build_audio_runtime(
        metrics=tuple(requested & {"dnsmos", "wv-mos", "utmos"}),
        language="n/a",
        device=device,
        cache_dir=Path(cache_dir) if cache_dir else _default_cache_dir("se"),
    )
    reference_providers: dict[str, Any] = {}
    if "si-sdr" in requested:
        from sure_eval.evaluation.nodes.scoring._full_reference_audio import SISDRProvider

        reference_providers["si-sdr"] = SISDRProvider()
    if "stoi" in requested:
        from sure_eval.evaluation.nodes.scoring._full_reference_audio import STOIProvider

        reference_providers["stoi"] = STOIProvider()
    if "pesq" in requested:
        from sure_eval.evaluation.nodes.scoring._full_reference_audio import PESQProvider

        reference_providers["pesq"] = PESQProvider()
    return {
        "mos_providers": runtime["mos_providers"],
        "reference_providers": reference_providers,
    }


def build_tse_runtime(
    *,
    metrics: tuple[str, ...] | list[str],
    language: str,
    device: str = "cuda",
    cache_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the injected runtime objects needed by ``evaluate_tse_samples``."""

    return _build_audio_runtime(
        metrics=tuple(metrics),
        language=language,
        device=device,
        cache_dir=Path(cache_dir) if cache_dir else _default_cache_dir("tse"),
    )


def _build_audio_runtime(
    *,
    metrics: tuple[str, ...],
    language: str,
    device: str,
    cache_dir: Path,
) -> dict[str, dict[str, Any]]:
    requested = {metric.lower() for metric in metrics}
    transcribers: dict[str, Any] = {}
    speaker_providers: dict[str, Any] = {}
    mos_providers: dict[str, Any] = {}

    if requested & {"tts_wer", "tts_cer", "vc_wer", "vc_cer", "tse_wer", "tse_cer"}:
        if language.lower().startswith(("zh", "cmn", "yue")):
            from sure_eval.evaluation.nodes.transcription.common.providers import (
                NodeLocalTranscriber,
            )

            transcribers["zh"] = NodeLocalTranscriber(
                node_id="transcription/paraformer_zh",
                node_dir=NODES_ROOT / "transcription" / "paraformer_zh",
                device=device,
            )
        else:
            from sure_eval.evaluation.nodes.transcription.common.providers import (
                NodeLocalTranscriber,
            )

            transcribers["en"] = NodeLocalTranscriber(
                node_id="transcription/whisper_large_v3",
                node_dir=NODES_ROOT / "transcription" / "whisper_large_v3",
                device=device,
            )

    speaker_metrics = {
        metric.removeprefix("sim/") for metric in requested if metric.startswith("sim/")
    }
    if speaker_metrics:
        from sure_eval.evaluation.nodes.scoring.common.node_local import NodeLocalSpeakerProvider

        if "wavlm-large" in speaker_metrics:
            speaker_providers["wavlm-large"] = NodeLocalSpeakerProvider(
                node_id="scoring/wavlm_large_sim",
                node_dir=NODES_ROOT / "scoring" / "wavlm_large_sim",
                device=device,
            )
        if "ecapa-tdnn" in speaker_metrics:
            speaker_providers["ecapa-tdnn"] = NodeLocalSpeakerProvider(
                node_id="scoring/ecapa_tdnn_sim",
                node_dir=NODES_ROOT / "scoring" / "ecapa_tdnn_sim",
                device=device,
            )
        if "eres2net" in speaker_metrics:
            speaker_providers["eres2net"] = NodeLocalSpeakerProvider(
                node_id="scoring/eres2net_sim",
                node_dir=NODES_ROOT / "scoring" / "eres2net_sim",
                device=device,
            )

    mos_metrics = requested & {"dnsmos", "wv-mos", "utmos"}
    if mos_metrics:
        from sure_eval.evaluation.nodes.scoring.common.node_local import NodeLocalMOSProvider

        if "dnsmos" in mos_metrics:
            mos_providers["dnsmos"] = NodeLocalMOSProvider(
                node_id="scoring/dnsmos",
                node_dir=NODES_ROOT / "scoring" / "dnsmos",
                device=device,
            )
        if "wv-mos" in mos_metrics:
            mos_providers["wv-mos"] = NodeLocalMOSProvider(
                node_id="scoring/wv_mos",
                node_dir=NODES_ROOT / "scoring" / "wv_mos",
                device=device,
            )
        if "utmos" in mos_metrics:
            mos_providers["utmos"] = NodeLocalMOSProvider(
                node_id="scoring/utmos",
                node_dir=NODES_ROOT / "scoring" / "utmos",
                device=device,
            )

    return {
        "transcribers": transcribers,
        "speaker_providers": speaker_providers,
        "mos_providers": mos_providers,
    }


def _default_cache_dir(task: str) -> Path:
    return get_cache_dir(f"{task}-metrics")


def _normalize_se_metric(metric: str) -> str:
    normalized = str(metric).strip().lower().replace("_", "-")
    return {
        "sisdr": "si-sdr",
        "si-sdr": "si-sdr",
        "wvmos": "wv-mos",
        "wv-mos": "wv-mos",
    }.get(normalized, normalized)
