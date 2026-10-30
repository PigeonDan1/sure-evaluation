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
    transcription_node_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the injected runtime objects needed by ``evaluate_tts_samples``."""

    return _build_audio_runtime(
        metrics=tuple(metrics),
        language=language,
        device=device,
        cache_dir=Path(cache_dir) if cache_dir else _default_cache_dir("tts"),
        transcription_node_id=transcription_node_id,
    )


def build_vc_runtime(
    *,
    metrics: tuple[str, ...] | list[str],
    language: str,
    device: str = "cuda",
    cache_dir: str | Path | None = None,
    transcription_node_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the injected runtime objects needed by ``evaluate_vc_samples``."""

    return _build_audio_runtime(
        metrics=tuple(metrics),
        language=language,
        device=device,
        cache_dir=Path(cache_dir) if cache_dir else _default_cache_dir("vc"),
        transcription_node_id=transcription_node_id,
    )


def build_se_runtime(
    *,
    metrics: tuple[str, ...] | list[str],
    device: str = "cuda",
    cache_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build injected runtime objects needed by ``evaluate_se_samples``."""

    requested = {_normalize_se_metric(metric) for metric in metrics}
    dns_components = requested & {"dnsmos_p808", "dnsmos_sig", "dnsmos_bak"}
    runtime = _build_audio_runtime(
        metrics=tuple((requested & {"dnsmos", "wv-mos", "utmos"}) | ({"dnsmos"} if dns_components else set())),
        language="n/a",
        device=device,
        cache_dir=Path(cache_dir) if cache_dir else _default_cache_dir("se"),
        transcription_node_id=None,
    )
    reference_providers: dict[str, Any] = {}
    if "si-sdr" in requested:
        from sure_eval.evaluation.nodes.scoring._full_reference_audio import SISDRProvider

        reference_providers["si-sdr"] = SISDRProvider()
    if "stoi" in requested:
        from sure_eval.evaluation.nodes.scoring._full_reference_audio import STOIProvider

        reference_providers["stoi"] = STOIProvider()
    if "estoi" in requested:
        from sure_eval.evaluation.nodes.scoring._full_reference_audio import ESTOIProvider

        reference_providers["estoi"] = ESTOIProvider()
    if "lsd" in requested:
        from sure_eval.evaluation.nodes.scoring._full_reference_audio import LSDProvider

        reference_providers["lsd"] = LSDProvider()
    if "lps" in requested:
        from sure_eval.evaluation.nodes.scoring.lps.node import LPSProvider

        reference_providers["lps"] = LPSProvider(device=device)
    if "pesq" in requested:
        from sure_eval.evaluation.nodes.scoring._full_reference_audio import PESQProvider

        reference_providers["pesq"] = PESQProvider()
    if "nsrmr" in requested:
        from sure_eval.evaluation.nodes.scoring.nsrmr.node import NSRMRProvider

        runtime["mos_providers"]["nsrmr"] = NSRMRProvider()
    squim_metrics = requested & {"squim_stoi", "squim_pesq", "squim_si_sdr"}
    if squim_metrics:
        from sure_eval.evaluation.nodes.scoring._squim_objective import SQUIMObjectiveProvider

        squim_provider = SQUIMObjectiveProvider(device=device)
        for metric in squim_metrics:
            runtime["mos_providers"][metric] = squim_provider
    nisqa_metrics = requested & {"nisqa_mos", "nisqa_noi", "nisqa_dis", "nisqa_col", "nisqa_loud"}
    if nisqa_metrics:
        from sure_eval.evaluation.nodes.scoring._nisqa_dim import NISQADIMProvider

        nisqa_provider = NISQADIMProvider(device=device)
        for metric in nisqa_metrics:
            runtime["mos_providers"][metric] = nisqa_provider
    if dns_components:
        dns_provider = runtime["mos_providers"]["dnsmos"]
        for metric in dns_components:
            runtime["mos_providers"][metric] = dns_provider
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
    transcription_node_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the injected runtime objects needed by ``evaluate_tse_samples``."""

    return _build_audio_runtime(
        metrics=tuple(metrics),
        language=language,
        device=device,
        cache_dir=Path(cache_dir) if cache_dir else _default_cache_dir("tse"),
        transcription_node_id=transcription_node_id,
    )


def _build_audio_runtime(
    *,
    metrics: tuple[str, ...],
    language: str,
    device: str,
    cache_dir: Path,
    transcription_node_id: str | None,
) -> dict[str, dict[str, Any]]:
    requested = {metric.lower() for metric in metrics}
    transcribers: dict[str, Any] = {}
    speaker_providers: dict[str, Any] = {}
    mos_providers: dict[str, Any] = {}

    if requested & {"tts_wer", "tts_cer", "vc_wer", "vc_cer", "tse_wer", "tse_cer"}:
        if transcription_node_id == "transcription/cohere_transcribe_arabic_07_2026":
            from sure_eval.evaluation.nodes.transcription.common.providers import (
                NodeLocalTranscriber,
            )

            transcribers["ar"] = NodeLocalTranscriber(
                node_id="transcription/cohere_transcribe_arabic_07_2026",
                node_dir=NODES_ROOT / "transcription" / "cohere_transcribe_arabic_07_2026",
                device=device,
            )
        elif transcription_node_id == "transcription/qwen3_asr_1_7b":
            from sure_eval.evaluation.nodes.transcription.common.providers import (
                NodeLocalTranscriber,
            )

            family = "zh" if language.lower().startswith(("zh", "cmn", "yue")) else "en"
            transcribers[family] = NodeLocalTranscriber(
                node_id="transcription/qwen3_asr_1_7b",
                node_dir=NODES_ROOT / "transcription" / "qwen3_asr_1_7b",
                device=device,
            )
        elif transcription_node_id not in {
            None,
            "transcription/paraformer_zh",
            "transcription/whisper_large_v3",
        }:
            raise ValueError(f"Unsupported semantic transcription node: {transcription_node_id}")
        elif language.lower().startswith(("zh", "cmn", "yue")):
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

    mos_metrics = requested & {"dnsmos", "wv-mos", "utmos", "komos"}
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
        if "komos" in mos_metrics:
            mos_providers["komos"] = NodeLocalMOSProvider(
                node_id="scoring/komos",
                node_dir=NODES_ROOT / "scoring" / "komos",
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
        **{name.replace("_", "-"): name for name in (
            "dnsmos_p808", "dnsmos_sig", "dnsmos_bak", "squim_stoi",
            "squim_pesq", "squim_si_sdr", "nisqa_mos", "nisqa_noi",
            "nisqa_dis", "nisqa_col", "nisqa_loud",
        )},
    }.get(normalized, normalized)


def build_tta_runtime(
    *,
    metrics: tuple[str, ...] | list[str],
    device: str = "cuda",
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build node-local providers for text-to-audio metrics."""
    requested = {str(metric).lower().replace("-", "_") for metric in metrics}
    providers: dict[str, Any] = {}
    if requested & {"fad", "fd", "kl", "is"}:
        providers["audioldm_eval"] = _NodeLocalTTAProvider("scoring/audioldm_eval", device)
    if requested & {"clap", "clap_score"}:
        providers["clap_score"] = _NodeLocalTTAProvider("scoring/clap_score", device)
    if requested & {
        "ce",
        "cu",
        "pc",
        "pq",
        "audiobox_ce",
        "audiobox_cu",
        "audiobox_pc",
        "audiobox_pq",
    }:
        providers["audiobox_aesthetics"] = _NodeLocalTTAProvider(
            "scoring/audiobox_aesthetics", device
        )
    # Keep the runtime contract identical to ``evaluate_tta_samples``: callers
    # receive the provider mapping directly and can pass it to the task
    # executor without an additional unwrap step.
    return providers


def _tta_result_summary(node_id: str, result: dict[str, Any]) -> str:
    if node_id == "scoring/audioldm_eval":
        values = result.get("metrics", {})
    elif node_id == "scoring/clap_score":
        block = result.get("result", {})
        values = {"clap_score": block.get("score")} if isinstance(block, dict) else {}
    else:
        axes = result.get("axes", {})
        values = {
            f"audiobox_{str(axis).lower()}": block.get("score")
            for axis, block in axes.items()
            if isinstance(block, dict)
        }
    return ", ".join(
        f"{name}={float(value):.6g}"
        for name, value in values.items()
        if isinstance(value, (int, float))
    )


class _NodeLocalTTAProvider:
    def __init__(self, node_id: str, device: str) -> None:
        self.node_id, self.device = node_id, device

    def __call__(self, rows, *, metrics=()):
        import json
        import sys
        import tempfile
        import time
        from sure_eval.evaluation.nodes.scoring.common.node_local import _run_node_json

        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        path = Path(handle.name)
        try:
            for row in rows:
                payload = {"key": row[0], "prediction_audio": row[1]}
                if self.node_id == "scoring/audioldm_eval":
                    payload["reference_audio"] = row[2]
                elif self.node_id == "scoring/clap_score":
                    payload["caption"] = row[2]
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.close()
            args = ["--input-jsonl", str(path), "--device", self.device]
            if self.node_id == "scoring/audioldm_eval":
                args.extend(["--metrics", ",".join(metrics)])
            metric_text = f", metrics={','.join(metrics)}" if metrics else ""
            started = time.monotonic()
            print(
                f"[TTA] starting {self.node_id}: {len(rows)} samples{metric_text}",
                file=sys.stderr,
                flush=True,
            )
            try:
                result = _run_node_json(
                    node_id=self.node_id,
                    node_dir=NODES_ROOT / "scoring" / self.node_id.split("/", 1)[1],
                    module_name=f"sure_eval.evaluation.nodes.scoring.{self.node_id.split('/', 1)[1]}.node",
                    args=args,
                    forward_stderr=True,
                )
            except Exception:
                elapsed = time.monotonic() - started
                print(
                    f"[TTA] failed {self.node_id} after {elapsed:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                raise
            summary = _tta_result_summary(self.node_id, result)
            if summary:
                print(
                    f"[TTA] result {self.node_id}: {summary}",
                    file=sys.stderr,
                    flush=True,
                )
            elapsed = time.monotonic() - started
            print(
                f"[TTA] completed {self.node_id} in {elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        finally:
            handle.close()
            path.unlink(missing_ok=True)
        if self.node_id == "scoring/audioldm_eval":
            return dict(result["metrics"])
        if self.node_id == "scoring/clap_score":
            return [item["score"] for item in result["result"]["per_sample"]]
        axes = result["axes"]
        by_id = {str(key): {} for key, _path in rows}
        for axis, block in axes.items():
            for item in block["per_sample"]:
                by_id[str(item["sample_id"])][axis] = item["score"]
        return [by_id[str(key)] for key, _path in rows]
