"""Validation entrypoint for TTS metric runner dependencies and smoke checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sure_eval.evaluation.cache import get_cache_dir
from sure_eval.evaluation.nodes.transcription.common.providers import configure_model_cache

DEFAULT_CACHE_DIR = get_cache_dir("tts-metrics")


def build_validation_plan(
    suite: str = "all",
    device: str = "cuda",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> dict[str, Any]:
    """Return the models and smoke fixtures used by TTS metric validation."""
    cache_path = Path(cache_dir)
    models = []
    if suite in {"all", "semantic", "pipeline"}:
        models.extend(
            [
                {
                    "metric": "tts_wer",
                    "model_id": "openai/whisper-large-v3",
                    "backend": "transformers",
                    "device": device,
                    "cache_dir": str(cache_path / "semantic" / "huggingface"),
                },
                {
                    "metric": "tts_cer",
                    "model_id": "paraformer-zh",
                    "backend": "funasr",
                    "device": device,
                    "cache_dir": str(cache_path / "semantic" / "modelscope"),
                },
            ]
        )
    if suite in {"all", "speaker", "pipeline"}:
        models.extend(
            [
                {
                    "metric": "sim",
                    "model_id": "wavlm_large_finetune.pth",
                    "base_model_id": "microsoft/wavlm-large",
                    "backend": "seed-tts-wavlm-large-cosine",
                    "device": device,
                    "cache_dir": str(cache_path / "speaker"),
                },
                {
                    "metric": "sim",
                    "model_id": "speechbrain/spkrec-ecapa-voxceleb",
                    "backend": "speechbrain-ecapa-tdnn-cosine",
                    "device": device,
                    "cache_dir": str(cache_path / "speaker" / "speechbrain"),
                },
                {
                    "metric": "sim",
                    "model_id": "iic/speech_eres2net_sv_zh-cn_16k-common",
                    "backend": "modelscope-eres2net-cosine",
                    "device": device,
                    "cache_dir": str(cache_path / "speaker" / "modelscope"),
                },
            ]
        )
    if suite in {"all", "mos", "pipeline"}:
        models.extend(
            [
                {
                    "metric": "dnsmos",
                    "model_id": "DNSMOS P.835 ONNX",
                    "backend": "onnxruntime",
                    "device": "cpu",
                    "cache_dir": str(cache_path / "mos" / "dnsmos"),
                },
                {
                    "metric": "wv-mos",
                    "model_id": "wv_mos checkpoint",
                    "backend": "emergenttts-eval-public",
                    "device": device,
                    "cache_dir": str(cache_path / "mos" / "wv-mos"),
                },
                {
                    "metric": "utmos",
                    "model_id": "sarulab-speech/UTMOS22",
                    "backend": "utmos22",
                    "device": device,
                    "cache_dir": str(cache_path / "mos" / "utmos22"),
                },
            ]
        )
    return {
        "suite": suite,
        "cache_dir": str(cache_path),
        "models": models,
        "fixtures": {
            "english_asr": "tests/fixtures/librispeech/sample_1_367-130732-0006.wav",
            "chinese_asr": "tests/fixtures/aishell1-test/sample_1_BAC009S0764W0385.wav",
            "speaker_enroll": "tests/fixtures/shared/speaker_verification/spk1_enroll.wav",
            "speaker_trial": "tests/fixtures/shared/speaker_verification/spk1_trial.wav",
        },
    }


def _record_ok(name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "status": "ok", "result": result}


def _record_failed(name: str, exc: Exception) -> dict[str, Any]:
    return {
        "name": name,
        "status": "failed",
        "error_type": exc.__class__.__name__,
        "error": str(exc),
    }


def run_semantic_smoke(
    plan: dict[str, Any],
    *,
    device: str,
    cache_dir: str | Path,
) -> list[dict[str, Any]]:
    """Run Whisper/Paraformer semantic smoke checks."""
    from sure_eval.evaluation.nodes.transcription.common.providers import (
        ParaformerZHTranscriber,
        TTSSemanticErrorRateProvider,
        WhisperLargeV3Transcriber,
    )

    fixtures = plan["fixtures"]
    checks: list[dict[str, Any]] = []
    try:
        provider = TTSSemanticErrorRateProvider(
            WhisperLargeV3Transcriber(device=device, cache_dir=Path(cache_dir) / "semantic")
        )
        row = provider(
            fixtures["english_asr"],
            "lobster a la newberg",
            language="en",
            metric="wer",
        )
        checks.append(_record_ok("semantic_whisper_large_v3_en", row))
    except Exception as exc:
        checks.append(_record_failed("semantic_whisper_large_v3_en", exc))

    try:
        provider = TTSSemanticErrorRateProvider(
            ParaformerZHTranscriber(device=device, cache_dir=Path(cache_dir) / "semantic")
        )
        row = provider(
            fixtures["chinese_asr"],
            "在此次申办冬奥会的过程中",
            language="zh",
            metric="cer",
        )
        checks.append(_record_ok("semantic_paraformer_zh", row))
    except Exception as exc:
        checks.append(_record_failed("semantic_paraformer_zh", exc))
    return checks


def run_speaker_smoke(
    plan: dict[str, Any],
    *,
    device: str,
    cache_dir: str | Path,
) -> list[dict[str, Any]]:
    """Run speaker-similarity smoke checks for available backends."""
    from sure_eval.evaluation.nodes.scoring.common.speaker_providers import (
        ECAPATDNNEmbeddingProvider,
        ERes2NetSimilarityProvider,
        EmbeddingSpeakerSimilarityProvider,
        ERes2NetEmbeddingProvider,
        SeedWavLMSpeakerEmbeddingProvider,
    )
    from sure_eval.evaluation.nodes.scoring.wavlm_large_sim.node import DEFAULT_CACHE_DIR, DEFAULT_CHECKPOINT_PATH

    fixtures = plan["fixtures"]
    checks: list[dict[str, Any]] = []
    try:
        provider = EmbeddingSpeakerSimilarityProvider(
            SeedWavLMSpeakerEmbeddingProvider(
                checkpoint_path=DEFAULT_CHECKPOINT_PATH,
                device=device,
                cache_dir=DEFAULT_CACHE_DIR,
            ),
            backend="seed-tts-wavlm-large-cosine",
        )
        row = provider(fixtures["speaker_trial"], fixtures["speaker_enroll"])
        checks.append(_record_ok("speaker_wavlm_large", row))
    except Exception as exc:
        checks.append(_record_failed("speaker_wavlm_large", exc))

    try:
        provider = EmbeddingSpeakerSimilarityProvider(
            ECAPATDNNEmbeddingProvider(device=device, cache_dir=Path(cache_dir) / "speaker"),
            backend="speechbrain-ecapa-tdnn-cosine",
        )
        row = provider(fixtures["speaker_trial"], fixtures["speaker_enroll"])
        checks.append(_record_ok("speaker_ecapa_tdnn", row))
    except Exception as exc:
        checks.append(_record_failed("speaker_ecapa_tdnn", exc))

    try:
        provider = EmbeddingSpeakerSimilarityProvider(
            ERes2NetEmbeddingProvider(device=device, cache_dir=Path(cache_dir) / "speaker"),
            backend="modelscope-eres2net-cosine",
        )
        row = provider(fixtures["speaker_trial"], fixtures["speaker_enroll"])
        checks.append(_record_ok("speaker_eres2net", row))
    except Exception as exc:
        try:
            provider = ERes2NetSimilarityProvider(
                device=device,
                cache_dir=Path(cache_dir) / "speaker",
            )
            row = provider(fixtures["speaker_trial"], fixtures["speaker_enroll"])
            row["fallback"] = "pairwise_pipeline"
            checks.append(_record_ok("speaker_eres2net", row))
        except Exception:
            checks.append(_record_failed("speaker_eres2net", exc))
    return checks


def run_mos_smoke(plan: dict[str, Any], *, device: str) -> list[dict[str, Any]]:
    """Run lightweight MOS provider-normalization checks."""
    from sure_eval.evaluation.nodes.scoring.common.mos_providers import DNSMOSProvider, UTMOSProvider, WVMOSProvider

    checks: list[dict[str, Any]] = []
    try:
        row = DNSMOSProvider(command_runner=lambda _cmd: '{"OVRL": 3.0, "SIG": 3.1}')(
            plan["fixtures"]["english_asr"],
            "",
        )
        checks.append(_record_ok("mos_dnsmos_provider_normalization", row))
    except Exception as exc:
        checks.append(_record_failed("mos_dnsmos_provider_normalization", exc))
    try:
        row = WVMOSProvider(command=["wv-mos", "{audio_path}"], command_runner=lambda _cmd: "3.2")(
            plan["fixtures"]["english_asr"],
            "",
        )
        checks.append(_record_ok("mos_wvmos_provider_normalization", row))
    except Exception as exc:
        checks.append(_record_failed("mos_wvmos_provider_normalization", exc))
    try:
        row = UTMOSProvider(command=["utmos", "{audio_path}"], command_runner=lambda _cmd: '{"utmos": 3.3}')(
            plan["fixtures"]["english_asr"],
            "",
        )
        checks.append(_record_ok("mos_utmos_provider_normalization", row))
    except Exception as exc:
        checks.append(_record_failed("mos_utmos_provider_normalization", exc))
    return checks


def run_pipeline_smoke(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Run the connected TTS pipeline with deterministic local providers."""
    from sure_eval.evaluation.tasks.tts.compat import TTSMetricPipeline, TTSSample
    from sure_eval.evaluation.nodes.transcription.common.providers import StaticTranscriber

    checks: list[dict[str, Any]] = []
    try:
        pipeline = TTSMetricPipeline(
            semantic_transcribers={
                "en": StaticTranscriber("lobster a la newberg"),
                "zh": StaticTranscriber("在此次申办冬奥会的过程中"),
            },
            speaker_providers={
                "wavlm-large": lambda prediction, reference, **kwargs: {"ASV": 0.39},
                "ecapa-tdnn": lambda prediction, reference, **kwargs: {"ASV": 0.31},
                "eres2net": lambda prediction, reference, **kwargs: {"ASV": 0.56},
            },
            mos_providers={
                "dnsmos": lambda prediction, reference="", **kwargs: {"OVRL": 3.0, "SIG": 3.1},
                "wv-mos": lambda prediction, reference="", **kwargs: {"mos": 3.29},
                "utmos": lambda prediction, reference="", **kwargs: {"utmos": 2.75},
            },
        )
        report = pipeline.evaluate(
            [
                TTSSample(
                    prediction_audio=plan["fixtures"]["english_asr"],
                    reference_text="lobster a la newberg",
                    reference_audio=plan["fixtures"]["speaker_enroll"],
                    language="en",
                    sample_id="english_fixture",
                ),
                TTSSample(
                    prediction_audio=plan["fixtures"]["chinese_asr"],
                    reference_text="在此次申办冬奥会的过程中",
                    reference_audio=plan["fixtures"]["speaker_enroll"],
                    language="zh",
                    sample_id="chinese_fixture",
                ),
            ]
        )
        checks.append(
            _record_ok(
                "tts_metric_pipeline_connected",
                {
                    "metrics": {
                        name: result.score
                        for name, result in sorted(report.results.items())
                    },
                    "num_rows": len(report.rows),
                },
            )
        )
    except Exception as exc:
        checks.append(_record_failed("tts_metric_pipeline_connected", exc))
    return checks


def run_smoke(plan: dict[str, Any], *, device: str, cache_dir: str | Path) -> dict[str, Any]:
    """Run selected smoke checks and return a structured validation report."""
    configure_model_cache(cache_dir)
    suite = plan["suite"]
    checks: list[dict[str, Any]] = []
    if suite in {"all", "semantic"}:
        checks.extend(run_semantic_smoke(plan, device=device, cache_dir=cache_dir))
    if suite in {"all", "speaker"}:
        checks.extend(run_speaker_smoke(plan, device=device, cache_dir=cache_dir))
    if suite in {"all", "mos"}:
        checks.extend(run_mos_smoke(plan, device=device))
    if suite in {"all", "pipeline"}:
        checks.extend(run_pipeline_smoke(plan))

    return {
        "suite": suite,
        "cache_dir": plan["cache_dir"],
        "models": plan["models"],
        "fixtures": plan["fixtures"],
        "checks": checks,
        "ok": all(check["status"] == "ok" for check in checks),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["all", "semantic", "speaker", "mos", "pipeline"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    plan = build_validation_plan(suite=args.suite, device=args.device, cache_dir=args.cache_dir)
    if args.plan_only:
        payload = json.dumps(plan, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
        return 0

    report = run_smoke(plan, device=args.device, cache_dir=args.cache_dir)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
