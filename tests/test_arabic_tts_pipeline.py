from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


class RecordingTranscriber:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.calls: list[tuple[str, str]] = []

    def transcribe(self, audio_path: str, *, language: str = "ar") -> str:
        self.calls.append((audio_path, language))
        return self.transcript


def _identity_nemo_normalizer(files):
    from sure_eval.evaluation.core.types import PipelineNodeResult

    return (
        files,
        PipelineNodeResult(
            stage="normalization",
            node_id="normalization/nemo_norm",
            version="v1",
            details={"language": "ar", "profile": "ar_tn"},
            internal_stages=("ar_tn",),
        ),
    )


def test_arabic_tts_default_route_uses_cohere_nemo_and_cer() -> None:
    from sure_eval.evaluation.scripts.tts import describe_pipeline

    description = describe_pipeline(language="ar")

    assert description.metric == "cer"
    assert description.execution_metrics == ("tts_cer",)
    assert description.pipeline_id == (
        "tts.ar.cer.cohere_transcribe_arabic_07_2026_v1."
        "nemo_norm_ar_tn_v1.wenet_cer_v1"
    )
    assert description.node_ids == (
        "transcription/cohere_transcribe_arabic_07_2026",
        "normalization/nemo_norm",
        "scoring/wenet_cer",
    )


def test_arabic_default_transcription_does_not_add_funasr_frontend() -> None:
    from sure_eval.evaluation.nodes.transcription.common.audio_semantic import (
        transcription_node_needs_frontend,
    )

    assert transcription_node_needs_frontend("", "ar") is False
    assert transcription_node_needs_frontend("transcription/cohere_transcribe_arabic_07_2026", "ar") is False


def test_arabic_tts_semantic_route_reuses_nemo_asr_cer(monkeypatch) -> None:
    from sure_eval.evaluation.tasks.asr import pipeline as asr_pipeline
    from sure_eval.evaluation.tasks.tts.pipeline import evaluate_tts_samples
    from sure_eval.evaluation.tasks.tts.types import TTSSample

    monkeypatch.setattr(asr_pipeline, "normalize_nemo_key_text_files", _identity_nemo_normalizer)
    transcriber = RecordingTranscriber("واحد وعشرون كتابا")

    report = evaluate_tts_samples(
        [
            TTSSample(
                prediction_audio="hyp.wav",
                reference_text="واحد وعشرون كتابا",
                reference_audio="ref.wav",
                language="ar",
                sample_id="utt1",
            )
        ],
        metrics=("tts_cer",),
        transcribers={"ar": transcriber},
    )

    assert report.score == 0.0
    assert report.pipeline_id == (
        "tts.ar.cer.cohere_transcribe_arabic_07_2026_v1."
        "nemo_norm_ar_tn_v1.wenet_cer_v1"
    )
    assert [node.node_id for node in report.pipeline_trace] == [
        "transcription/cohere_transcribe_arabic_07_2026",
        "normalization/nemo_norm",
        "scoring/wenet_cer",
    ]
    assert report.details["results"]["cer"]["asr_pipeline_id"] == (
        "asr.ar.cer.nemo_norm_ar_tn_v1.wenet_cer_v1"
    )
    assert report.details["rows"][0]["semantic"]["normalizer"] == "nemo:ar_tn"
    assert transcriber.calls == [("hyp.wav", "ar")]


def test_arabic_tts_full_metric_bundle_declares_existing_speaker_and_mos_nodes() -> None:
    from sure_eval.evaluation.scripts.tts import describe_pipeline

    description = describe_pipeline(
        language="ar",
        metrics=(
            "tts_cer",
            "sim/wavlm-large",
            "sim/ecapa-tdnn",
            "sim/eres2net",
            "dnsmos",
            "wv-mos",
            "utmos",
        ),
    )

    assert description.pipeline_kind == "bundle"
    assert description.node_ids == (
        "transcription/cohere_transcribe_arabic_07_2026",
        "normalization/nemo_norm",
        "scoring/wenet_cer",
        "scoring/wavlm_large_sim",
        "scoring/ecapa_tdnn_sim",
        "scoring/eres2net_sim",
        "scoring/dnsmos",
        "scoring/wv_mos",
        "scoring/utmos",
    )
    assert len(description.member_pipeline_ids) == 7
    assert description.required_roles == (
        "prediction_audio",
        "reference_text",
        "reference_audio",
    )


def test_cohere_arabic_transcription_node_records_runtime_managed_frontend() -> None:
    from sure_eval.evaluation.nodes.transcription.cohere_transcribe_arabic_07_2026 import (
        transcribe_cohere_transcribe_arabic_07_2026,
    )

    runner = RecordingTranscriber("مرحبا بالعالم")
    runner.resolved_model_path = "/private/checkpoints/cohere-transcribe-arabic-07-2026"
    transcript, trace = transcribe_cohere_transcribe_arabic_07_2026(
        "sample.wav",
        language="ar",
        runner=runner,
    )

    assert transcript == "مرحبا بالعالم"
    assert trace.node_id == "transcription/cohere_transcribe_arabic_07_2026"
    assert trace.version == "v1"
    assert trace.details["model_id"] == "CohereLabs/cohere-transcribe-arabic-07-2026"
    assert trace.details["resolved_model_path"] == "cohere-transcribe-arabic-07-2026"
    assert trace.details["audio_frontend_policy"] == "runtime_managed"
    assert trace.details["runtime_normalized_sample_rate_hz"] == 16000
    assert trace.internal_stages == (
        "runtime_managed_audio_frontend",
        "batching",
        "asr_inference",
        "text_extraction",
    )
    assert runner.calls == [("sample.wav", "ar")]


def test_cohere_model_assets_are_not_copied_into_tracked_source() -> None:
    node_dir = (
        Path(__file__).resolve().parents[1]
        / "src/sure_eval/evaluation/nodes/transcription/cohere_transcribe_arabic_07_2026"
    )

    assert not (node_dir / "model.safetensors").exists()
    assert not (node_dir / "modeling_cohere_asr.py").exists()


def test_cohere_cuda_dtype_falls_back_to_float16_without_bfloat16_support() -> None:
    from sure_eval.evaluation.nodes.transcription.cohere_transcribe_arabic_07_2026.node import (
        _select_torch_dtype,
    )

    fake_torch = SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        float32="fp32",
        cuda=SimpleNamespace(is_bf16_supported=lambda: False),
    )

    dtype, name = _select_torch_dtype(fake_torch, use_cuda=True)

    assert dtype == "fp16"
    assert name == "float16"


def test_cohere_device_map_normalizes_cli_device_values() -> None:
    from sure_eval.evaluation.nodes.transcription.cohere_transcribe_arabic_07_2026.node import (
        _normalize_device_map,
    )

    assert _normalize_device_map("cuda") == "cuda:0"
    assert _normalize_device_map("0") == "cuda:0"
    assert _normalize_device_map("cuda:2") == "cuda:2"
    assert _normalize_device_map("cpu") == "cpu"


def test_cohere_runner_uses_official_processor_generate_decode_flow(monkeypatch) -> None:
    from sure_eval.evaluation.nodes.transcription.cohere_transcribe_arabic_07_2026 import (
        node as cohere_node,
    )

    calls: dict[str, object] = {}

    class FakeInputs(dict):
        def to(self, device, dtype=None):
            calls["to"] = (device, dtype)
            return self

    class FakeProcessor:
        def __call__(self, audio, *, sampling_rate, return_tensors, language):
            calls["processor"] = (audio, sampling_rate, return_tensors, language)
            return FakeInputs(input_features="features", length="length")

        def batch_decode(self, outputs, *, skip_special_tokens):
            calls["decode"] = (outputs, skip_special_tokens)
            return [" النص الأول ", "النص الثاني"]

    class FakeModel:
        device = "cuda:0"
        dtype = "fp16"

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return "token-ids"

    runner = cohere_node.CohereArabicTranscriber(device="cuda:0")
    runner._model = FakeModel()
    runner._processor = FakeProcessor()
    monkeypatch.setattr(cohere_node, "_load_audio_file", lambda path: f"waveform:{path}")

    transcripts = runner.transcribe_batch(["a.wav", "b.wav"], language="ar")

    assert transcripts == ["النص الأول", "النص الثاني"]
    assert calls["processor"] == (
        ["waveform:a.wav", "waveform:b.wav"],
        16000,
        "pt",
        "ar",
    )
    assert calls["to"] == ("cuda:0", "fp16")
    assert calls["generate"] == {
        "input_features": "features",
        "length": "length",
        "max_new_tokens": 256,
        "do_sample": False,
        "num_beams": 1,
    }
    assert calls["decode"] == ("token-ids", True)


def test_audio_runtime_builds_node_local_cohere_transcriber_for_arabic() -> None:
    from sure_eval.evaluation.audio_runtime import build_tts_runtime

    runtime = build_tts_runtime(
        metrics=("tts_cer",),
        language="ar",
        device="cuda:0",
        transcription_node_id="transcription/cohere_transcribe_arabic_07_2026",
    )

    transcriber = runtime["transcribers"]["ar"]
    assert transcriber.node_id == "transcription/cohere_transcribe_arabic_07_2026"
    assert transcriber.node_dir.name == "cohere_transcribe_arabic_07_2026"
    assert transcriber.device == "cuda:0"


def test_arabic_tts_full_metric_bundle_executes_existing_provider_contracts(monkeypatch) -> None:
    from sure_eval.evaluation.tasks.asr import pipeline as asr_pipeline
    from sure_eval.evaluation.tasks.tts.pipeline import evaluate_tts_samples
    from sure_eval.evaluation.tasks.tts.types import TTSSample

    monkeypatch.setattr(asr_pipeline, "normalize_nemo_key_text_files", _identity_nemo_normalizer)
    report = evaluate_tts_samples(
        [
            TTSSample(
                prediction_audio="hyp.wav",
                reference_text="مرحبا بالعالم",
                reference_audio="ref.wav",
                language="ar",
                sample_id="utt1",
            )
        ],
        metrics=(
            "tts_cer",
            "sim/wavlm-large",
            "sim/ecapa-tdnn",
            "sim/eres2net",
            "dnsmos",
            "wv-mos",
            "utmos",
        ),
        transcribers={"ar": RecordingTranscriber("مرحبا بالعالم")},
        speaker_providers={
            "wavlm-large": lambda *_args, **_kwargs: {"ASV": 0.71},
            "ecapa-tdnn": lambda *_args, **_kwargs: {"ASV": 0.72},
            "eres2net": lambda *_args, **_kwargs: {"ASV": 0.73},
        },
        mos_providers={
            "dnsmos": lambda *_args, **_kwargs: {"OVRL": 3.1},
            "wv-mos": lambda *_args, **_kwargs: {"mos": 3.2},
            "utmos": lambda *_args, **_kwargs: {"utmos": 3.3},
        },
    )

    assert report.pipeline_kind == "bundle"
    assert len(report.member_pipeline_ids) == 7
    assert report.computation_node_ids == (
        "transcription/cohere_transcribe_arabic_07_2026",
        "normalization/nemo_norm",
        "scoring/wenet_cer",
        "scoring/wavlm_large_sim",
        "scoring/ecapa_tdnn_sim",
        "scoring/eres2net_sim",
        "scoring/dnsmos",
        "scoring/wv_mos",
        "scoring/utmos",
    )
    assert set(report.details["results"]) == {
        "cer",
        "spk_sim",
        "sim_ecapa_tdnn",
        "sim_eres2net",
        "dnsmos",
        "wv_mos",
        "utmos",
    }
    assert report.details["results"]["cer"]["score"] == 0.0
    assert report.details["results"]["spk_sim"]["score"] == 0.71
    assert report.details["results"]["sim_ecapa_tdnn"]["score"] == 0.72
    assert report.details["results"]["sim_eres2net"]["score"] == 0.73
    assert report.details["results"]["dnsmos"]["score"] == 3.1
    assert report.details["results"]["wv_mos"]["score"] == 3.2
    assert report.details["results"]["utmos"]["score"] == 3.3
