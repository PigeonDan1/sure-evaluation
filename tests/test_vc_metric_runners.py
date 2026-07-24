from __future__ import annotations

import json
import subprocess
import sys


class PathEchoTranscriber:
    def transcribe(self, audio_path: str, *, language: str = "en") -> str:
        return "你好世界" if "same" in audio_path else "你好火星"


def test_vc_metric_pipeline_connects_speaker_and_mos() -> None:
    from sure_eval.evaluation.tasks.vc.compat import VCMetricPipeline, VCSample

    sample = VCSample(
        converted_audio="converted.wav",
        reference_audio="target.wav",
        source_audio="source.wav",
    )
    pipeline = VCMetricPipeline(
        speaker_providers={
            "wavlm-large": lambda prediction, reference, **kwargs: {"ASV": 0.8},
            "ecapa-tdnn": lambda prediction, reference, **kwargs: {"ASV": 0.7},
        },
        mos_providers={
            "dnsmos": lambda prediction, reference="", **kwargs: {"OVRL": 3.1},
            "wv-mos": lambda prediction, reference="", **kwargs: {"mos": 3.2},
            "utmos": lambda prediction, reference="", **kwargs: {"utmos": 3.3},
        },
    )

    report = pipeline.evaluate([sample])

    assert set(report.results) == {"sim/wavlm-large", "sim/ecapa-tdnn", "sim", "dnsmos", "wv-mos", "utmos"}
    assert report.results["sim"].score == 0.75
    assert report.results["dnsmos"].score == 3.1
    assert report.results["sim/wavlm-large"].details["pipeline_trace"][0]["node_id"] == (
        "scoring/wavlm_large_sim"
    )
    assert report.results["utmos"].details["pipeline_trace"][0]["node_id"] == "scoring/utmos"
    assert report.rows[0]["converted_audio"] == "converted.wav"
    assert report.rows[0]["reference_audio"] == "target.wav"


def test_vc_metric_pipeline_scores_semantic_against_reference_text_when_available() -> None:
    from sure_eval.evaluation.nodes.transcription import StaticTranscriber
    from sure_eval.evaluation.tasks.vc.compat import VCMetricPipeline, VCSample

    sample = VCSample(
        converted_audio="converted.wav",
        reference_audio="target.wav",
        reference_text="你好世界",
        language="zh",
    )
    pipeline = VCMetricPipeline(
        semantic_transcribers={"zh": StaticTranscriber("你好世界")},
    )

    report = pipeline.evaluate([sample])

    assert set(report.results) == {"vc_cer"}
    assert report.results["vc_cer"].score == 0.0
    assert (
        report.results["vc_cer"].details["pipeline_id"]
        == "vc.zh.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1."
        "punctuation_strip_norm_v1.wenet_cer_v1"
    )
    assert report.rows[0]["semantic"]["metric"] == "vc_cer"
    assert report.rows[0]["semantic"]["asr_metric"] == "cer"
    assert report.rows[0]["semantic"]["normalizer"] == "punctuation_strip"


def test_vc_metric_pipeline_forwards_explicit_semantic_normalizer(monkeypatch) -> None:
    from sure_eval.evaluation.core.types import PipelineNodeResult
    from sure_eval.evaluation.nodes.transcription import StaticTranscriber
    from sure_eval.evaluation.tasks.asr import pipeline as asr_pipeline
    from sure_eval.evaluation.tasks.vc.compat import VCMetricPipeline, VCSample

    def fake_wetext(files, *, profile: str):
        return (
            files,
            PipelineNodeResult(
                stage="normalization",
                node_id="normalization/wetext_norm",
                version="v1",
                details={"profile": profile},
                internal_stages=("fake_wetext",),
            ),
        )

    monkeypatch.setattr(asr_pipeline, "normalize_wetext_key_text_files", fake_wetext)
    sample = VCSample(
        converted_audio="converted.wav",
        reference_audio="target.wav",
        reference_text="你好世界",
        language="zh",
    )
    pipeline = VCMetricPipeline(
        semantic_transcribers={"zh": StaticTranscriber("你好世界")},
        semantic_normalizer="wetext:zh_tn",
    )

    report = pipeline.evaluate([sample])

    assert report.results["vc_cer"].details["pipeline_id"] == (
        "vc.zh.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1.wetext_norm_zh_tn_v1.wenet_cer_v1"
    )
    assert report.results["vc_cer"].details["pipeline_trace"][2]["node_id"] == "normalization/wetext_norm"
    assert report.results["vc_cer"].details["pipeline_trace"][2]["profile"] == "zh_tn"


def test_vc_metric_pipeline_scores_semantic_between_converted_and_reference_audio() -> None:
    from sure_eval.evaluation.tasks.vc.compat import VCMetricPipeline, VCSample

    sample = VCSample(
        converted_audio="converted_same.wav",
        reference_audio="target_same.wav",
        language="zh",
    )
    pipeline = VCMetricPipeline(
        semantic_transcribers={"zh": PathEchoTranscriber()},
    )

    report = pipeline.evaluate([sample])

    assert set(report.results) == {"vc_cer"}
    assert report.results["vc_cer"].score == 0.0
    assert (
        report.results["vc_cer"].details["pipeline_id"]
        == "vc.zh.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1."
        "funasr_loader_16k_mono_v1.paraformer_zh_v1.punctuation_strip_norm_v1.wenet_cer_v1"
    )
    assert report.rows[0]["semantic"]["reference_audio_transcript"] == "你好世界"
    assert report.rows[0]["semantic"]["asr_metric"] == "cer"
    assert report.rows[0]["semantic"]["normalizer"] == "punctuation_strip"


def test_build_default_vc_metric_pipeline_wires_expected_backends() -> None:
    from sure_eval.evaluation.tasks.vc.compat import build_default_vc_metric_pipeline

    pipeline = build_default_vc_metric_pipeline(device="cpu", cache_dir="/tmp/sure-eval-vc")

    assert set(pipeline.semantic_transcribers) == {"en", "zh"}
    assert set(pipeline.speaker_providers) == {"wavlm-large", "ecapa-tdnn", "eres2net"}
    assert set(pipeline.mos_providers) == {"dnsmos", "wv-mos", "utmos"}


def test_vc_pipeline_cli_runs_with_stub_backend(tmp_path) -> None:
    output = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_vc_metric_pipeline.py",
            "--converted-audio",
            "converted.wav",
            "--reference-audio",
            "target.wav",
            "--reference-text",
            "你好世界",
            "--language",
            "zh",
            "--stub",
            "--output",
            str(output),
        ],
        check=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["sample"]["converted_audio"] == "converted.wav"
    assert report["sample"]["reference_audio"] == "target.wav"
    assert report["metrics"]["vc_cer"]["score"] == 0.0
    assert report["metrics"]["sim"]["score"] == 0.42
    assert report["metrics"]["dnsmos"]["score"] == 3.0


def test_vc_pipeline_cli_runs_semantic_without_reference_text(tmp_path) -> None:
    output = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_vc_metric_pipeline.py",
            "--converted-audio",
            "converted.wav",
            "--reference-audio",
            "target.wav",
            "--language",
            "zh",
            "--stub",
            "--speaker-backends",
            "",
            "--mos-backends",
            "",
            "--output",
            str(output),
        ],
        check=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["metrics"]["vc_cer"]["score"] == 0.0
    assert report["rows"][0]["semantic"]["reference_audio_transcript"] == "你好世界"


def test_vc_pipeline_cli_runner_forwards_semantic_normalizer(monkeypatch) -> None:
    import importlib.util
    from pathlib import Path

    from sure_eval.evaluation.core.types import PipelineNodeResult
    from sure_eval.evaluation.nodes.transcription import StaticTranscriber
    from sure_eval.evaluation.tasks.asr import pipeline as asr_pipeline
    from sure_eval.evaluation.tasks.vc.compat import VCMetricPipeline, VCSample

    spec = importlib.util.spec_from_file_location(
        "run_vc_metric_pipeline",
        Path("scripts/run_vc_metric_pipeline.py"),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    def fake_wetext(files, *, profile: str):
        return (
            files,
            PipelineNodeResult(
                stage="normalization",
                node_id="normalization/wetext_norm",
                version="v1",
                details={"profile": profile},
                internal_stages=("fake_wetext",),
            ),
        )

    monkeypatch.setattr(asr_pipeline, "normalize_wetext_key_text_files", fake_wetext)
    sample = VCSample(
        converted_audio="converted.wav",
        reference_audio="target.wav",
        reference_text="你好世界",
        language="zh",
    )
    pipeline = VCMetricPipeline(semantic_transcribers={"zh": StaticTranscriber("你好世界")})

    report = module._run_one(
        pipeline,
        sample,
        fail_fast=False,
        semantic_normalizer="wetext:zh_tn",
    )

    assert report["ok"] is True
    assert report["metrics"]["vc_cer"]["details"]["pipeline_id"] == (
        "vc.zh.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1.wetext_norm_zh_tn_v1.wenet_cer_v1"
    )


def test_vc_pipeline_docker_wrapper_plans_required_segments() -> None:
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "run_vc_metric_pipeline_docker",
        Path("scripts/run_vc_metric_pipeline_docker.py"),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    args = module.parse_args(
        [
            "--converted-audio",
            "/tmp/converted.wav",
            "--reference-audio",
            "/tmp/target.wav",
            "--reference-text",
            "hello world",
            "--language",
            "en",
            "--output",
            "/tmp/out.json",
        ]
    )
    segments = module.build_segments(args)

    assert [segment.name for segment in segments] == [
        "semantic",
        "speaker_wavlm_ecapa",
        "speaker_eres2net",
        "mos_dnsmos_wvmos",
        "mos_utmos",
    ]
    semantic = segments[0]
    assert semantic.speaker_backends == ""
    assert semantic.mos_backends == ""
    eres2net = next(segment for segment in segments if segment.name == "speaker_eres2net")
    assert any("libsox.so" in mount for mount in eres2net.extra_mounts)


def test_vc_pipeline_docker_wrapper_passes_semantic_normalizer_only_to_semantic_segment() -> None:
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "run_vc_metric_pipeline_docker",
        Path("scripts/run_vc_metric_pipeline_docker.py"),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    args = module.parse_args(
        [
            "--converted-audio",
            "/tmp/converted.wav",
            "--reference-audio",
            "/tmp/target.wav",
            "--reference-text",
            "你好世界",
            "--language",
            "zh",
            "--semantic-normalizer",
            "wetext:zh_tn",
            "--output",
            "/tmp/out.json",
        ]
    )
    segments = module.build_segments(args)
    semantic_command = module._segment_command(args, segments[0], args.work_dir / segments[0].output_name)
    speaker_command = module._segment_command(args, segments[1], args.work_dir / segments[1].output_name)

    assert "--semantic-normalizer" in semantic_command
    assert semantic_command[semantic_command.index("--semantic-normalizer") + 1] == "wetext:zh_tn"
    assert "--semantic-normalizer" not in speaker_command


def test_vc_pipeline_docker_wrapper_runs_semantic_without_reference_text() -> None:
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "run_vc_metric_pipeline_docker",
        Path("scripts/run_vc_metric_pipeline_docker.py"),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    args = module.parse_args(
        [
            "--converted-audio",
            "/tmp/converted.wav",
            "--reference-audio",
            "/tmp/target.wav",
            "--output",
            "/tmp/out.json",
        ]
    )
    segments = module.build_segments(args)

    assert [segment.name for segment in segments] == [
        "semantic",
        "speaker_wavlm_ecapa",
        "speaker_eres2net",
        "mos_dnsmos_wvmos",
        "mos_utmos",
    ]


def test_vc_docker_shell_accepts_semantic_normalizer_argument() -> None:
    completed = subprocess.run(
        [
            "bash",
            "scripts/run_vc_metric_pipeline_docker.sh",
            "--converted-audio",
            "converted.wav",
            "--reference-audio",
            "target.wav",
            "--output",
            "/tmp/out.json",
            "--semantic-normalizer",
            "wetext:zh_tn",
            "--help",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert "Unknown argument: --semantic-normalizer" not in completed.stderr
