from __future__ import annotations


class RecordingTranscriber:
    def __init__(self, transcripts: dict[str, str]) -> None:
        self.transcripts = transcripts
        self.calls: list[tuple[str, str]] = []

    def transcribe(self, audio_path: str, *, language: str = "en") -> str:
        self.calls.append((audio_path, language))
        return self.transcripts[audio_path]


def _fake_wetext_normalizer(files, *, profile: str):
    from sure_eval.evaluation.core.types import PipelineNodeResult

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


def test_vc_semantic_route_reuses_asr_when_reference_text_is_available() -> None:
    from sure_eval.evaluation.tasks.vc.pipeline import evaluate_vc_samples
    from sure_eval.evaluation.tasks.vc.compat import VCSample

    transcriber = RecordingTranscriber({"converted.wav": "你好世界"})
    report = evaluate_vc_samples(
        [
            VCSample(
                converted_audio="converted.wav",
                reference_audio="target.wav",
                reference_text="你好世界",
                language="zh",
                sample_id="utt1",
            )
        ],
        metrics=("vc_cer",),
        transcribers={"zh": transcriber},
    )

    assert report.task == "VC"
    assert report.language == "zh"
    assert report.metric == "cer"
    assert report.score == 0.0
    assert report.pipeline_id == "vc.zh.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1.punctuation_strip_norm_v1.wenet_cer_v1"
    assert transcriber.calls == [("converted.wav", "zh")]
    assert report.details["rows"][0]["semantic"]["reference_audio_transcript"] == ""
    assert report.details["rows"][0]["semantic"]["asr_metric"] == "cer"
    assert [node.node_id for node in report.pipeline_trace] == [
        "frontend/funasr_loader_16k_mono",
        "transcription/paraformer_zh",
        "normalization/punctuation_strip_norm",
        "scoring/wenet_cer",
    ]
    assert report.details["rows"][0]["semantic"]["normalizer"] == "punctuation_strip"
    assert report.pipeline_trace[0].details["audio_path"] == "converted.wav"
    assert report.pipeline_trace[0].details["role"] == "converted_audio"
    assert report.pipeline_trace[0].details["materialized_audio_path"] is None
    assert report.input_contract is not None
    assert report.input_contract.required_roles == ("converted_audio", "reference_text")


def test_vc_semantic_route_transcribes_reference_audio_before_reusing_asr() -> None:
    from sure_eval.evaluation.tasks.vc.pipeline import evaluate_vc_samples
    from sure_eval.evaluation.tasks.vc.compat import VCSample

    transcriber = RecordingTranscriber(
        {
            "converted.wav": "hello world",
            "target.wav": "hello brave world",
        }
    )
    report = evaluate_vc_samples(
        [
            VCSample(
                converted_audio="converted.wav",
                reference_audio="target.wav",
                language="en",
                sample_id="utt1",
            )
        ],
        metrics=("vc_wer",),
        transcribers={"en": transcriber},
    )

    assert report.task == "VC"
    assert report.language == "en"
    assert report.metric == "wer"
    assert report.pipeline_id == (
        "vc.en.wer.whisper_large_v3_v1.whisper_large_v3_v1."
        "whisper_norm_english_v1.wenet_wer_v1"
    )
    assert report.score > 0
    assert transcriber.calls == [("converted.wav", "en"), ("target.wav", "en")]
    assert report.details["rows"][0]["semantic"]["reference_audio_transcript"] == "hello brave world"
    assert [node.node_id for node in report.pipeline_trace] == [
        "transcription/whisper_large_v3",
        "transcription/whisper_large_v3",
        "normalization/whisper_norm",
        "scoring/wenet_wer",
    ]
    assert report.input_contract is not None
    assert report.input_contract.required_roles == ("converted_audio", "reference_audio")


def test_vc_zh_audio_reference_route_records_funasr_loader_for_both_audios() -> None:
    from sure_eval.evaluation.tasks.vc.pipeline import evaluate_vc_samples
    from sure_eval.evaluation.tasks.vc.compat import VCSample

    transcriber = RecordingTranscriber(
        {
            "converted.wav": "你好世界",
            "target.wav": "你好世界",
        }
    )
    report = evaluate_vc_samples(
        [
            VCSample(
                converted_audio="converted.wav",
                reference_audio="target.wav",
                language="zh",
                sample_id="utt1",
            )
        ],
        metrics=("vc_cer",),
        transcribers={"zh": transcriber},
    )

    assert report.pipeline_id == (
        "vc.zh.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1."
        "funasr_loader_16k_mono_v1.paraformer_zh_v1.punctuation_strip_norm_v1.wenet_cer_v1"
    )
    assert transcriber.calls == [("converted.wav", "zh"), ("target.wav", "zh")]
    assert [node.node_id for node in report.pipeline_trace] == [
        "frontend/funasr_loader_16k_mono",
        "transcription/paraformer_zh",
        "frontend/funasr_loader_16k_mono",
        "transcription/paraformer_zh",
        "normalization/punctuation_strip_norm",
        "scoring/wenet_cer",
    ]
    assert report.details["rows"][0]["semantic"]["normalizer"] == "punctuation_strip"
    assert report.pipeline_trace[0].details["role"] == "converted_audio"
    assert report.pipeline_trace[2].details["role"] == "reference_audio"
    assert report.pipeline_trace[0].details["materialized_audio_path"] is None
    assert report.pipeline_trace[2].details["materialized_audio_path"] is None


def test_vc_semantic_route_can_explicitly_use_wetext_normalizer(monkeypatch) -> None:
    from sure_eval.evaluation.tasks.asr import pipeline as asr_pipeline
    from sure_eval.evaluation.tasks.vc.pipeline import evaluate_vc_samples
    from sure_eval.evaluation.tasks.vc.compat import VCSample

    monkeypatch.setattr(asr_pipeline, "normalize_wetext_key_text_files", _fake_wetext_normalizer)
    transcriber = RecordingTranscriber({"converted.wav": "你好世界"})
    report = evaluate_vc_samples(
        [
            VCSample(
                converted_audio="converted.wav",
                reference_audio="target.wav",
                reference_text="你好世界",
                language="zh",
                sample_id="utt1",
            )
        ],
        metrics=("vc_cer",),
        semantic_normalizer="wetext:zh_tn",
        transcribers={"zh": transcriber},
    )

    assert report.pipeline_id == "vc.zh.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1.wetext_norm_zh_tn_v1.wenet_cer_v1"
    assert report.details["results"]["cer"]["asr_pipeline_id"] == "asr.zh.cer.wetext_norm_zh_tn_v1.wenet_cer_v1"
    assert [node.node_id for node in report.pipeline_trace] == [
        "frontend/funasr_loader_16k_mono",
        "transcription/paraformer_zh",
        "normalization/wetext_norm",
        "scoring/wenet_cer",
    ]
    assert report.pipeline_trace[2].details["profile"] == "zh_tn"


def test_vc_task_route_scores_speaker_and_mos_nodes() -> None:
    from sure_eval.evaluation.tasks.vc.pipeline import evaluate_vc_samples
    from sure_eval.evaluation.tasks.vc.compat import VCSample

    report = evaluate_vc_samples(
        [
            VCSample(
                converted_audio="converted.wav",
                reference_audio="target.wav",
                source_audio="source.wav",
                language="en",
                sample_id="utt1",
            )
        ],
        metrics=("sim/ecapa-tdnn", "utmos"),
        speaker_providers={"ecapa-tdnn": lambda prediction, reference, **kwargs: {"ASV": 0.8}},
        mos_providers={"utmos": lambda prediction, reference="", **kwargs: {"utmos": 3.7}},
    )

    assert report.task == "VC"
    assert report.metric == "multi"
    assert report.details["results"]["spk_sim"]["score"] == 0.8
    assert report.details["results"]["utmos"]["score"] == 3.7
    assert report.details["rows"][0]["speaker"]["ecapa-tdnn"]["ASV"] == 0.8
    assert report.details["rows"][0]["mos"]["utmos"]["utmos"] == 3.7
    assert [node.node_id for node in report.pipeline_trace] == [
        "scoring/ecapa_tdnn_sim",
        "scoring/utmos",
    ]
