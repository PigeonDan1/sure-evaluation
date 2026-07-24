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


def test_tts_zh_semantic_route_uses_punctuation_strip_norm() -> None:
    from sure_eval.evaluation.tasks.tts.pipeline import evaluate_tts_samples
    from sure_eval.evaluation.tasks.tts.compat import TTSSample

    transcriber = RecordingTranscriber({"hyp.wav": "你好世界"})
    report = evaluate_tts_samples(
        [
            TTSSample(
                prediction_audio="hyp.wav",
                reference_text="你好世界",
                reference_audio="ref.wav",
                language="zh",
                sample_id="utt1",
            )
        ],
        metrics=("tts_cer",),
        transcribers={"zh": transcriber},
    )

    assert report.task == "TTS"
    assert report.language == "zh"
    assert report.metric == "cer"
    assert report.score == 0.0
    assert report.pipeline_id == (
        "tts.zh.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1.punctuation_strip_norm_v1.wenet_cer_v1"
    )
    assert report.details["results"]["cer"]["score"] == 0.0
    assert (
        report.details["results"]["cer"]["asr_pipeline_id"]
        == "asr.zh.cer.punctuation_strip_norm_v1.wenet_cer_v1"
    )
    assert report.details["rows"][0]["semantic"]["metric"] == "cer"
    assert report.details["rows"][0]["semantic"]["execution_metric"] == "tts_cer"
    assert report.details["rows"][0]["semantic"]["transcript"] == "你好世界"
    assert report.details["rows"][0]["semantic"]["asr_metric"] == "cer"
    assert report.details["rows"][0]["semantic"]["normalizer"] == "punctuation_strip"
    assert transcriber.calls == [("hyp.wav", "zh")]

    trace_ids = [node.node_id for node in report.pipeline_trace]
    assert trace_ids == [
        "frontend/funasr_loader_16k_mono",
        "transcription/paraformer_zh",
        "normalization/punctuation_strip_norm",
        "scoring/wenet_cer",
    ]
    assert "normalization/aispeech_norm" not in trace_ids
    assert report.pipeline_trace[0].details["audio_path"] == "hyp.wav"
    assert report.pipeline_trace[0].details["materialized_audio_path"] is None
    assert report.pipeline_trace[0].details["cv3_compatible"] is True
    assert report.input_contract is not None
    assert report.input_contract.required_roles == ("prediction_audio", "reference_text")
    assert report.details["input_files"] == {
        "prediction_audio": "hyp.wav",
        "reference_text": "inline",
        "reference_audio": "ref.wav",
    }


def test_tts_en_semantic_route_reuses_asr_wer_pipeline() -> None:
    from sure_eval.evaluation.tasks.tts.pipeline import evaluate_tts_samples
    from sure_eval.evaluation.tasks.tts.compat import TTSSample

    transcriber = RecordingTranscriber({"hyp.wav": "hello brave world"})
    report = evaluate_tts_samples(
        [
            TTSSample(
                prediction_audio="hyp.wav",
                reference_text="hello world",
                language="en",
                sample_id="utt1",
            )
        ],
        metrics=("tts_wer",),
        transcribers={"en": transcriber},
    )

    assert report.task == "TTS"
    assert report.language == "en"
    assert report.metric == "wer"
    assert report.pipeline_id == "tts.en.wer.whisper_large_v3_v1.whisper_norm_english_v1.wenet_wer_v1"
    assert report.score > 0
    assert report.details["results"]["wer"]["score"] == report.score
    assert [node.node_id for node in report.pipeline_trace] == [
        "transcription/whisper_large_v3",
        "normalization/whisper_norm",
        "scoring/wenet_wer",
    ]
    assert report.details["scoring_result"] == report.details["results"]["wer"]["asr_result"]


def test_tts_semantic_route_can_explicitly_use_wetext_normalizer(monkeypatch) -> None:
    from sure_eval.evaluation.tasks.asr import pipeline as asr_pipeline
    from sure_eval.evaluation.tasks.tts.pipeline import evaluate_tts_samples
    from sure_eval.evaluation.tasks.tts.compat import TTSSample

    monkeypatch.setattr(asr_pipeline, "normalize_wetext_key_text_files", _fake_wetext_normalizer)
    transcriber = RecordingTranscriber({"hyp.wav": "你好世界"})
    report = evaluate_tts_samples(
        [
            TTSSample(
                prediction_audio="hyp.wav",
                reference_text="你好世界",
                language="zh",
                sample_id="utt1",
            )
        ],
        metrics=("tts_cer",),
        semantic_normalizer="wetext:zh_tn",
        transcribers={"zh": transcriber},
    )

    assert report.pipeline_id == "tts.zh.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1.wetext_norm_zh_tn_v1.wenet_cer_v1"
    assert report.details["results"]["cer"]["asr_pipeline_id"] == "asr.zh.cer.wetext_norm_zh_tn_v1.wenet_cer_v1"
    assert [node.node_id for node in report.pipeline_trace] == [
        "frontend/funasr_loader_16k_mono",
        "transcription/paraformer_zh",
        "normalization/wetext_norm",
        "scoring/wenet_cer",
    ]
    assert report.pipeline_trace[2].details["profile"] == "zh_tn"


def test_tts_task_route_scores_speaker_and_mos_nodes() -> None:
    from sure_eval.evaluation.tasks.tts.pipeline import evaluate_tts_samples
    from sure_eval.evaluation.tasks.tts.compat import TTSSample

    report = evaluate_tts_samples(
        [
            TTSSample(
                prediction_audio="hyp.wav",
                reference_text="hello",
                reference_audio="ref.wav",
                language="en",
                sample_id="utt1",
            )
        ],
        metrics=("sim/wavlm-large", "dnsmos"),
        speaker_providers={"wavlm-large": lambda prediction, reference, **kwargs: {"ASV": 0.7}},
        mos_providers={"dnsmos": lambda prediction, reference="", **kwargs: {"OVRL": 3.1}},
    )

    assert report.task == "TTS"
    assert report.metric == "multi"
    assert report.details["results"]["spk_sim"]["score"] == 0.7
    assert report.details["results"]["dnsmos"]["score"] == 3.1
    assert report.details["rows"][0]["speaker"]["wavlm-large"]["ASV"] == 0.7
    assert report.details["rows"][0]["mos"]["dnsmos"]["OVRL"] == 3.1
    assert [node.node_id for node in report.pipeline_trace] == [
        "scoring/wavlm_large_sim",
        "scoring/dnsmos",
    ]
