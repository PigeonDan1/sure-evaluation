from __future__ import annotations

from pathlib import Path


def _write_key_text(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"{key}\t{text}\n" for key, text in rows), encoding="utf-8")


def test_asr_metric_modules_match_sure_evaluator(tmp_path: Path) -> None:
    from sure_eval.evaluation.registry import MetricRegistry
    from sure_eval.evaluation.sure_evaluator import SUREEvaluator
    from sure_eval.evaluation.tasks.asr.metrics import CERMetric, WERMetric

    ref_zh = tmp_path / "ref_zh.txt"
    hyp_zh = tmp_path / "hyp_zh.txt"
    _write_key_text(ref_zh, [("utt1", "你好世界")])
    _write_key_text(hyp_zh, [("utt1", "你好世")])
    evaluator_cer = SUREEvaluator(language="zh").evaluate("ASR", str(ref_zh), str(hyp_zh), tochar=True)
    registry_cer = MetricRegistry.get_metric("cer").calculate("你好世", "你好世界", language="zh")
    task_cer = CERMetric().calculate("你好世", "你好世界", language="zh")
    assert task_cer == registry_cer
    assert task_cer.score == evaluator_cer["score"]
    assert task_cer.details["pipeline_id"] == "asr.zh.cer.aispeech_norm.wenet_cer"
    assert task_cer.details["input_contract"]["required_roles"] == ["hyp", "ref"]
    assert task_cer.details["input_roles"] == ["ref", "hyp"]
    assert task_cer.details["pipeline_trace"][0]["node_id"] == "normalization/aispeech_norm"
    assert task_cer.details["pipeline_trace"][0]["profile"] == "zh"

    ref_en = tmp_path / "ref_en.txt"
    hyp_en = tmp_path / "hyp_en.txt"
    _write_key_text(ref_en, [("utt1", "hello world")])
    _write_key_text(hyp_en, [("utt1", "hello brave world")])
    registry_wer = MetricRegistry.get_metric("wer").calculate("hello brave world", "hello world")
    task_wer = WERMetric().calculate("hello brave world", "hello world")
    assert task_wer == registry_wer
    assert task_wer.score > 0
    assert task_wer.details["pipeline_id"] == "asr.en.wer.whisper_norm.wenet_wer"
    assert task_wer.details["input_contract"]["required_roles"] == ["hyp", "ref"]
    assert task_wer.details["input_roles"] == ["ref", "hyp"]
    assert task_wer.details["pipeline_trace"][0]["node_id"] == "normalization/whisper_norm"
    assert task_wer.details["pipeline_trace"][0]["profile"] == "english"


def test_classification_metric_module_matches_legacy_metrics() -> None:
    from sure_eval.evaluation.registry import MetricRegistry
    from sure_eval.evaluation.tasks.classification.metrics import AccuracyMetric

    registry = MetricRegistry.get_metric("accuracy").calculate_batch(["happy", "female"], ["hap", "woman"])
    task = AccuracyMetric().calculate_batch(["happy", "female"], ["hap", "woman"])
    assert task == registry


def test_s2tt_metric_module_is_registry_metric() -> None:
    from sure_eval.evaluation.registry import MetricRegistry
    from sure_eval.evaluation.nodes.scoring.bleurt_20.node import SegmentScore as BLEURTSegmentScore
    from sure_eval.evaluation.nodes.scoring.xcomet_xl.node import SegmentScore as XCOMETSegmentScore
    from sure_eval.evaluation.tasks.s2tt.metrics import BLEUMetric, BLEURT20Metric, XCOMETXLMetric

    assert isinstance(MetricRegistry.get_metric("bleu"), BLEUMetric)
    result = MetricRegistry.get_metric("bleu").calculate_batch(["你好世界"], ["你好世界"], language="zh")
    assert result.details["pipeline_id"] == "s2tt.zh.bleu.sacrebleu"
    assert result.details["input_contract"]["required_roles"] == ["hyp", "ref"]
    assert result.details["input_roles"] == ["ref", "hyp"]
    assert result.details["pipeline_trace"][0]["node_id"] == "scoring/sacrebleu"
    assert result.details["pipeline_trace"][0]["tokenizer_profile"] == "zh"

    assert isinstance(MetricRegistry.get_metric("xcomet_xl"), XCOMETXLMetric)
    xcomet = MetricRegistry.get_metric("xcomet_xl").calculate_batch(
        ["你好世界"],
        ["你好世界"],
        sources=["hello world"],
        xcomet_runner=lambda rows: [XCOMETSegmentScore(key=rows[0]["key"], score=0.9)],
    )
    assert xcomet.score == 0.9
    assert xcomet.details["input_contract"]["required_roles"] == ["src", "hyp", "ref"]

    assert isinstance(MetricRegistry.get_metric("bleurt_20"), BLEURT20Metric)
    bleurt = MetricRegistry.get_metric("bleurt_20").calculate_batch(
        ["你好世界"],
        ["你好世界"],
        bleurt_runner=lambda rows: [BLEURTSegmentScore(key=rows[0]["key"], score=0.8)],
    )
    assert bleurt.score == 0.8
    assert bleurt.details["input_contract"]["required_roles"] == ["hyp", "ref"]


def test_tts_metric_definitions_are_registry_metrics() -> None:
    from sure_eval.evaluation.registry import MetricRegistry
    from sure_eval.evaluation.tasks.tts.metrics import (
        CERMetric,
        DNSMOSMetric,
        SIMMetric,
        UTMOSMetric,
        WERMetric,
        WVMOSMetric,
    )

    expected = {
        "tts_cer": CERMetric,
        "tts_wer": WERMetric,
        "sim": SIMMetric,
        "dnsmos": DNSMOSMetric,
        "wv-mos": WVMOSMetric,
        "utmos": UTMOSMetric,
    }
    for name, metric_type in expected.items():
        assert isinstance(MetricRegistry.get_metric(name), metric_type)


def test_vc_metric_definitions_are_registry_metrics() -> None:
    from sure_eval.evaluation.registry import MetricRegistry
    from sure_eval.evaluation.tasks.vc.metrics import CERMetric, WERMetric

    assert isinstance(MetricRegistry.get_metric("vc_cer"), CERMetric)
    assert isinstance(MetricRegistry.get_metric("vc_wer"), WERMetric)


def test_tts_audio_metrics_use_injected_score_provider() -> None:
    from sure_eval.evaluation.tasks.tts.metrics import (
        DNSMOSMetric,
        SIMMetric,
        UTMOSMetric,
        WVMOSMetric,
    )

    def sim_provider(prediction: str, reference: str, **kwargs) -> dict[str, float]:
        assert prediction.endswith("hyp.wav")
        assert reference.endswith("ref.wav")
        assert kwargs["prompt_audio_path"].endswith("prompt.wav")
        return {"score": 0.81, "ref_score": 0.77}

    sim = SIMMetric(score_provider=sim_provider).calculate(
        "hyp.wav",
        "ref.wav",
        prompt_audio_path="prompt.wav",
    )
    assert sim.metric_name == "sim"
    assert sim.score == 0.81
    assert sim.details["mean_ref_similarity"] == 0.77
    assert sim.details["score_key"] == "ASV"
    assert sim.details["source"]["primary_reference"] == "BytedanceSpeech/seed-tts-eval"

    dnsmos = DNSMOSMetric(
        score_provider=lambda prediction, reference, **kwargs: {"OVRL": 3.4, "SIG": 3.8}
    )
    assert dnsmos.calculate("hyp.wav", "").score == 3.4

    wv_mos = WVMOSMetric(score_provider=lambda prediction, reference, **kwargs: 4.1)
    assert wv_mos.calculate("hyp.wav", "").details["score_key"] == "mos"

    utmos = UTMOSMetric(score_provider=lambda prediction, reference, **kwargs: {"utmos": 4.2})
    assert utmos.calculate("hyp.wav", "").score == 4.2


def test_tts_metric_sources_document_reference_implementations() -> None:
    from sure_eval.evaluation.tasks.tts.metrics import (
        CERMetric,
        DNSMOSMetric,
        SIMMetric,
        UTMOSMetric,
        WERMetric,
        WVMOSMetric,
    )

    assert WERMetric.source.primary_reference == "BytedanceSpeech/seed-tts-eval"
    assert WERMetric.source.score_key == "wer"
    assert "Whisper-large-v3" in WERMetric.source.method
    assert "Paraformer-zh" in CERMetric.source.method
    assert SIMMetric.source.score_key == "ASV"
    assert DNSMOSMetric.source.score_key == "OVRL"
    assert WVMOSMetric.source.primary_reference == "boson-ai/emergenttts-eval-public"
    assert UTMOSMetric.source.primary_reference == "sarulab-speech/UTMOS22"


def test_tts_audio_metrics_normalize_reference_runner_outputs() -> None:
    from sure_eval.evaluation.tasks.tts.metrics import DNSMOSMetric, SIMMetric, UTMOSMetric, WVMOSMetric

    sim = SIMMetric(
        score_provider=lambda prediction, reference, **kwargs: {
            "score": 0.72,
            "score_var": 0.03,
        }
    ).calculate_batch(["hyp1.wav", "hyp2.wav"], ["ref1.wav", "ref2.wav"])
    assert sim.score == 0.72
    assert sim.details["mean_ASV_var"] == 0.03

    dnsmos = DNSMOSMetric(
        score_provider=lambda prediction, reference, **kwargs: {
            "ovrl": 3.2,
            "sig": 3.6,
            "bak": 3.1,
            "p808_mos": 3.4,
        }
    ).calculate("hyp.wav", "")
    assert dnsmos.score == 3.2
    assert dnsmos.details["mean_SIG"] == 3.6
    assert dnsmos.details["mean_BAK"] == 3.1
    assert dnsmos.details["mean_P808_MOS"] == 3.4

    wv_mos = WVMOSMetric(score_provider=lambda prediction, reference, **kwargs: {"MOS": 4.05})
    assert wv_mos.calculate("hyp.wav", "").score == 4.05

    utmos = UTMOSMetric(score_provider=lambda prediction, reference, **kwargs: {"predicted_mos": 4.4})
    assert utmos.calculate("hyp.wav", "").score == 4.4


def test_vc_reuses_tts_audio_metric_definitions() -> None:
    from sure_eval.evaluation.tasks.vc.metrics import CERMetric, DNSMOSMetric, SIMMetric, WERMetric

    sim = SIMMetric(score_provider=lambda prediction, reference, **kwargs: 0.5)
    assert (
        sim.calculate("converted.wav", "source.wav", prompt_audio_path="target.wav").score
        == 0.5
    )
    metric = DNSMOSMetric(score_provider=lambda prediction, reference, **kwargs: {"OVRL": 3.0})
    assert metric.calculate(
        "converted.wav",
        "",
    ).score == 3.0

    assert WERMetric().metric_name == "vc_wer"
    assert CERMetric().metric_name == "vc_cer"


def test_vc_audio_metrics_filter_runtime_and_path_provider_fields() -> None:
    from sure_eval.evaluation.tasks.vc.metrics import SIMMetric

    metric = SIMMetric(
        score_provider=lambda prediction, reference, **kwargs: {
            "ASV": 0.91,
            "rtf": 0.12,
            "prediction_audio": prediction,
            "reference_audio": reference,
            "audio_path": prediction,
            "duration_seconds": 1.5,
        }
    ).calculate("converted.wav", "target.wav")

    assert metric.score == 0.91
    assert "rtf" not in metric.details
    assert "prediction_audio" not in metric.details
    assert "reference_audio" not in metric.details
    assert "audio_path" not in metric.details
    assert "duration_seconds" not in metric.details
