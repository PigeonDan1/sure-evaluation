from __future__ import annotations

from pathlib import Path


def test_tse_describe_si_sdr() -> None:
    from sure_eval.evaluation.scripts.tse import describe_pipeline

    desc = describe_pipeline(language="zh", metrics="si_sdr")
    assert desc.task == "TSE"
    assert desc.language == "zh"
    assert desc.metric == "si_sdr"
    assert desc.pipeline_id == "tse.zh.si_sdr.si_sdr_v1"
    assert desc.node_ids == ("scoring/si_sdr",)


def test_tse_describe_sim() -> None:
    from sure_eval.evaluation.scripts.tse import describe_pipeline

    desc = describe_pipeline(language="zh", metrics="sim/wavlm-large")
    assert desc.task == "TSE"
    assert desc.metric == "spk_sim"
    assert desc.execution_metrics == ("sim/wavlm-large",)
    assert desc.pipeline_id == "tse.zh.spk_sim.wavlm_large_sim_v1"
    assert desc.node_ids == ("scoring/wavlm_large_sim",)


def test_tse_describe_mos() -> None:
    from sure_eval.evaluation.scripts.tse import describe_pipeline

    desc = describe_pipeline(language="en", metrics="dnsmos")
    assert desc.task == "TSE"
    assert desc.metric == "dnsmos"
    assert desc.pipeline_id == "tse.en.dnsmos.dnsmos_v1"
    assert desc.execution_metrics == ("dnsmos",)
    assert desc.node_ids == ("scoring/dnsmos",)


def test_tse_describe_semantic_zh() -> None:
    from sure_eval.evaluation.scripts.tse import describe_pipeline

    desc = describe_pipeline(language="zh", metrics="tse_cer")
    assert desc.task == "TSE"
    assert desc.metric == "cer"
    assert desc.execution_metrics == ("tse_cer",)
    assert desc.language == "zh"
    assert "frontend/funasr_loader_16k_mono" in desc.node_ids
    assert "transcription/paraformer_zh" in desc.node_ids
    assert "normalization/punctuation_strip_norm" in desc.node_ids
    assert "scoring/wenet_cer" in desc.node_ids


def test_tse_describe_semantic_en() -> None:
    from sure_eval.evaluation.scripts.tse import describe_pipeline

    desc = describe_pipeline(language="en", metrics="tse_wer")
    assert desc.task == "TSE"
    assert desc.metric == "wer"
    assert desc.execution_metrics == ("tse_wer",)
    assert desc.language == "en"
    assert "transcription/whisper_large_v3" in desc.node_ids
    assert "normalization/whisper_norm" in desc.node_ids
    assert "scoring/wenet_wer" in desc.node_ids


def test_tse_describe_multi_metric() -> None:
    from sure_eval.evaluation.scripts.tse import describe_pipeline

    desc = describe_pipeline(language="zh", metrics=["si_sdr", "sim/wavlm-large", "dnsmos"])
    assert desc.metric == "multi"
    assert desc.pipeline_id == (
        "tse.zh.multi.si_sdr.si_sdr_v1__spk_sim.wavlm_large_sim_v1__dnsmos.dnsmos_v1"
    )
    assert desc.pipeline_kind == "bundle"
    assert desc.member_pipeline_ids == (
        "tse.zh.si_sdr.si_sdr_v1",
        "tse.zh.spk_sim.wavlm_large_sim_v1",
        "tse.zh.dnsmos.dnsmos_v1",
    )
    assert desc.execution_metrics == ("si_sdr", "sim/wavlm-large", "dnsmos")
    assert "scoring/si_sdr" in desc.node_ids
    assert "scoring/wavlm_large_sim" in desc.node_ids
    assert "scoring/dnsmos" in desc.node_ids


def test_tse_describe_default_metric() -> None:
    from sure_eval.evaluation.scripts.tse import describe_pipeline

    desc = describe_pipeline(language="zh")
    assert desc.metric == "si_sdr"
    assert desc.pipeline_id == "tse.zh.si_sdr.si_sdr_v1"


def test_tse_describe_required_roles() -> None:
    from sure_eval.evaluation.scripts.tse import describe_pipeline

    desc = describe_pipeline(language="zh", metrics="si_sdr")
    assert "prediction_audio" in desc.required_roles
    assert "reference_audio" in desc.required_roles


def test_tse_run_through_scripts_dispatch(tmp_path: Path) -> None:
    import numpy as np
    import soundfile as sf

    from sure_eval.evaluation.scripts import run_task
    from sure_eval.evaluation.tasks.tse.types import TSESample

    rng = np.random.RandomState(42)
    ref = rng.randn(8000)
    pred = ref.copy()

    pred_path = tmp_path / "pred.wav"
    ref_path = tmp_path / "ref.wav"
    sf.write(str(pred_path), pred.astype("float32"), 16000)
    sf.write(str(ref_path), ref.astype("float32"), 16000)

    output_dir = tmp_path / "tse_out"
    report = run_task(
        "tse",
        samples=[
            TSESample(
                prediction_audio=str(pred_path),
                reference_audio=str(ref_path),
                language="en",
                sample_id="utt1",
            )
        ],
        metrics=("si_sdr",),
        output_dir=str(output_dir),
    )

    assert report.task == "TSE"
    assert report.metric == "si_sdr"
    assert (output_dir / "report.json").exists()
    assert (output_dir / "pipeline_description.json").exists()
