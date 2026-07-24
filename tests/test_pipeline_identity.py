from __future__ import annotations

import re
from pathlib import Path

import yaml

from sure_eval.evaluation.scripts import describe_pipeline
from sure_eval.evaluation.scripts.contracts import route_pipeline_id


TASKS_ROOT = Path("src/sure_eval/evaluation/tasks")
PIPELINE_ID_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+){3,}$")


def _load_routes(task: str) -> dict:
    return yaml.safe_load((TASKS_ROOT / task / "routes.yaml").read_text(encoding="utf-8"))


def test_route_pipeline_ids_use_atomic_identity_tokens() -> None:
    forbidden = (
        "cer_canonical",
        "wer_canonical",
        "mer_canonical",
        "audio_metric_nodes",
        "enhancement_quality_nodes",
        "kws.default",
        ".tts_cer.",
        ".tts_wer.",
        ".vc_cer.",
        ".vc_wer.",
        ".tse_cer.",
        ".tse_wer.",
        "macro-recall",
    )

    for routes_path in TASKS_ROOT.glob("*/routes.yaml"):
        routes = yaml.safe_load(routes_path.read_text(encoding="utf-8"))
        for route in routes.get("routes") or ():
            pipeline_id = route_pipeline_id(route, language=route.get("language") or "zh")
            assert PIPELINE_ID_RE.match(pipeline_id), pipeline_id
            assert "-" not in pipeline_id
            for token in forbidden:
                assert token not in pipeline_id, pipeline_id


def test_route_pipeline_ids_are_unique() -> None:
    for routes_path in TASKS_ROOT.glob("*/routes.yaml"):
        routes = yaml.safe_load(routes_path.read_text(encoding="utf-8"))
        pipeline_ids = [
            route_pipeline_id(route, language=route.get("language") or "zh")
            for route in routes.get("routes") or ()
        ]
        assert len(pipeline_ids) == len(set(pipeline_ids)), routes_path


def test_kws_input_modes_encode_input_conversion_in_pipeline_id() -> None:
    descriptions = [
        describe_pipeline("kws", metric="macro-recall", input_mode=input_mode)
        for input_mode in ("sure_json", "wekws_score_ctc", "wekws_frame_score")
    ]

    assert {desc.pipeline_id for desc in descriptions} == {
        "kws.any.macro_recall.conversion_kws_sure_json_to_samples_v1.wekws_det_v1",
        "kws.any.macro_recall.conversion_kws_wekws_score_ctc_to_samples_v1.wekws_det_v1",
        "kws.any.macro_recall.conversion_kws_wekws_frame_score_to_samples_v1.wekws_det_v1",
    }
    assert {desc.pipeline_kind for desc in descriptions} == {"atomic"}


def test_audio_multi_descriptions_are_bundles_with_atomic_members() -> None:
    cases = [
        (
            "tts",
            {"language": "zh", "metrics": ("tts_cer", "sim/wavlm-large", "dnsmos")},
            (
                "tts.zh.multi.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1."
                "punctuation_strip_norm_v1.wenet_cer_v1__spk_sim.wavlm_large_sim_v1__"
                "dnsmos.dnsmos_v1"
            ),
            (
                "tts.zh.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1.punctuation_strip_norm_v1.wenet_cer_v1",
                "tts.zh.spk_sim.wavlm_large_sim_v1",
                "tts.zh.dnsmos.dnsmos_v1",
            ),
        ),
        (
            "vc",
            {"language": "zh", "metrics": ("vc_cer", "sim/ecapa-tdnn", "utmos")},
            (
                "vc.zh.multi.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1."
                "punctuation_strip_norm_v1.wenet_cer_v1__spk_sim.ecapa_tdnn_sim_v1__"
                "utmos.utmos_v1"
            ),
            (
                "vc.zh.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1.punctuation_strip_norm_v1.wenet_cer_v1",
                "vc.zh.spk_sim.ecapa_tdnn_sim_v1",
                "vc.zh.utmos.utmos_v1",
            ),
        ),
        (
            "se",
            {"metrics": ("si-sdr", "stoi")},
            "se.any.multi.si_sdr.si_sdr_v1__stoi.stoi_v1",
            ("se.any.si_sdr.si_sdr_v1", "se.any.stoi.stoi_v1"),
        ),
        (
            "tse",
            {"language": "en", "metrics": ("si_sdr", "sim/wavlm-large", "dnsmos")},
            "tse.en.multi.si_sdr.si_sdr_v1__spk_sim.wavlm_large_sim_v1__dnsmos.dnsmos_v1",
            (
                "tse.en.si_sdr.si_sdr_v1",
                "tse.en.spk_sim.wavlm_large_sim_v1",
                "tse.en.dnsmos.dnsmos_v1",
            ),
        ),
    ]

    for task, kwargs, pipeline_id, members in cases:
        desc = describe_pipeline(task, **kwargs)
        assert desc.pipeline_id == pipeline_id
        assert desc.pipeline_kind == "bundle"
        assert desc.metric == "multi"
        assert desc.member_pipeline_ids == members
        assert len(desc.execution_metrics) == len(members)


def test_sa_asr_conversion_is_part_of_computation_identity() -> None:
    desc = describe_pipeline("sa_asr", metric="cpwer")

    assert desc.pipeline_id == (
        "sa_asr.en.cpwer.conversion_sa_asr_cpwer_v1.gstar_norm_v1.meeteval_v1"
    )
    assert desc.computation_node_ids == (
        "conversion/sa_asr__cpwer",
        "normalization/gstar_norm",
        "scoring/meeteval",
    )
    assert desc.conversion_steps[0]["id"] == "sa_asr__cpwer"
