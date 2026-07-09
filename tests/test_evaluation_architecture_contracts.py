from __future__ import annotations

import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_aispeech_normalization_impl_lives_under_node_package() -> None:
    impl = importlib.import_module(
        "sure_eval.evaluation.nodes.normalization.aispeech_norm.normalization_impl.asr_simple_tn"
    )

    impl_path = Path(impl.__file__).resolve()
    assert "nodes/normalization/aispeech_norm/normalization_impl" in impl_path.as_posix()
    assert callable(impl.asr_num2words)


def test_legacy_top_level_normalization_package_removed() -> None:
    assert not (REPO_ROOT / "src/sure_eval/evaluation/normalization").exists()


def test_audio_semantic_helpers_live_under_transcription_nodes() -> None:
    assert not (REPO_ROOT / "src/sure_eval/evaluation/tasks/audio_semantic.py").exists()

    module = importlib.import_module("sure_eval.evaluation.nodes.transcription.common.audio_semantic")
    module_path = Path(module.__file__).resolve()

    assert "nodes/transcription/common/audio_semantic.py" in module_path.as_posix()
    assert callable(module.transcribe_audio)
    assert callable(module.score_transcripts_with_asr)


def test_audio_quality_compatibility_scoring_nodes_removed() -> None:
    scoring_root = REPO_ROOT / "src/sure_eval/evaluation/nodes/scoring"

    for removed_dir in ("default_sim", "mos", "speaker_similarity"):
        assert not (scoring_root / removed_dir).exists()


def test_tts_vc_speaker_routes_use_named_backend_nodes() -> None:
    for route_file in (
        REPO_ROOT / "src/sure_eval/evaluation/tasks/tts/routes.yaml",
        REPO_ROOT / "src/sure_eval/evaluation/tasks/vc/routes.yaml",
    ):
        content = route_file.read_text(encoding="utf-8")
        assert "scoring/default_sim" not in content
        assert "sim.default_sim" not in content
        assert "scoring/wavlm_large_sim" in content


def test_root_architecture_points_to_new_evaluation_layout() -> None:
    content = (REPO_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")

    for old_path in (
        "src/sure_eval/evaluation/asr/",
        "src/sure_eval/evaluation/classification/",
        "src/sure_eval/evaluation/s2tt/",
        "src/sure_eval/evaluation/tts/",
        "src/sure_eval/evaluation/vc/",
        "src/sure_eval/evaluation/kws/",
        "evaluation/asr/wenet_compute_cer.py",
    ):
        assert old_path not in content

    assert "src/sure_eval/evaluation/tasks/asr/" in content
    assert "src/sure_eval/evaluation/nodes/scoring/wenet_wer/" in content
