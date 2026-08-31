"""Tests for the Open ASR Leaderboard recipe scripts (no GPU, no real data)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_dataset(root: Path, name: str, refs: list[tuple[str, str]]) -> Path:
    dataset_dir = root / name / "v1.0.0"
    (dataset_dir / "wavs").mkdir(parents=True)
    (dataset_dir / "ref.txt").write_text(
        "".join(f"{key}\t{text}\n" for key, text in refs), encoding="utf-8"
    )
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"dataset": name, "total_audio_s": 20.0, "num_samples": len(refs)}),
        encoding="utf-8",
    )
    return dataset_dir


def test_prepare_script_rejects_unknown_dataset(tmp_path: Path) -> None:
    module = _load_script("prepare_openasr_leaderboard")
    assert "librispeech_test.clean" in module.ALL_DATASETS
    assert "earnings22_cleaned_aa_test" in module.ALL_DATASETS
    assert len(module.ALL_DATASETS) == 7


def test_prepare_script_extract_text_fallback_order() -> None:
    module = _load_script("prepare_openasr_leaderboard")
    assert module.extract_text({"text": "a", "transcript": "b"}) == "a"
    assert module.extract_text({"transcript": "b"}) == "b"
    with pytest.raises(ValueError, match="text columns"):
        module.extract_text({"id": "x"})


def test_run_script_aggregate_mean_and_rtfx() -> None:
    module = _load_script("run_openasr_leaderboard_eval")
    result = module.aggregate(
        {
            "a": {"wer_percent": 10.0, "audio_s": 100.0, "inference_s": 20.0},
            "b": {"wer_percent": 20.0, "audio_s": 300.0, "inference_s": 60.0},
        }
    )
    assert result["mean_wer_percent"] == 15.0
    assert result["rtfx"] == 5.0
    assert module.aggregate({}) == {"mean_wer_percent": None, "rtfx": None}


def test_run_script_scores_existing_hypotheses(tmp_path: Path) -> None:
    module = _load_script("run_openasr_leaderboard_eval")
    data_root = tmp_path / "data"
    _write_dataset(
        data_root,
        "librispeech_test.clean",
        [("utt1", "Um, the B B C paid $5."), ("utt2", "ignore time segment in scoring")],
    )
    hyp_dir = tmp_path / "hyps" / "librispeech_test.clean"
    hyp_dir.mkdir(parents=True)
    (hyp_dir / "hyp.txt").write_text(
        "utt1\tthe bbc paid five dollars\nutt2\tignored\n", encoding="utf-8"
    )

    scores = module.score_dataset(
        data_root / "librispeech_test.clean" / "v1.0.0" / "ref.txt",
        hyp_dir / "hyp.txt",
    )
    assert scores["pipeline_id"] == "asr.en.wer.openasr_norm_english_v1.wenet_wer_v1"
    assert scores["wer_percent"] == 0.0
    assert scores["num_ref_utts"] == 1  # ignore-marker ref dropped by the node


@pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None, reason="pyarrow not installed"
)
def test_prepare_script_parquet_roundtrip(tmp_path: Path) -> None:
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    module = _load_script("prepare_openasr_leaderboard")

    import numpy as np
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(16000, dtype="float32"), 16000, format="wav", subtype="PCM_16")
    audio_bytes = buffer.getvalue()

    source = tmp_path / "src" / "librispeech"
    source.mkdir(parents=True)
    table = pa.table(
        {
            "id": ["utt1", "utt2"],
            "text": ["Hello, World!", "second sample"],
            "audio": [{"bytes": audio_bytes, "path": "utt1.wav"}, {"bytes": audio_bytes, "path": "utt2.wav"}],
            "audio_length_s": [1.0, 1.0],
        }
    )
    pq.write_table(table, source / "test.clean-00000-of-00001.parquet")

    output_root = tmp_path / "out"
    manifest = module.prepare_parquet_dataset(
        "librispeech_test.clean", tmp_path / "src", output_root, limit=None
    )

    dataset_dir = output_root / "librispeech_test.clean" / "v1.0.0"
    assert manifest["num_samples"] == 2
    assert manifest["total_audio_s"] == 2.0
    assert manifest["sources"][0]["path"] == "librispeech/test.clean-00000-of-00001.parquet"
    assert len(manifest["sources"][0]["sha256"]) == 64
    assert (dataset_dir / "ref.txt").read_text(encoding="utf-8") == (
        "utt1\tHello, World!\nutt2\tsecond sample\n"
    )
    for key in ("utt1", "utt2"):
        wav = dataset_dir / "wavs" / f"{key}.wav"
        assert wav.exists()
        data, rate = sf.read(str(wav))
        assert rate == 16000


def test_prepare_script_earnings22_jsonl_variant(tmp_path: Path) -> None:
    pytest.importorskip("librosa")
    import numpy as np
    import soundfile as sf

    module = _load_script("prepare_openasr_leaderboard")

    source = tmp_path / "src" / "earnings22_cleaned_aa"
    (source / "audio").mkdir(parents=True)
    sf.write(
        str(source / "audio" / "a1.wav"),
        np.zeros(16000, dtype="float32"),
        16000,
        subtype="PCM_16",
    )
    (source / "earnings22_cleaned_aa_v1.jsonl").write_text(
        json.dumps(
            {
                "id": "utt1",
                "duration": 1.0,
                "transcript": "hello earnings",
                "file_name": "a1.wav",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "out"
    manifest = module.prepare_earnings22(tmp_path / "src", output_root, limit=None)

    dataset_dir = output_root / "earnings22_cleaned_aa_test" / "v1.0.0"
    assert manifest["num_samples"] == 1
    assert (dataset_dir / "ref.txt").read_text(encoding="utf-8") == "utt1\thello earnings\n"
    assert (dataset_dir / "wavs" / "utt1.wav").exists()
