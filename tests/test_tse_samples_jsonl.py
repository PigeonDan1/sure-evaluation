from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_tse_samples_jsonl_resolves_relative_paths_and_metadata(tmp_path: Path) -> None:
    from sure_eval.evaluation.audio_samples import load_tse_samples_jsonl

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "pred.wav").write_bytes(b"fake")
    (audio_dir / "ref.wav").write_bytes(b"fake")
    (audio_dir / "mixed.wav").write_bytes(b"fake")
    (audio_dir / "enroll.wav").write_bytes(b"fake")
    samples_jsonl = tmp_path / "samples.jsonl"
    _write_jsonl(
        samples_jsonl,
        [
            {
                "sample_id": "utt1",
                "prediction_audio": "audio/pred.wav",
                "reference_audio": "audio/ref.wav",
                "mixed_audio": "audio/mixed.wav",
                "enrollment_audio": "audio/enroll.wav",
                "language": "zh",
                "metadata": {"speaker_id": "spk1"},
            }
        ],
    )

    samples = load_tse_samples_jsonl(samples_jsonl, metrics=("si_sdr",))

    assert samples[0].sample_id == "utt1"
    assert samples[0].prediction_audio == str((audio_dir / "pred.wav").resolve())
    assert samples[0].reference_audio == str((audio_dir / "ref.wav").resolve())
    assert samples[0].mixed_audio == str((audio_dir / "mixed.wav").resolve())
    assert samples[0].enrollment_audio == str((audio_dir / "enroll.wav").resolve())
    assert samples[0].language == "zh"
    assert samples[0].metadata == {"speaker_id": "spk1"}


def test_load_tse_samples_jsonl_requires_reference_text_for_semantic_metric(tmp_path: Path) -> None:
    from sure_eval.evaluation.audio_samples import SampleJsonlError, load_tse_samples_jsonl

    (tmp_path / "pred.wav").write_bytes(b"fake")
    (tmp_path / "ref.wav").write_bytes(b"fake")
    samples_jsonl = tmp_path / "samples.jsonl"
    _write_jsonl(
        samples_jsonl,
        [
            {
                "sample_id": "utt1",
                "prediction_audio": "pred.wav",
                "reference_audio": "ref.wav",
                "language": "zh",
            }
        ],
    )

    with pytest.raises(SampleJsonlError, match="reference_text.*tse_wer/tse_cer"):
        load_tse_samples_jsonl(samples_jsonl, metrics=("tse_cer",))


def test_load_tse_samples_jsonl_rejects_duplicate_sample_id(tmp_path: Path) -> None:
    from sure_eval.evaluation.audio_samples import SampleJsonlError, load_tse_samples_jsonl

    (tmp_path / "a.wav").write_bytes(b"fake")
    (tmp_path / "b.wav").write_bytes(b"fake")
    (tmp_path / "c.wav").write_bytes(b"fake")
    (tmp_path / "d.wav").write_bytes(b"fake")
    samples_jsonl = tmp_path / "samples.jsonl"
    _write_jsonl(
        samples_jsonl,
        [
            {"sample_id": "dup", "prediction_audio": "a.wav", "reference_audio": "c.wav", "language": "en"},
            {"sample_id": "dup", "prediction_audio": "b.wav", "reference_audio": "d.wav", "language": "en"},
        ],
    )

    with pytest.raises(SampleJsonlError, match="duplicate sample_id: dup"):
        load_tse_samples_jsonl(samples_jsonl)


def test_load_tse_samples_jsonl_rejects_missing_required_field(tmp_path: Path) -> None:
    from sure_eval.evaluation.audio_samples import SampleJsonlError, load_tse_samples_jsonl

    (tmp_path / "pred.wav").write_bytes(b"fake")
    samples_jsonl = tmp_path / "samples.jsonl"
    _write_jsonl(
        samples_jsonl,
        [
            {
                "sample_id": "utt1",
                "prediction_audio": "pred.wav",
                "language": "en",
            }
        ],
    )

    with pytest.raises(SampleJsonlError, match="reference_audio is required"):
        load_tse_samples_jsonl(samples_jsonl)


def test_load_tse_samples_jsonl_rejects_mixed_languages(tmp_path: Path) -> None:
    from sure_eval.evaluation.audio_samples import SampleJsonlError, load_tse_samples_jsonl

    (tmp_path / "a.wav").write_bytes(b"fake")
    (tmp_path / "b.wav").write_bytes(b"fake")
    (tmp_path / "c.wav").write_bytes(b"fake")
    (tmp_path / "d.wav").write_bytes(b"fake")
    samples_jsonl = tmp_path / "samples.jsonl"
    _write_jsonl(
        samples_jsonl,
        [
            {"sample_id": "utt1", "prediction_audio": "a.wav", "reference_audio": "c.wav", "language": "zh"},
            {"sample_id": "utt2", "prediction_audio": "b.wav", "reference_audio": "d.wav", "language": "en"},
        ],
    )

    with pytest.raises(SampleJsonlError, match="exactly one language"):
        load_tse_samples_jsonl(samples_jsonl)


def test_load_tse_samples_jsonl_optional_fields_default_empty(tmp_path: Path) -> None:
    from sure_eval.evaluation.audio_samples import load_tse_samples_jsonl

    (tmp_path / "pred.wav").write_bytes(b"fake")
    (tmp_path / "ref.wav").write_bytes(b"fake")
    samples_jsonl = tmp_path / "samples.jsonl"
    _write_jsonl(
        samples_jsonl,
        [
            {
                "sample_id": "utt1",
                "prediction_audio": "pred.wav",
                "reference_audio": "ref.wav",
                "language": "en",
            }
        ],
    )

    samples = load_tse_samples_jsonl(samples_jsonl, metrics=("si_sdr",))
    assert samples[0].mixed_audio == ""
    assert samples[0].enrollment_audio == ""
    assert samples[0].reference_text == ""