from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_tts_samples_jsonl_resolves_relative_paths_and_metadata(tmp_path: Path) -> None:
    from sure_eval.evaluation.audio_samples import load_tts_samples_jsonl

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "pred.wav").write_bytes(b"fake")
    (audio_dir / "ref.wav").write_bytes(b"fake")
    samples_jsonl = tmp_path / "samples.jsonl"
    _write_jsonl(
        samples_jsonl,
        [
            {
                "sample_id": "utt1",
                "prediction_audio": "audio/pred.wav",
                "reference_audio": "audio/ref.wav",
                "reference_text": "你好世界",
                "language": "zh",
                "metadata": {"speaker_id": "spk1"},
            }
        ],
    )

    samples = load_tts_samples_jsonl(samples_jsonl, metrics=("tts_cer", "sim/wavlm-large"))

    assert samples[0].sample_id == "utt1"
    assert samples[0].prediction_audio == str((audio_dir / "pred.wav").resolve())
    assert samples[0].reference_audio == str((audio_dir / "ref.wav").resolve())
    assert samples[0].reference_text == "你好世界"
    assert samples[0].metadata == {"speaker_id": "spk1"}


def test_load_vc_samples_jsonl_reports_line_number_for_missing_metric_role(tmp_path: Path) -> None:
    from sure_eval.evaluation.audio_samples import SampleJsonlError, load_vc_samples_jsonl

    (tmp_path / "converted.wav").write_bytes(b"fake")
    samples_jsonl = tmp_path / "samples.jsonl"
    _write_jsonl(
        samples_jsonl,
        [
            {
                "sample_id": "utt1",
                "converted_audio": "converted.wav",
                "language": "zh",
            }
        ],
    )

    with pytest.raises(SampleJsonlError, match="line 1.*reference_audio.*sim/wavlm-large"):
        load_vc_samples_jsonl(samples_jsonl, metrics=("sim/wavlm-large",))


def test_load_tts_samples_jsonl_rejects_duplicate_sample_id(tmp_path: Path) -> None:
    from sure_eval.evaluation.audio_samples import SampleJsonlError, load_tts_samples_jsonl

    (tmp_path / "a.wav").write_bytes(b"fake")
    (tmp_path / "b.wav").write_bytes(b"fake")
    samples_jsonl = tmp_path / "samples.jsonl"
    _write_jsonl(
        samples_jsonl,
        [
            {"sample_id": "dup", "prediction_audio": "a.wav", "reference_text": "a", "language": "en"},
            {"sample_id": "dup", "prediction_audio": "b.wav", "reference_text": "b", "language": "en"},
        ],
    )

    with pytest.raises(SampleJsonlError, match="line 2.*duplicate sample_id"):
        load_tts_samples_jsonl(samples_jsonl, metrics=("tts_wer",))


def test_load_tts_samples_jsonl_reports_json_parse_line_number(tmp_path: Path) -> None:
    from sure_eval.evaluation.audio_samples import SampleJsonlError, load_tts_samples_jsonl

    samples_jsonl = tmp_path / "samples.jsonl"
    samples_jsonl.write_text('{"sample_id": "utt1"\n', encoding="utf-8")

    with pytest.raises(SampleJsonlError, match="line 1.*Invalid JSON"):
        load_tts_samples_jsonl(samples_jsonl, metrics=("tts_wer",))
