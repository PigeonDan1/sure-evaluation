import json
from pathlib import Path
from sure_eval.evaluation.audio_samples import SampleJsonlError, load_tta_samples_jsonl


def test_load_tta_samples(tmp_path: Path):
    pred, ref = tmp_path / "pred.wav", tmp_path / "ref.wav"
    pred.touch()
    ref.touch()
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "x",
                "prediction_audio": "pred.wav",
                "reference_audio": "ref.wav",
                "caption": "sound",
            }
        )
        + "\n"
    )
    samples = load_tta_samples_jsonl(manifest, metrics=("fad", "clap_score"))
    assert samples[0].prediction_audio == str(pred)
    assert samples[0].caption == "sound"


def test_load_tta_missing_reference(tmp_path: Path):
    pred = tmp_path / "pred.wav"
    pred.touch()
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(json.dumps({"sample_id": "x", "prediction_audio": "pred.wav"}) + "\n")
    try:
        load_tta_samples_jsonl(manifest, metrics=("fad",))
        assert False
    except SampleJsonlError as exc:
        assert "reference_audio" in str(exc)
