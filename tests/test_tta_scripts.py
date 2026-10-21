from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sure_eval.evaluation.scripts.tta import describe_pipeline


def test_tta_describe_atomic():
    desc = describe_pipeline(metrics="fad")
    assert desc.task == "TTA"
    assert desc.language == "n/a"
    assert desc.pipeline_id == "tta.any.fad.audioldm_eval_v1"
    assert desc.required_roles == ("prediction_audio", "reference_audio")


def test_tta_describe_default_bundle():
    desc = describe_pipeline()
    assert desc.pipeline_kind == "bundle"
    assert len(desc.member_pipeline_ids) == 9
    assert desc.node_ids == (
        "scoring/audioldm_eval",
        "scoring/clap_score",
        "scoring/audiobox_aesthetics",
    )


def test_tta_model_assets_are_pinned_for_setup_download():
    from sure_eval.evaluation.cli import _setup_plan_for_node

    for node_id in (
        "scoring/audioldm_eval",
        "scoring/clap_score",
        "scoring/audiobox_aesthetics",
    ):
        downloads = _setup_plan_for_node(node_id, no_download=False)["downloads"]
        assert downloads
        assert all(asset["auto_download"] for asset in downloads)
        assert all(asset.get("sha256") or asset.get("md5") for asset in downloads)


def test_tta_clap_declares_compatible_torchvision():
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib
    from pathlib import Path

    from sure_eval.evaluation.nodes.scoring.clap_score import node

    project = tomllib.loads(Path(node.__file__).with_name("pyproject.toml").read_text())
    assert "torchvision==0.29.1" in project["project"]["dependencies"]


def test_tta_clap_downloads_only_required_roberta_files():
    from sure_eval.evaluation.cli import _setup_plan_for_node

    downloads = _setup_plan_for_node("scoring/clap_score", no_download=False)["downloads"]
    roberta = [asset for asset in downloads if str(asset["id"]).startswith("roberta-base/")]

    assert {asset["id"] for asset in roberta} == {
        "roberta-base/config.json",
        "roberta-base/merges.txt",
        "roberta-base/tokenizer_config.json",
        "roberta-base/vocab.json",
    }
    assert all(asset["provider"] == "url" for asset in roberta)
    assert not any("model.safetensors" in str(asset.get("source")) for asset in downloads)
    assert not any("bert-base-uncased" in str(asset.get("source")) for asset in downloads)


def test_url_download_is_atomic_and_verifies_md5(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import io

    from sure_eval.evaluation import cli

    payload = b"panns-checkpoint"
    calls = []

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
    target = tmp_path / "checkpoints" / "cnn14.pth"
    cli._download_asset(
        {
            "provider": "url",
            "id": "PANNs/Cnn14",
            "source": "https://example.invalid/cnn14.pth",
            "target_path": str(target),
            "md5": hashlib.md5(payload).hexdigest(),
        }
    )

    assert target.read_bytes() == payload
    assert calls == [("https://example.invalid/cnn14.pth", 60)]
    assert list(target.parent.glob("*.part")) == []


def test_env_setup_automatically_downloads_supported_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sure_eval.evaluation import cli

    node_path = tmp_path / "node"
    node_path.mkdir()
    monkeypatch.setattr(cli, "_setup_log_file", lambda node_id: tmp_path / "setup.log")
    monkeypatch.setattr(cli, "_execute_uv_setup", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli,
        "_download_plan_for_node",
        lambda node_id: {
            "node_id": node_id,
            "assets": [
                {
                    "provider": "url",
                    "id": "asset",
                    "status": "planned",
                    "auto_download": True,
                }
            ],
        },
    )
    monkeypatch.setattr(
        cli,
        "_execute_automatic_download_plan",
        lambda plan: {
            **plan,
            "assets": [{**plan["assets"][0], "status": "ok"}],
        },
    )

    result = cli._execute_setup_action(
        {
            "node_id": "scoring/example",
            "node_path": str(node_path),
            "runtime": "uv",
            "no_download": False,
            "downloads": [{"id": "asset", "auto_download": True}],
        },
        force=False,
    )

    assert result["status"] == "ok"
    assert result["downloads"][0]["status"] == "ok"
