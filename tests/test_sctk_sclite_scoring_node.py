from __future__ import annotations

import sys
from pathlib import Path


def _write_key_text(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"{key}\t{text}\n" for key, text in rows), encoding="utf-8")


def _write_fake_sclite(path: Path) -> None:
    path.write_text(
        f"""#!{sys.executable}
from __future__ import annotations

import sys
from pathlib import Path

args = sys.argv[1:]
out_dir = Path(args[args.index("-O") + 1])
root = args[args.index("-n") + 1]
metric = "cer" if "-c" in args else "wer"
(out_dir / f"{{root}}.dtl").write_text(
    "\\n".join([
        "Fake SCTK sclite detail report",
        "id: sure-000001",
        "Scores: (#C #S #D #I) 2 1 0 1",
        "id: sure-000002",
        "Scores: (#C #S #D #I) 3 0 1 0",
        f"metric={{metric}}",
    ]) + "\\n",
    encoding="utf-8",
)
(out_dir / f"{{root}}.sys").write_text("Percent Total Error = 50.0%\\n", encoding="utf-8")
print("fake sclite ok")
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_sctk_sclite_node_scores_with_fake_binary(tmp_path: Path) -> None:
    from sure_eval.evaluation.core.types import KeyTextFiles
    from sure_eval.evaluation.nodes.scoring.sctk_sclite import score_sctk_sclite_wer

    fake_sclite = tmp_path / "sclite"
    _write_fake_sclite(fake_sclite)
    ref = tmp_path / "ref.txt"
    hyp = tmp_path / "hyp.txt"
    _write_key_text(ref, [("utt-a", "hello world"), ("utt-b", "nice day")])
    _write_key_text(hyp, [("utt-a", "hello brave world"), ("utt-b", "nice")])

    _, result = score_sctk_sclite_wer(
        KeyTextFiles(ref_file=str(ref), hyp_file=str(hyp)),
        sclite_bin=str(fake_sclite),
    )

    payload = result.details["result"]
    assert result.node_id == "scoring/sctk_sclite"
    assert result.internal_stages == ("key_text_parse", "trn_materialize", "sclite", "sclite_report_parse")
    assert payload["wer"] == 3 / 7
    assert payload["wer_percent"] == (3 / 7) * 100
    assert payload["all"] == 7
    assert payload["cor"] == 5
    assert payload["sub"] == 1
    assert payload["del"] == 1
    assert payload["ins"] == 1
    assert result.details["id_map"] == {"utt-a": "sure-000001", "utt-b": "sure-000002"}
    assert result.details["binary"]["path"] == str(fake_sclite)


def test_sctk_sclite_node_uses_noascii_for_cer(tmp_path: Path) -> None:
    from sure_eval.evaluation.core.types import KeyTextFiles
    from sure_eval.evaluation.nodes.scoring.sctk_sclite import score_sctk_sclite_cer

    fake_sclite = tmp_path / "sclite"
    _write_fake_sclite(fake_sclite)
    ref = tmp_path / "ref.txt"
    hyp = tmp_path / "hyp.txt"
    _write_key_text(ref, [("utt-a", "你好世界")])
    _write_key_text(hyp, [("utt-a", "你好世")])

    _, result = score_sctk_sclite_cer(
        KeyTextFiles(ref_file=str(ref), hyp_file=str(hyp)),
        sclite_bin=str(fake_sclite),
    )

    assert result.details["result"]["cer"] == 3 / 7
    assert "-c" in result.details["command"]
    assert "NOASCII" in result.details["command"]


def test_sctk_sclite_node_rejects_key_mismatch(tmp_path: Path) -> None:
    from sure_eval.evaluation.core.types import KeyTextFiles
    from sure_eval.evaluation.nodes.scoring.sctk_sclite import score_sctk_sclite_wer

    fake_sclite = tmp_path / "sclite"
    _write_fake_sclite(fake_sclite)
    ref = tmp_path / "ref.txt"
    hyp = tmp_path / "hyp.txt"
    _write_key_text(ref, [("utt-a", "hello")])
    _write_key_text(hyp, [("utt-b", "hello")])

    try:
        score_sctk_sclite_wer(
            KeyTextFiles(ref_file=str(ref), hyp_file=str(hyp)),
            sclite_bin=str(fake_sclite),
        )
    except ValueError as exc:
        assert "identical ref/hyp key sets" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_sctk_sclite_binary_resolution_reports_searched_paths(monkeypatch) -> None:
    from sure_eval.evaluation.nodes.scoring.sctk_sclite.node import ENV_SCLITE_BIN, ENV_SCTK_ROOT, resolve_sclite_binary

    monkeypatch.delenv(ENV_SCLITE_BIN, raising=False)
    monkeypatch.delenv(ENV_SCTK_ROOT, raising=False)
    monkeypatch.setenv("PATH", "")

    try:
        resolve_sclite_binary("/missing/sclite")
    except RuntimeError as exc:
        message = str(exc)
        assert "Searched:" in message
        assert "/missing/sclite" in message
        assert ENV_SCLITE_BIN in message
    else:
        raise AssertionError("expected RuntimeError")


def test_sctk_sclite_binary_resolution_prefers_env(monkeypatch, tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring.sctk_sclite.node import ENV_SCLITE_BIN, resolve_sclite_binary

    fake_sclite = tmp_path / "sclite"
    _write_fake_sclite(fake_sclite)
    monkeypatch.setenv(ENV_SCLITE_BIN, str(fake_sclite))

    resolved = resolve_sclite_binary()

    assert resolved.path == str(fake_sclite)
    assert resolved.source == ENV_SCLITE_BIN
