from __future__ import annotations

import zipfile
from pathlib import Path

from scripts.check_wheel_node_assets import missing_node_assets


def _write_wheel(path: Path, *, include_lock: bool, include_runtime: bool = False) -> None:
    node_dir = "sure_eval/evaluation/nodes/normalization/example"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{node_dir}/node_env.yaml",
            "runtime:\n  type: uv\n  project: pyproject.toml\n  frozen: true\n",
        )
        archive.writestr(f"{node_dir}/pyproject.toml", "[project]\nname = 'example'\n")
        if include_lock:
            archive.writestr(f"{node_dir}/uv.lock", "version = 1\n")
        if include_runtime:
            archive.writestr(f"{node_dir}/runtime/generated.py", "")


def test_wheel_node_assets_accepts_complete_runtime(tmp_path: Path) -> None:
    wheel = tmp_path / "complete.whl"
    _write_wheel(wheel, include_lock=True)
    assert missing_node_assets(wheel) == []


def test_wheel_node_assets_reports_missing_frozen_lock(tmp_path: Path) -> None:
    wheel = tmp_path / "missing.whl"
    _write_wheel(wheel, include_lock=False)
    assert missing_node_assets(wheel) == [
        "sure_eval/evaluation/nodes/normalization/example/node_env.yaml: missing uv.lock"
    ]


def test_wheel_node_assets_rejects_local_runtime(tmp_path: Path) -> None:
    wheel = tmp_path / "runtime.whl"
    _write_wheel(wheel, include_lock=True, include_runtime=True)
    assert missing_node_assets(wheel) == [
        "wheel contains node runtime asset: "
        "sure_eval/evaluation/nodes/normalization/example/runtime/generated.py"
    ]
