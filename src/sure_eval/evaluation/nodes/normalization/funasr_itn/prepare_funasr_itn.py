"""Prepare the immutable FunASR text-processing source for this node."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

NODE_DIR = Path(__file__).resolve().parent
LOCK_FILE = NODE_DIR / "source_lock.json"
RUNTIME_DIR = NODE_DIR / "runtime"
MARKER_FILE = RUNTIME_DIR / "funasr_revision.json"


def _load_lock() -> dict[str, str]:
    payload = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    required = ("repository", "revision", "subdirectory", "source_tree", "license_file")
    missing = [key for key in required if not str(payload.get(key) or "").strip()]
    if missing:
        raise RuntimeError(f"{LOCK_FILE} is missing required fields: {', '.join(missing)}")
    return {key: str(payload[key]) for key in required}


def _read_marker() -> dict[str, Any]:
    if not MARKER_FILE.is_file():
        return {}
    try:
        payload = json.loads(MARKER_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_prepared(lock: dict[str, str]) -> bool:
    marker = _read_marker()
    return (
        marker.get("repository") == lock["repository"]
        and marker.get("revision") == lock["revision"]
        and marker.get("subdirectory") == lock["subdirectory"]
        and marker.get("source_tree") == lock["source_tree"]
        and (RUNTIME_DIR / lock["subdirectory"] / "__init__.py").is_file()
        and (RUNTIME_DIR / lock["license_file"]).is_file()
    )


def _run(command: list[str], *, cwd: Path | None = None, timeout: int = 600) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"required executable is missing: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"command timed out after {timeout}s: {' '.join(command)}") from exc
    if completed.returncode != 0:
        output = (completed.stdout or "").strip()
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {' '.join(command)}"
            + (f"\n{output}" if output else "")
        )
    return (completed.stdout or "").strip()


def _checkout_source(checkout_dir: Path, lock: dict[str, str]) -> str:
    _run(["git", "init", "--quiet", str(checkout_dir)])
    _run(["git", "remote", "add", "origin", lock["repository"]], cwd=checkout_dir)
    _run(["git", "config", "remote.origin.promisor", "true"], cwd=checkout_dir)
    _run(["git", "config", "remote.origin.partialclonefilter", "blob:none"], cwd=checkout_dir)
    _run(["git", "sparse-checkout", "init", "--cone"], cwd=checkout_dir)
    _run(["git", "sparse-checkout", "set", lock["subdirectory"]], cwd=checkout_dir)
    _run(
        [
            "git",
            "-c",
            "protocol.version=2",
            "fetch",
            "--filter=blob:none",
            "--depth",
            "1",
            "origin",
            lock["revision"],
        ],
        cwd=checkout_dir,
    )
    _run(["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=checkout_dir)
    actual_revision = _run(["git", "rev-parse", "HEAD"], cwd=checkout_dir)
    if actual_revision != lock["revision"]:
        raise RuntimeError(
            f"FunASR revision mismatch: expected {lock['revision']}, got {actual_revision}"
        )
    source_tree = _run(["git", "rev-parse", f"HEAD:{lock['subdirectory']}"], cwd=checkout_dir)
    if source_tree != lock["source_tree"]:
        raise RuntimeError(
            f"FunASR source tree mismatch: expected {lock['source_tree']}, got {source_tree}"
        )
    return source_tree


def _stage_runtime(build_root: Path, lock: dict[str, str], source_tree: str) -> Path:
    checkout_dir = build_root / "checkout"
    staged_runtime = build_root / "runtime"
    shutil.copytree(checkout_dir / lock["subdirectory"], staged_runtime / lock["subdirectory"])
    shutil.copy2(checkout_dir / lock["license_file"], staged_runtime / lock["license_file"])
    marker = {
        "repository": lock["repository"],
        "revision": lock["revision"],
        "subdirectory": lock["subdirectory"],
        "source_tree": source_tree,
        "license_file": lock["license_file"],
    }
    (staged_runtime / MARKER_FILE.name).write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return staged_runtime


def _install_runtime(staged_runtime: Path) -> None:
    backup = NODE_DIR / f".runtime-backup-{os.getpid()}"
    if backup.exists():
        shutil.rmtree(backup)
    if RUNTIME_DIR.exists():
        RUNTIME_DIR.rename(backup)
    try:
        staged_runtime.rename(RUNTIME_DIR)
    except Exception:
        if backup.exists() and not RUNTIME_DIR.exists():
            backup.rename(RUNTIME_DIR)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def prepare(*, force: bool = False) -> dict[str, Any]:
    lock = _load_lock()
    if not force and _is_prepared(lock):
        return {"status": "already_prepared", **_read_marker()}

    builds_dir = NODE_DIR / ".runtime-builds"
    builds_dir.mkdir(parents=True, exist_ok=True)
    build_root = Path(tempfile.mkdtemp(prefix="prepare-", dir=builds_dir))
    try:
        checkout_dir = build_root / "checkout"
        source_tree = _checkout_source(checkout_dir, lock)
        staged_runtime = _stage_runtime(build_root, lock, source_tree)
        _install_runtime(staged_runtime)
    finally:
        shutil.rmtree(build_root, ignore_errors=True)
    return {"status": "prepared", **_read_marker()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(prepare(force=args.force), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
