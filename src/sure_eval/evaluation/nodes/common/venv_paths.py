"""Platform-aware paths for node-local Python virtual environments."""

from __future__ import annotations

import os
from pathlib import Path


def venv_python_candidates(
    venv_dir: Path,
    python_name: str = "python",
    *,
    windows: bool | None = None,
) -> tuple[Path, ...]:
    """Return the preferred node-local Python path and portable fallbacks."""

    use_windows_layout = os.name == "nt" if windows is None else windows
    if use_windows_layout:
        return (venv_dir / "Scripts" / "python.exe",)

    preferred = venv_dir / "bin" / python_name
    fallback = venv_dir / "bin" / "python"
    return (preferred,) if preferred == fallback else (preferred, fallback)
