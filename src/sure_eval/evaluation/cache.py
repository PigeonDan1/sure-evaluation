"""Shared cache path helpers for SURE-EVAL.

Public code must not default to a user-specific HPC path. Runtime assets can
still live on large local storage by setting ``SURE_EVAL_CACHE_DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path

CACHE_ENV_VAR = "SURE_EVAL_CACHE_DIR"


def get_cache_root(*, create: bool = True) -> Path:
    """Return the root cache directory for local runtime artifacts."""

    raw = os.environ.get(CACHE_ENV_VAR)
    root = Path(raw).expanduser() if raw else Path.home() / ".cache" / "sure-eval"
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def get_cache_dir(*parts: str, create: bool = True) -> Path:
    """Return a cache subdirectory under ``SURE_EVAL_CACHE_DIR``."""

    path = get_cache_root(create=False).joinpath(*parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
