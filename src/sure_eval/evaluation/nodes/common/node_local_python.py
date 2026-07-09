"""Utilities for executing node-local uv environments from subprocess nodes."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NodeLocalPython:
    """Resolved Python command and PYTHONPATH policy for a node-local env."""

    command_prefix: tuple[str, ...]
    extra_pythonpath: tuple[str, ...]
    inherit_pythonpath: bool
    isolated: bool


def resolve_node_local_python(node_dir: Path, node_id: str) -> NodeLocalPython:
    """Resolve a node-local Python command without leaking incompatible deps.

    Node `.venv/bin/python` is the preferred execution path. Some vc base
    images do not contain the interpreter target used by the node-local venv
    symlink; when the node has site-packages for the current interpreter, run
    the current interpreter with `-S` and only the node-local site-packages.
    """

    python_bin = node_dir / ".venv" / "bin" / "python"
    if not python_bin.exists() and not python_bin.is_symlink():
        raise RuntimeError(f"{node_id} local environment is missing: {python_bin}")

    venv_site_packages = (
        node_dir
        / ".venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    extra_pythonpath = (str(venv_site_packages),) if venv_site_packages.exists() else ()
    override = _node_python_override(node_id)
    if override:
        override_path = Path(override)
        if not override_path.exists() or not os.access(override_path, os.X_OK):
            raise RuntimeError(f"{node_id} Python override is not executable: {override_path}")
        return NodeLocalPython(
            command_prefix=(str(override_path),),
            extra_pythonpath=extra_pythonpath,
            inherit_pythonpath=False,
            isolated=False,
        )

    if os.access(python_bin, os.X_OK):
        return NodeLocalPython(
            command_prefix=(str(python_bin),),
            extra_pythonpath=extra_pythonpath,
            inherit_pythonpath=False,
            isolated=False,
        )

    if venv_site_packages.exists():
        return NodeLocalPython(
            command_prefix=(sys.executable, "-S"),
            extra_pythonpath=extra_pythonpath,
            inherit_pythonpath=False,
            isolated=True,
        )

    raise RuntimeError(
        f"{node_id} local environment python is not executable in this container "
        f"and no compatible Python {sys.version_info.major}.{sys.version_info.minor} "
        f"site-packages were found: {python_bin}"
    )


def _node_python_override(node_id: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in node_id.upper()).strip("_")
    keys = [
        f"SURE_EVAL_NODE_LOCAL_PYTHON_{normalized}",
        "SURE_EVAL_NODE_LOCAL_PYTHON",
    ]
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    return ""


def build_node_local_env(
    *,
    repo_src: Path,
    extra_pythonpath: tuple[str, ...],
    inherit_pythonpath: bool,
) -> dict[str, str]:
    """Build subprocess env for a node-local invocation."""

    env = os.environ.copy()
    pythonpath_parts = [str(repo_src), *extra_pythonpath]
    if inherit_pythonpath and env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["PYTHONNOUSERSITE"] = "1"
    return env
