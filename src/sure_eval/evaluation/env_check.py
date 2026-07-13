"""Environment diagnostics for route-backed evaluation nodes."""

from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sure_eval.evaluation.cache import CACHE_ENV_VAR, get_cache_root
from sure_eval.evaluation.scripts.contracts import NODES_ROOT, load_node_manifest

NODE_LOCAL_PROJECTS = {
    "scoring/bleurt_20",
    "scoring/wavlm_large_sim",
    "scoring/ecapa_tdnn_sim",
    "scoring/eres2net_sim",
    "scoring/dnsmos",
    "scoring/wv_mos",
    "scoring/utmos",
    "scoring/xcomet_xl",
    "transcription/paraformer_zh",
    "transcription/whisper_large_v3",
}

OPTIONAL_NODE_RUNTIME_TYPES = {"uv", "binary", "pip"}

DEFAULT_CHECKPOINTS_BY_NODE = {
    "scoring/bleurt_20": (
        "BLEURT_20_CHECKPOINT",
        "checkpoints/bleurt_20/saved_model/saved_model.pb",
    ),
    "scoring/xcomet_xl": (
        "XCOMET_XL_CHECKPOINT_PATH",
        "checkpoints/xcomet_xl/modelscope/evalscope/XCOMET-XL/checkpoints/model.ckpt",
    ),
    "scoring/wavlm_large_sim": (
        "WAVLM_LARGE_SIM_CHECKPOINT",
        "checkpoints/wavlm_large_finetune.pth",
    ),
    "scoring/ecapa_tdnn_sim": (
        "ECAPA_TDNN_SIM_CHECKPOINT",
        "checkpoints/huggingface/hub/models--speechbrain--spkrec-ecapa-voxceleb/snapshots/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286/embedding_model.ckpt",
    ),
    "scoring/eres2net_sim": (
        "ERES2NET_SIM_CHECKPOINT",
        "checkpoints/modelscope/models/iic/speech_eres2net_sv_zh-cn_16k-common/pretrained_eres2net_aug.ckpt",
    ),
    "scoring/dnsmos": (
        "DNSMOS_CHECKPOINT",
        "checkpoints/DNSMOS/model_v8.onnx",
    ),
    "scoring/wv_mos": (
        "WV_MOS_CHECKPOINT",
        "checkpoints/wv-mos/wav2vec2.ckpt",
    ),
    "scoring/utmos": (
        "UTMOS_CHECKPOINT",
        "checkpoints/UTMOS-demo/epoch=3-step=7459.ckpt",
    ),
    "transcription/paraformer_zh": (
        "PARAFORMER_ZH_CHECKPOINT",
        "checkpoints/modelscope/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/model.pt",
    ),
    "transcription/whisper_large_v3": (
        "WHISPER_LARGE_V3_CHECKPOINT",
        "checkpoints/huggingface/hub/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1/model.safetensors",
    ),
}


@dataclass(frozen=True)
class EnvCheckResult:
    name: str
    status: str
    message: str
    node_id: str = ""
    runtime: str = ""
    required: bool = False
    fix: str = ""
    details: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "status": self.status,
            "message": self.message,
        }
        if self.node_id:
            payload["node_id"] = self.node_id
        if self.runtime:
            payload["runtime"] = self.runtime
        if self.required:
            payload["required"] = self.required
        if self.fix:
            payload["fix"] = self.fix
        if self.details:
            payload["details"] = self.details
        return payload

    def as_warning(self, message: str | None = None) -> "EnvCheckResult":
        """Return a warning copy for optional checks used by doctor."""

        return EnvCheckResult(
            name=self.name,
            status="warning",
            message=message or self.message,
            node_id=self.node_id,
            runtime=self.runtime,
            required=self.required,
            fix=self.fix,
            details=self.details,
        )


class EnvironmentCheckError(RuntimeError):
    def __init__(self, results: list[EnvCheckResult]) -> None:
        failed = [result for result in results if result.status == "failed"]
        message = "Environment check failed: " + ", ".join(result.node_id or result.name for result in failed)
        super().__init__(message)
        self.results = results


class NodeEnvChecker:
    def __init__(self, nodes_root: Path = NODES_ROOT) -> None:
        self.nodes_root = nodes_root

    def check_node(self, node_id: str) -> EnvCheckResult:
        manifest, _ = load_node_manifest(node_id)
        node_path = self._node_path(node_id)
        node_env = self.load_node_env(node_id)
        runtime = self._runtime(node_id, node_path, node_env=node_env)
        if runtime == "in_process":
            return EnvCheckResult(
                name=node_id,
                node_id=node_id,
                runtime=runtime,
                required=False,
                status="ok",
                message="in-process node",
            )
        if runtime == "binary":
            return self._check_binary_node(node_id, node_path, node_env or {})
        if runtime == "pip_optional":
            return self._check_pip_node(node_id, node_path, node_env or {})
        python_name = self._runtime_python_name(node_env)
        venv_python = node_path / ".venv" / "bin" / python_name
        fallback_python = node_path / ".venv" / "bin" / "python"
        details = {
            "node_path": str(node_path),
            "pyproject": str(node_path / "pyproject.toml"),
            "venv_python": str(venv_python),
        }
        if fallback_python != venv_python:
            details["fallback_venv_python"] = str(fallback_python)
        checkpoint_path, checkpoint_env = self._checkpoint_path(node_id, node_path, node_env)
        if checkpoint_path is not None:
            details["checkpoint_path"] = str(checkpoint_path)
        if checkpoint_path is not None and not checkpoint_path.exists():
            return EnvCheckResult(
                name=node_id,
                node_id=node_id,
                runtime=runtime,
                required=True,
                status="failed",
                message=f"checkpoint is missing: {checkpoint_path}",
                fix=f"export {checkpoint_env}=/path/to/checkpoint",
                details=details,
            )
        venv_exists, venv_error = _path_exists(venv_python)
        fallback_exists, fallback_error = _path_exists(fallback_python)
        if not venv_exists and not fallback_exists:
            permission_error = venv_error or fallback_error
            message = ".venv is missing"
            if permission_error:
                message = f".venv python is not accessible: {permission_error}"
            return EnvCheckResult(
                name=node_id,
                node_id=node_id,
                runtime=runtime,
                required=True,
                status="failed",
                message=message,
                fix=f"cd {node_path} && uv sync",
                details=details,
            )
        return EnvCheckResult(
            name=node_id,
            node_id=node_id,
            runtime=runtime,
            required=True,
            status="ok",
            message="node-local environment exists",
            details=details,
        )

    def check_pipeline(self, pipeline: dict[str, Any]) -> list[EnvCheckResult]:
        return [self.check_node(node["node_id"]) for node in pipeline.get("nodes") or ()]

    def node_path(self, node_id: str) -> Path:
        return self._node_path(node_id)

    def node_env_path(self, node_id: str) -> Path:
        return self._node_path(node_id) / "node_env.yaml"

    def load_node_env(self, node_id: str) -> dict[str, Any] | None:
        """Load optional node environment metadata."""

        path = self.node_env_path(node_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"node_env.yaml must contain a mapping: {path}")
        return data

    def _node_path(self, node_id: str) -> Path:
        stage, name = node_id.split("/", 1)
        return self.nodes_root / stage / name

    def _checkpoint_path(
        self,
        node_id: str,
        node_path: Path,
        node_env: dict[str, Any] | None,
    ) -> tuple[Path | None, str]:
        if node_env:
            first_declared_path: Path | None = None
            first_declared_env = ""
            for model in node_env.get("models") or ():
                if not isinstance(model, dict):
                    continue
                env_name = str(model.get("env") or "")
                target = str(model.get("target") or "")
                if not target:
                    continue
                model_path = Path(os.environ.get(env_name, node_path / target)).expanduser()
                if first_declared_path is None:
                    first_declared_path = model_path
                    first_declared_env = env_name
                if not model_path.exists():
                    return model_path, env_name
            for file_name in (node_env.get("verify") or {}).get("files") or ():
                file_path = node_path / str(file_name)
                if not file_path.exists():
                    return file_path, ""
            if first_declared_path is not None:
                return first_declared_path, first_declared_env
        checkpoint_env, default_checkpoint = DEFAULT_CHECKPOINTS_BY_NODE.get(node_id, ("", ""))
        if default_checkpoint:
            return Path(os.environ.get(checkpoint_env, node_path / default_checkpoint)).expanduser(), checkpoint_env
        return None, ""

    def _check_binary_node(self, node_id: str, node_path: Path, node_env: dict[str, Any]) -> EnvCheckResult:
        details = {
            "node_path": str(node_path),
            "node_env": str(self.node_env_path(node_id)),
        }
        tools = [tool for tool in node_env.get("tools") or () if isinstance(tool, dict)]
        for tool in tools:
            env_name = str(tool.get("env") or "")
            if env_name and os.environ.get(env_name):
                candidate = Path(os.environ[env_name]).expanduser()
                details["binary_path"] = str(candidate)
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return EnvCheckResult(
                        name=node_id,
                        node_id=node_id,
                        runtime="binary",
                        required=True,
                        status="ok",
                        message="binary exists",
                        details=details,
                    )
                return EnvCheckResult(
                    name=node_id,
                    node_id=node_id,
                    runtime="binary",
                    required=True,
                    status="failed",
                    message=f"binary is not executable: {candidate}",
                    fix=f"export {env_name}=/path/to/executable",
                    details=details,
                )
        return EnvCheckResult(
            name=node_id,
            node_id=node_id,
            runtime="binary",
            required=True,
            status="warning",
            message="binary node is optional and not validated without an explicit env var",
            fix="Set the binary env var or run the node build script.",
            details=details,
        )

    def _check_pip_node(self, node_id: str, node_path: Path, node_env: dict[str, Any]) -> EnvCheckResult:
        details = {
            "node_path": str(node_path),
            "node_env": str(self.node_env_path(node_id)),
            "packages": package_install_specs(node_env),
        }
        verify = node_env.get("verify") if isinstance(node_env.get("verify"), dict) else {}
        imports = [str(name) for name in verify.get("imports") or ()]
        details["imports"] = imports

        missing_files = []
        for file_name in verify.get("files") or ():
            file_path = node_path / str(file_name)
            if not file_path.exists():
                missing_files.append(str(file_path))
        if missing_files:
            details["missing_files"] = missing_files
            return EnvCheckResult(
                name=node_id,
                node_id=node_id,
                runtime="pip_optional",
                required=True,
                status="failed",
                message=f"verify file is missing: {missing_files[0]}",
                details=details,
            )

        missing_imports = [name for name in imports if importlib.util.find_spec(name) is None]
        if missing_imports:
            details["missing_imports"] = missing_imports
            install_specs = package_install_specs(node_env)
            fix = "Install missing Python package(s)"
            if install_specs:
                fix = "python -m pip install " + " ".join(shlex.quote(spec) for spec in install_specs)
            return EnvCheckResult(
                name=node_id,
                node_id=node_id,
                runtime="pip_optional",
                required=True,
                status="failed",
                message=f"missing import(s): {', '.join(missing_imports)}",
                fix=fix,
                details=details,
            )

        return EnvCheckResult(
            name=node_id,
            node_id=node_id,
            runtime="pip_optional",
            required=True,
            status="ok",
            message="pip runtime imports available",
            details=details,
        )

    @staticmethod
    def _runtime_python_name(node_env: dict[str, Any] | None) -> str:
        if not node_env:
            return "python"
        runtime = node_env.get("runtime") if isinstance(node_env.get("runtime"), dict) else {}
        python_version = str(runtime.get("python") or "")
        if python_version and python_version.count(".") == 1:
            return f"python{python_version}"
        return "python"

    @staticmethod
    def _runtime(node_id: str, node_path: Path, *, node_env: dict[str, Any] | None = None) -> str:
        if node_env:
            runtime = node_env.get("runtime") if isinstance(node_env.get("runtime"), dict) else {}
            runtime_type = str(runtime.get("type") or "").strip()
            if runtime_type == "binary":
                return "binary"
            if runtime_type == "uv":
                return "node_local_project"
            if runtime_type == "pip":
                return "pip_optional"
        if node_id in NODE_LOCAL_PROJECTS:
            return "node_local_project"
        if (node_path / "pyproject.toml").exists() and node_id not in {
            "scoring/sacrebleu",
            "scoring/wekws_det",
            "scoring/meeteval",
        }:
            return "node_local_optional"
        return "in_process"


def package_install_specs(node_env: dict[str, Any]) -> list[str]:
    specs = []
    for package in node_env.get("packages") or ():
        if not isinstance(package, dict):
            continue
        name = str(package.get("name") or "")
        version = str(package.get("version") or "")
        if not name:
            continue
        specs.append(f"{name}{version}" if version else name)
    return specs


def check_pipeline_environment(pipeline: dict[str, Any]) -> list[EnvCheckResult]:
    return NodeEnvChecker().check_pipeline(pipeline)


def _path_exists(path: Path) -> tuple[bool, str]:
    try:
        return path.exists(), ""
    except OSError as exc:
        return False, str(exc)


def raise_if_environment_failed(results: list[EnvCheckResult]) -> None:
    if any(result.status == "failed" for result in results):
        raise EnvironmentCheckError(results)


def iter_known_node_ids() -> tuple[str, ...]:
    """Return node ids that need explicit environment visibility."""

    discovered = {
        f"{path.relative_to(NODES_ROOT).parts[0]}/{path.relative_to(NODES_ROOT).parts[1]}"
        for path in NODES_ROOT.glob("*/*/node_env.yaml")
    }
    return tuple(sorted(discovered | NODE_LOCAL_PROJECTS))


def doctor_checks() -> list[EnvCheckResult]:
    checks = [
        EnvCheckResult(
            name="python",
            status="ok" if sys.version_info >= (3, 10) else "failed",
            message=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            fix="Use Python >= 3.10",
        ),
        EnvCheckResult(
            name="uv",
            status="ok" if shutil.which("uv") else "warning",
            message=shutil.which("uv") or "uv not found on PATH",
            fix="Install uv to use `sure-eval env setup`",
        ),
        EnvCheckResult(
            name="sure_eval",
            status="ok" if importlib.util.find_spec("sure_eval") else "failed",
            message="importable" if importlib.util.find_spec("sure_eval") else "not importable",
            fix="Set PYTHONPATH=src or install the package",
        ),
        EnvCheckResult(
            name="cache_root",
            status="ok",
            message=str(get_cache_root(create=False)),
            details={"env_var": CACHE_ENV_VAR},
        ),
    ]
    checker = NodeEnvChecker()
    for node_id in sorted(NODE_LOCAL_PROJECTS):
        result = checker.check_node(node_id)
        if result.status == "failed":
            result = result.as_warning(
                f"optional node is not prepared: {result.message}"
            )
        checks.append(result)
    return checks


def doctor_payload() -> dict[str, Any]:
    checks = doctor_checks()
    failed = [check for check in checks if check.status == "failed"]
    warnings = [check for check in checks if check.status == "warning"]
    status = "failed" if failed else "warning" if warnings else "ok"
    return {
        "status": status,
        "checks": [check.as_dict() for check in checks],
    }
