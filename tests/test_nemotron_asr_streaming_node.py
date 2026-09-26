"""Tests for the Nemotron-3.5-ASR-0.6B transcription node (no GPU, no real data)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

NODE_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "sure_eval"
    / "evaluation"
    / "nodes"
    / "transcription"
    / "nemotron_asr_streaming_06b"
)


def _load_node_module(name: str = "node"):
    spec = importlib.util.spec_from_file_location(
        name, NODE_DIR / "node.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_node_module_exports_helpers() -> None:
    module = _load_node_module()
    assert module.NODE_ID == "transcription/nemotron_asr_streaming_06b"
    assert module.NODE_VERSION == "v1"
    assert callable(module._load_audio_as_float32_vector)
    assert callable(module._transcribe_batched)


def test_language_resolution() -> None:
    module = _load_node_module()
    assert module._resolve_checkpoint() == module.MODEL_ID


def test_manifest_yaml_is_parseable() -> None:
    import yaml

    manifest = yaml.safe_load((NODE_DIR / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["id"] == "transcription/nemotron_asr_streaming_06b"
    assert manifest["version"] == "v1"
    assert "stage" in manifest
