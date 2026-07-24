"""Helpers for stable atomic and bundle pipeline identities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

EVALUATION_ROOT = Path(__file__).resolve().parent
NODES_ROOT = EVALUATION_ROOT / "nodes"
CONVERSION_ROOT = EVALUATION_ROOT / "conversion"

NODE_MANIFEST_ALIASES = {
    "scoring/wenet_cer": "scoring/wenet_wer",
    "scoring/wenet_mer": "scoring/wenet_wer",
}

METRIC_ALIASES = {
    "cer_canonical": "cer",
    "wer_canonical": "wer",
    "mer_canonical": "mer",
    "tts_cer": "cer",
    "tts_wer": "wer",
    "vc_cer": "cer",
    "vc_wer": "wer",
    "tse_cer": "cer",
    "tse_wer": "wer",
    "sim/wavlm-large": "spk_sim",
    "sim/ecapa-tdnn": "spk_sim",
    "sim/eres2net": "spk_sim",
    "sim_wavlm_large": "spk_sim",
    "sim_ecapa_tdnn": "spk_sim",
    "sim_eres2net": "spk_sim",
    "wv-mos": "wv_mos",
    "wvmos": "wv_mos",
    "si-sdr": "si_sdr",
    "macro-recall": "macro_recall",
}


@dataclass(frozen=True)
class PipelineComponent:
    """One score-affecting component in a pipeline identity."""

    component_id: str
    kind: Literal["node", "conversion"] = "node"
    profile: str | None = None
    version: str | None = None


def slug(value: object) -> str:
    """Normalize a display token for use in pipeline IDs."""

    normalized = str(value).strip().lower()
    if normalized in {"", "n/a", "none", "null"}:
        return "any"
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "any"


def language_slug(language: str | None) -> str:
    return slug(language or "any")


def canonical_metric(metric: str) -> str:
    normalized = str(metric).strip().lower()
    return slug(METRIC_ALIASES.get(normalized, normalized))


def node_component(
    node_id: str,
    *,
    profile: str | None = None,
    version: str | None = None,
) -> PipelineComponent:
    return PipelineComponent(component_id=node_id, kind="node", profile=profile, version=version)


def conversion_component(
    conversion_id: str,
    *,
    version: str | None = None,
) -> PipelineComponent:
    return PipelineComponent(component_id=conversion_id, kind="conversion", version=version)


def build_atomic_pipeline_id(
    task: str,
    language: str | None,
    metric: str,
    components: tuple[PipelineComponent, ...] | list[PipelineComponent],
) -> str:
    parts = [slug(task), language_slug(language), canonical_metric(metric)]
    parts.extend(component_instance_id(component) for component in components)
    return ".".join(parts)


def build_bundle_pipeline_id(
    task: str,
    language: str | None,
    member_pipeline_ids: tuple[str, ...] | list[str],
) -> str:
    metric_part = "__".join(_member_pipeline_tail(task, language, pipeline_id) for pipeline_id in member_pipeline_ids)
    return ".".join([slug(task), language_slug(language), "multi", metric_part])


def _member_pipeline_tail(task: str, language: str | None, pipeline_id: str) -> str:
    prefix = f"{slug(task)}.{language_slug(language)}."
    if pipeline_id.startswith(prefix):
        return pipeline_id[len(prefix) :]
    return slug(pipeline_id)


def component_instance_id(component: PipelineComponent) -> str:
    base = _component_base_name(component)
    version = _component_version(component)
    profile = slug(component.profile) if component.profile else ""
    pieces = [base]
    if profile:
        pieces.append(profile)
    pieces.append(version)
    return "_".join(pieces)


def component_trace_id(component: PipelineComponent) -> str:
    if component.kind == "conversion":
        return f"conversion/{component.component_id}"
    return component.component_id


def component_trace_ids(components: tuple[PipelineComponent, ...] | list[PipelineComponent]) -> tuple[str, ...]:
    return tuple(component_trace_id(component) for component in components)


def _component_base_name(component: PipelineComponent) -> str:
    if component.kind == "conversion":
        return f"conversion_{slug(component.component_id)}"
    return slug(component.component_id.split("/", 1)[-1])


def _component_version(component: PipelineComponent) -> str:
    version = component.version or _manifest_version(component)
    normalized = slug(version)
    return normalized if normalized.startswith("v") else f"v{normalized}"


def _manifest_version(component: PipelineComponent) -> str:
    path = _manifest_path(component)
    if not path.exists():
        return "v1"
    with path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    return str(manifest.get("version") or "v1")


def _manifest_path(component: PipelineComponent) -> Path:
    if component.kind == "conversion":
        return CONVERSION_ROOT / component.component_id / "manifest.yaml"
    node_id = NODE_MANIFEST_ALIASES.get(component.component_id, component.component_id)
    stage, name = node_id.split("/", 1)
    return NODES_ROOT / stage / name / "manifest.yaml"
