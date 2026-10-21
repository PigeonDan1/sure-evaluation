"""Text-to-audio distribution, alignment, and aesthetic evaluation."""

from __future__ import annotations
from collections.abc import Iterable, Mapping
from typing import Any
from sure_eval.evaluation.core.types import EvaluationFiles, EvaluationReport
from sure_eval.evaluation.pipeline_identity import (
    build_atomic_pipeline_id,
    build_bundle_pipeline_id,
    canonical_metric,
    component_trace_ids,
    node_component,
)
from sure_eval.evaluation.tasks.tta.types import TTASample

DEFAULT_METRICS = (
    "fad",
    "fd",
    "kl",
    "is",
    "clap_score",
    "audiobox_ce",
    "audiobox_cu",
    "audiobox_pc",
    "audiobox_pq",
)
_DIST = {"fad", "fd", "kl", "is"}
_AES = {"audiobox_ce", "audiobox_cu", "audiobox_pc", "audiobox_pq"}
_NODES = {
    "fad": "scoring/audioldm_eval",
    "fd": "scoring/audioldm_eval",
    "kl": "scoring/audioldm_eval",
    "is": "scoring/audioldm_eval",
    "clap_score": "scoring/clap_score",
    "audiobox_ce": "scoring/audiobox_aesthetics",
    "audiobox_cu": "scoring/audiobox_aesthetics",
    "audiobox_pc": "scoring/audiobox_aesthetics",
    "audiobox_pq": "scoring/audiobox_aesthetics",
}


def evaluate_tta_samples(
    samples: list[TTASample],
    *,
    metrics: Iterable[str] | None = None,
    providers: Mapping[str, Any] | None = None,
) -> EvaluationReport:
    if not samples:
        raise ValueError("at least one TTA sample is required")
    requested = tuple(_norm(x) for x in (metrics or DEFAULT_METRICS))
    bad = [x for x in requested if x not in _NODES]
    if bad:
        raise ValueError(f"Unsupported TTA metric(s): {', '.join(bad)}")
    if len(set(requested)) != len(requested):
        raise ValueError("TTA metrics must not contain duplicates")
    _validate(samples, requested)
    providers = dict(providers or {})
    results, traces = {}, []
    dist = tuple(x for x in requested if x in _DIST)
    if dist:
        from sure_eval.evaluation.nodes.scoring.audioldm_eval.node import score_audio_distribution

        trace = score_audio_distribution(
            [(s.sample_id or "", s.prediction_audio, s.reference_audio) for s in samples],
            metrics=dist,
            provider=_provider(providers, "audioldm_eval"),
        )
        traces.append(trace)
        for metric in dist:
            results[metric] = _result(
                metric, trace.details["metrics"][metric], len(samples), "corpus_distribution"
            )
            if metric == "is" and "inception_score_std" in trace.details["raw_metrics"]:
                results[metric]["score_std"] = float(
                    trace.details["raw_metrics"]["inception_score_std"]
                )
    if "clap_score" in requested:
        from sure_eval.evaluation.nodes.scoring.clap_score.node import score_clap

        trace = score_clap(
            [(s.sample_id or "", s.prediction_audio, s.caption) for s in samples],
            provider=_provider(providers, "clap_score"),
        )
        traces.append(trace)
        results["clap_score"] = dict(trace.details["result"])
    aes = tuple(x for x in requested if x in _AES)
    if aes:
        from sure_eval.evaluation.nodes.scoring.audiobox_aesthetics.node import (
            score_audiobox_aesthetics,
        )

        trace = score_audiobox_aesthetics(
            [(s.sample_id or "", s.prediction_audio) for s in samples],
            provider=_provider(providers, "audiobox_aesthetics"),
        )
        traces.append(trace)
        for metric in aes:
            axis = metric.rsplit("_", 1)[1].upper()
            block = trace.details["axes"][axis]
            results[metric] = _result(
                metric, block["score"], len(samples), "mean", block["per_sample"]
            )
    ids, nodes = {}, {}
    for metric in requested:
        components = (node_component(_NODES[metric]),)
        ids[metric] = build_atomic_pipeline_id("tta", "any", metric, components)
        nodes[metric] = component_trace_ids(components)
        results[metric].update(pipeline_id=ids[metric], computation_node_ids=list(nodes[metric]))
    if len(requested) == 1:
        metric, pipeline_id, kind, members, computation = (
            requested[0],
            ids[requested[0]],
            "atomic",
            (),
            nodes[requested[0]],
        )
    else:
        metric, members = "multi", tuple(ids[x] for x in requested)
        pipeline_id, kind = build_bundle_pipeline_id("tta", "any", members), "bundle"
        computation = tuple(node for x in requested for node in nodes[x])
    inputs = _inputs(samples)
    return EvaluationReport(
        task="TTA",
        language="n/a",
        metric=canonical_metric(metric),
        score=float(results[requested[0]]["score"]),
        pipeline_id=pipeline_id,
        pipeline_trace=tuple(traces),
        input_files=inputs,
        pipeline_kind=kind,
        member_pipeline_ids=members,
        computation_node_ids=computation,
        details={
            "results": results,
            "rows": [_row(s) for s in samples],
            "input_files": inputs.as_dict(),
        },
    )


def _provider(providers, name):
    if name not in providers:
        raise RuntimeError(f"TTA metric requires a '{name}' provider")
    return providers[name]


def _validate(samples, metrics):
    seen = set()
    for index, sample in enumerate(samples, 1):
        key = sample.sample_id or f"sample{index}"
        if key in seen:
            raise ValueError(f"duplicate TTA sample_id: {key}")
        seen.add(key)
        if not sample.prediction_audio:
            raise ValueError(f"TTA sample {key} requires prediction_audio")
        if set(metrics) & _DIST and not sample.reference_audio:
            raise ValueError(f"TTA sample {key} requires reference_audio for FAD/FD/KL/IS")
        if "clap_score" in metrics and not sample.caption:
            raise ValueError(f"TTA sample {key} requires caption for CLAP score")


def _result(metric, score, count, aggregation, per_sample=None):
    value = {
        "metric_name": metric,
        "score": float(score),
        "num_samples": count,
        "aggregation": aggregation,
    }
    if per_sample is not None:
        value["per_sample"] = per_sample
    return value


def _inputs(samples):
    first = samples[0]
    roles = {"prediction_audio": first.prediction_audio if len(samples) == 1 else "batch"}
    if first.reference_audio:
        roles["reference_audio"] = first.reference_audio if len(samples) == 1 else "batch"
    if first.caption:
        roles["caption"] = "inline"
    return EvaluationFiles(roles=roles)


def _row(sample):
    return {
        "sample_id": sample.sample_id,
        "prediction_audio": sample.prediction_audio,
        "reference_audio": sample.reference_audio,
        "caption": sample.caption,
        "metadata": dict(sample.metadata),
    }


def _norm(metric):
    value = str(metric).strip().lower().replace("-", "_")
    return {
        "inception_score": "is",
        "clap": "clap_score",
        "ce": "audiobox_ce",
        "cu": "audiobox_cu",
        "pc": "audiobox_pc",
        "pq": "audiobox_pq",
    }.get(value, value)
