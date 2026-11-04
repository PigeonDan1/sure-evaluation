"""Auditable timing scores for the fixed FD-Bench compatibility protocol.

This implementation is organized independently around per-user slots and an
event ledger. The operational reference is identified in ``EVALUATION_CONFIG``;
no upstream code, model runtime, or audio detector is imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from statistics import median
from typing import Any

from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.normalization.duplex_timeline.types import (
    ConversationTimeline,
    DuplexTimelineBundle,
    SpeechSegment,
)

NODE_ID = "scoring/fdbench_timing"
NODE_VERSION = "v1"
PROTOCOL = "fdbench_8a4b7df_compat"
SUPPORTED_METRICS = ("ird", "fsed", "ert", "eit")
TICKS_PER_SECOND = 16000
EARLY_THRESHOLD_TICKS = 8000
TICKS_PER_MS = 16
ARRAYS_BY_METRIC = {
    "ird": ("Interrupt Delay",),
    "fsed": ("Response Delay", "Response Delay to Interruption"),
    "ert": ("Lead Times", "Lead Times to Interruption"),
    "eit": ("Lead Times Interrupt",),
}
INTERNAL_STAGES = (
    "exact_tick_conversion",
    "protocol_event_traversal",
    "candidate_slot_selection",
    "positive_tick_filtering",
    "integer_millisecond_conversion",
    "pooled_event_median",
    "event_ledger_diagnostics",
)
EVALUATION_CONFIG = {
    "protocol": PROTOCOL,
    "source_commit": "8a4b7df1b4dcb0fc50a7a5660247a2ffe8d394eb",
    "source_file": "benchmark/benchmarking.py",
    "source_sha256": "86a515a2c86fcd9ae89da9c96b2a4dd461b5d41a68dc752f5d3d719b114ee979",
    "source_method": "analyze_VAD_interruption_new2",
    "input_time_unit": "seconds",
    "time_origin": "conversation_start",
    "ticks_per_second": TICKS_PER_SECOND,
    "early_threshold_ticks": EARLY_THRESHOLD_TICKS,
    "early_threshold_rule": "system.start < user.end - 8000 is EIT; equality is ERT",
    "endpoint_conversion": "exact integer 16k tick; off-grid endpoints rejected",
    "aggregation": "median over pooled positive raw deltas after per-sample ms truncation",
    "millisecond_conversion": "positive delta_ticks // 16 before median",
    "empty_sample_policy": "score=0.0, sample_count=0, status=no_events",
    "preamble_policy": "no implicit segment deletion or model-brand inference",
    "duration_policy": "validation boundary only; no censoring or latency penalty",
    "matching_policy": "ordered per-user traversal; system segments are not consumed",
}


@dataclass(frozen=True)
class _TickSegment:
    id: str
    start: int
    end: int


def score_fdbench_timing(
    bundle: DuplexTimelineBundle,
    *,
    metric: str = "ird",
    protocol: str = PROTOCOL,
) -> PipelineNodeResult:
    """Compute all four timing metrics once and select one primary score.

    The compatibility protocol deliberately retains non-intuitive source
    behavior: special IRD endpoints, ordinary ERT overwrites, strict boundary
    inequalities, and the distinct final-user rule. See the node README.
    """

    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported full-duplex metric: {metric!r}")
    if protocol != PROTOCOL:
        raise ValueError(f"Unsupported FD-Bench protocol: {protocol!r}; expected {PROTOCOL!r}")
    if not bundle.conversations:
        raise ValueError("FD-Bench scoring requires at least one conversation")

    pooled: dict[str, list[int]] = {name: [] for name in SUPPORTED_METRICS}
    per_conversation: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    for conversation in sorted(bundle.conversations, key=lambda item: item.key):
        row, events = _score_conversation(conversation)
        per_conversation.append(row)
        all_events.extend(events)
        for name in SUPPORTED_METRICS:
            pooled[name].extend(row["samples_ticks"][name])

    results = {name: _summarize(pooled[name]) for name in SUPPORTED_METRICS}
    return PipelineNodeResult(
        stage="scoring",
        node_id=NODE_ID,
        version=NODE_VERSION,
        internal_stages=INTERNAL_STAGES,
        details={
            "score": results[metric]["score"],
            "metric": metric,
            "unit": "ms",
            "protocol": PROTOCOL,
            "evaluation_config": dict(EVALUATION_CONFIG),
            "results": results,
            "conversation_count": len(per_conversation),
            "user_segment_count": sum(len(item.user_segments) for item in bundle.conversations),
            "system_segment_count": sum(len(item.system_segments) for item in bundle.conversations),
            "per_conversation": per_conversation,
            "events": all_events,
        },
    )


def _tick(value: Decimal, *, location: str) -> int:
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise ValueError(f"{location}: expected a finite nonnegative Decimal timestamp")
    # Integer ratios avoid rounding even if a JSON literal exceeds the current
    # Decimal context precision. No float or context-sensitive multiply is used.
    numerator, denominator = value.as_integer_ratio()
    ticks, remainder = divmod(numerator * TICKS_PER_SECOND, denominator)
    if remainder:
        raise ValueError(f"{location}: timestamp {value} is off the exact 16000 tick/s grid")
    return ticks


def _tick_segment(segment: SpeechSegment, *, key: str, channel: str) -> _TickSegment:
    location = f"conversation {key!r}, {channel} segment {segment.id!r}"
    start = _tick(segment.start, location=f"{location}.start")
    end = _tick(segment.end, location=f"{location}.end")
    if end <= start:
        raise ValueError(f"{location}: end must be greater than start")
    return _TickSegment(segment.id, start, end)


def _score_conversation(
    conversation: ConversationTimeline,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    users = tuple(
        _tick_segment(item, key=conversation.key, channel="user")
        for item in conversation.user_segments
    )
    systems = tuple(
        _tick_segment(item, key=conversation.key, channel="system")
        for item in conversation.system_segments
    )
    if not users:
        raise ValueError(f"conversation {conversation.key!r}: user_segments must not be empty")
    array_names = tuple(name for names in ARRAYS_BY_METRIC.values() for name in names)
    slots: dict[str, list[Any]] = {name: [0] * len(users) for name in array_names}
    selected: dict[tuple[str, int], list[int]] = {}
    events: list[dict[str, Any]] = []

    def record(
        user_index: int,
        system: _TickSegment,
        branch: str,
        *,
        target: str | None = None,
        delta: int | None = None,
        endpoints: tuple[str, str] | None = None,
        action: str = "skip",
        reason: str,
    ) -> None:
        user = users[user_index]
        event_index = len(events)
        metric_name = next(
            (name for name, arrays in ARRAYS_BY_METRIC.items() if target in arrays), None
        )
        event: dict[str, Any] = {
            "event_id": f"{conversation.key}:{event_index}",
            "conversation_key": conversation.key,
            "user_segment_id": user.id,
            "system_segment_id": system.id,
            "system_role": (
                "speech_overlapping_user_start"
                if system.start < user.start < system.end
                else "candidate_response"
            ),
            "branch_id": branch,
            "user_start_tick": user.start,
            "user_end_tick": user.end,
            "system_start_tick": system.start,
            "system_end_tick": system.end,
            "difference_endpoints": list(endpoints) if endpoints else [],
            "raw_tick_delta": delta,
            "sample_ms": delta // TICKS_PER_MS if delta is not None and delta > 0 else None,
            "target_array": target,
            "metric": metric_name,
            "action": action,
            "reason": reason,
            "selected": False,
            "exclusion_reason": reason if action == "skip" else None,
        }
        events.append(event)
        if action == "skip" or target is None or delta is None:
            return
        position = (target, user_index)
        if action == "append":
            previous = slots[target][user_index]
            if previous == 0:
                slots[target][user_index] = delta
            elif isinstance(previous, list):
                previous.append(delta)
            else:
                slots[target][user_index] = [previous, delta]
            selected.setdefault(position, []).append(event_index)
        else:
            for previous_index in selected.get(position, []):
                events[previous_index]["exclusion_reason"] = "overwritten_by_later_candidate"
            slots[target][user_index] = delta
            selected[position] = [event_index]

    def assign(
        index: int,
        system: _TickSegment,
        branch: str,
        target: str,
        delta: int,
        endpoints: tuple[str, str],
        *,
        overwrite: bool = False,
    ) -> None:
        occupied = slots[target][index] != 0
        action = "overwrite" if occupied and overwrite else "set"
        if occupied and not overwrite:
            action = "skip"
        record(
            index,
            system,
            branch,
            target=target,
            delta=delta,
            endpoints=endpoints,
            action=action,
            reason=(
                "later_candidate_replaces_slot"
                if action == "overwrite"
                else ("first_candidate_kept" if action == "skip" else "candidate_selected")
            ),
        )

    def response(index: int, system: _TickSegment, *, early: bool, interrupted: bool) -> None:
        user = users[index]
        target = (
            ("Lead Times to Interruption" if interrupted else "Lead Times")
            if early
            else ("Response Delay to Interruption" if interrupted else "Response Delay")
        )
        branch = ("early" if early else "late") + (
            "_interrupted_response" if interrupted else "_ordinary_response"
        )
        delta = user.end - system.start if early else system.start - user.end
        endpoints = ("user.end", "system.start") if early else ("system.start", "user.end")
        assign(index, system, branch, target, delta, endpoints, overwrite=early and not interrupted)
        if interrupted and slots["Interrupt Delay"][index] == 0:
            assign(
                index,
                system,
                "interrupted_response_ird_fallback",
                "Interrupt Delay",
                system.end - user.start,
                ("system.end", "user.start"),
            )

    for index, user in enumerate(users):
        interrupted = False
        last_user = index == len(users) - 1
        for system in systems:
            if system.start < user.start < system.end:
                interrupted = True
                record(index, system, "user_starts_during_system", reason="interrupt_context_set")

            if user.start < system.start < user.end - EARLY_THRESHOLD_TICKS:
                record(
                    index,
                    system,
                    "early_system_interruption",
                    target="Lead Times Interrupt",
                    delta=user.end - system.start,
                    endpoints=("user.end", "system.start"),
                    action="append",
                    reason="all_early_interruption_candidates_kept",
                )
                if system.end < user.end:
                    if slots["Interrupt Delay"][index] == 0:
                        interrupted = True
                    assign(
                        index,
                        system,
                        "early_interruption_ird_duration",
                        "Interrupt Delay",
                        system.end - system.start,
                        ("system.end", "system.start"),
                    )
                continue

            if user.start < system.start < user.end:
                if not interrupted:
                    response(index, system, early=True, interrupted=False)
                elif not (
                    slots["Lead Times to Interruption"][index]
                    or slots["Response Delay to Interruption"][index]
                ):
                    response(index, system, early=True, interrupted=True)
                    interrupted = False
                else:
                    record(
                        index,
                        system,
                        "early_interrupted_response",
                        target="Lead Times to Interruption",
                        delta=user.end - system.start,
                        endpoints=("user.end", "system.start"),
                        reason="interrupted_response_slot_already_filled",
                    )
                continue

            if system.start < user.start:
                eligible_end = user.start < system.end and (last_user or system.end <= user.end)
                if eligible_end and interrupted:
                    assign(
                        index,
                        system,
                        "last_user_old_speech_stop" if last_user else "nonlast_old_speech_stop",
                        "Interrupt Delay",
                        system.end - user.start,
                        ("system.end", "user.start"),
                    )
                else:
                    record(
                        index,
                        system,
                        "old_speech_continue",
                        reason="old_speech_end_outside_source_branch",
                    )
                # This continue intentionally leaves the later upstream
                # non-final old-speech-after-user-end branch unreachable.
                continue

            in_late_window = user.end < system.start and (
                last_user or system.start < users[index + 1].start
            )
            if in_late_window:
                if not interrupted:
                    response(index, system, early=False, interrupted=False)
                elif not (
                    slots["Response Delay to Interruption"][index]
                    or slots["Lead Times to Interruption"][index]
                ):
                    response(index, system, early=False, interrupted=True)
                    interrupted = False
                else:
                    record(
                        index,
                        system,
                        "late_interrupted_response",
                        target="Response Delay to Interruption",
                        delta=system.start - user.end,
                        endpoints=("system.start", "user.end"),
                        reason="interrupted_response_slot_already_filled",
                    )
                continue
            record(
                index,
                system,
                "unselected_system_start",
                reason="strict_boundary_or_outside_response_window",
            )

    for indices in selected.values():
        for event_index in indices:
            event = events[event_index]
            event["selected"] = event["raw_tick_delta"] > 0
            event["exclusion_reason"] = None if event["selected"] else "nonpositive_raw_delta"

    samples: dict[str, list[int]] = {}
    turns_with_samples: dict[str, int] = {}
    for metric_name, names in ARRAYS_BY_METRIC.items():
        samples[metric_name] = [
            value for name in names for value in _flatten(slots[name]) if value > 0
        ]
        turns_with_samples[metric_name] = sum(
            any(value > 0 for name in names for value in _flatten([slots[name][index]]))
            for index in range(len(users))
        )
    return {
        "key": conversation.key,
        "duration_seconds": str(conversation.duration),
        "user_segment_ids": [item.id for item in users],
        "system_segment_ids": [item.id for item in systems],
        "user_segment_count": len(users),
        "system_segment_count": len(systems),
        "raw_slots_ticks": slots,
        "samples_ticks": samples,
        "samples_ms": {
            name: [value // TICKS_PER_MS for value in values] for name, values in samples.items()
        },
        "results": {name: _summarize(values) for name, values in samples.items()},
        "user_segments_with_samples": turns_with_samples,
        "event_count": len(events),
        "boundary_hints": {
            "system_segment_ends_at_observation_boundary": [
                item.id
                for item in conversation.system_segments
                if item.end == conversation.duration
            ],
            "last_user_has_no_next_user_upper_bound": True,
        },
    }, events


def _flatten(values: list[Any]) -> list[int]:
    result: list[int] = []
    for value in values:
        if isinstance(value, list):
            result.extend(value)
        else:
            result.append(value)
    return result


def _summarize(ticks: list[int]) -> dict[str, Any]:
    samples_ms = [value // TICKS_PER_MS for value in ticks]
    return {
        "score": float(median(samples_ms)) if samples_ms else 0.0,
        "unit": "ms",
        "sample_count": len(samples_ms),
        "positive_tick_count": len(ticks),
        "zero_ms_count": samples_ms.count(0),
        "status": "ok" if samples_ms else "no_events",
        "reason": (
            "pooled_selected_positive_raw_deltas"
            if samples_ms
            else "no_selected_positive_raw_delta; zero is the compatibility empty-set value"
        ),
        "samples_ticks": ticks,
        "samples_ms": samples_ms,
    }
