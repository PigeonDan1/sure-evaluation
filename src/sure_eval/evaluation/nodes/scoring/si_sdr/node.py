"""SI-SDR scoring node for SE and TSE."""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path
from statistics import fmean
from typing import Any

from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.scoring._full_reference_audio import (
    FullReferenceAudioProvider,
    FullReferenceAudioRow,
    SISDRProvider,
    score_full_reference_audio_backend,
)

NODE_ID = "scoring/si_sdr"
NODE_VERSION = "v1"
SI_SDR_INTERNAL_STAGES = ("audio_load", "length_align", "si_sdr_compute", "mean_aggregation")

SignalRow = tuple[str, str, str]


def score_si_sdr(
    rows: list[SignalRow] | list[FullReferenceAudioRow],
    *,
    provider: FullReferenceAudioProvider | None = None,
    mixed_paths: list[str] | None = None,
) -> PipelineNodeResult:
    """Score SI-SDR.

    SE injects a full-reference provider and uses the provider-backed path.
    TSE uses the native path so it can optionally report SI-SDRi from mixtures.
    """

    if provider is not None and mixed_paths is None:
        return score_full_reference_audio_backend(
            rows,
            metric_name="si-sdr",
            node_id=NODE_ID,
            provider=provider,
            version=NODE_VERSION,
        )
    return _score_si_sdr_native(rows, mixed_paths=mixed_paths)


def build_default_provider() -> FullReferenceAudioProvider:
    return SISDRProvider()


def _load_audio(path: str, *, target_sr: int | None = None) -> tuple[Any, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=False, dtype="float64")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        import scipy.signal

        data = scipy.signal.resample_poly(data, target_sr, sr)
        sr = target_sr
    return data, sr


def _si_sdr(reference: Any, prediction: Any) -> float:
    reference = reference.astype("float64")
    prediction = prediction.astype("float64")
    min_len = min(len(reference), len(prediction))
    reference = reference[:min_len] - reference[:min_len].mean()
    prediction = prediction[:min_len] - prediction[:min_len].mean()
    reference_energy = float(reference @ reference)
    if reference_energy < 1e-12:
        return 0.0
    scale = float(reference @ prediction) / reference_energy
    s_target = scale * reference
    e_noise = prediction - s_target
    s_target_energy = float(s_target @ s_target)
    e_noise_energy = float(e_noise @ e_noise)
    if e_noise_energy < 1e-20:
        return float("inf")
    ratio = s_target_energy / e_noise_energy
    if ratio < 1e-20:
        return -float("inf")
    return float(10.0 * math.log10(ratio))


def _score_si_sdr_native(
    rows: list[SignalRow] | list[FullReferenceAudioRow],
    *,
    mixed_paths: list[str] | None = None,
) -> PipelineNodeResult:
    if not rows:
        raise ValueError("SI-SDR scoring requires at least one row")
    if mixed_paths is not None and len(mixed_paths) != len(rows):
        raise ValueError("mixed_paths must have the same length as rows")

    per_sample: list[dict[str, Any]] = []
    si_sdri_scores: list[float] = []
    for index, (key, prediction_path, reference_path) in enumerate(rows):
        prediction_data, pred_sr = _load_audio(prediction_path)
        reference_data, ref_sr = _load_audio(reference_path)
        target_sr = None if pred_sr == ref_sr else min(pred_sr, ref_sr)
        if target_sr is not None:
            prediction_data, _ = _load_audio(prediction_path, target_sr=target_sr)
            reference_data, _ = _load_audio(reference_path, target_sr=target_sr)
        raw_score = _si_sdr(reference_data, prediction_data)
        entry = _normalize_si_sdr_row(raw_score)

        mixed_path = mixed_paths[index] if mixed_paths is not None else ""
        if mixed_path:
            mixed_data, _ = _load_audio(mixed_path, target_sr=target_sr)
            mixture_score = _si_sdr(reference_data, mixed_data)
            improvement = _si_sdri(raw_score, mixture_score)
            entry["si_sdri"] = improvement
            si_sdri_scores.append(improvement)

        per_sample.append(entry)

    finite_scores = [row["si_sdr"] for row in per_sample if not math.isinf(row["si_sdr"])]
    mean_score = fmean(finite_scores) if finite_scores else float("inf")
    result: dict[str, Any] = {
        "metric_name": "si_sdr",
        "score": mean_score,
        "num_samples": len(rows),
        "score_key": "si_sdr",
        "per_sample": per_sample,
    }
    if si_sdri_scores:
        finite_si_sdri = [value for value in si_sdri_scores if not math.isinf(value)]
        result["si_sdri"] = fmean(finite_si_sdri) if finite_si_sdri else float("inf")
    return PipelineNodeResult(
        stage="scoring",
        node_id=NODE_ID,
        version=NODE_VERSION,
        details={
            "metric": "si_sdr",
            "keys": [key for key, _, _ in rows],
            "result": result,
        },
        internal_stages=SI_SDR_INTERNAL_STAGES,
    )


def _normalize_si_sdr_row(raw_score: float) -> dict[str, Any]:
    if math.isinf(raw_score):
        return {"si_sdr": raw_score}
    return {"si_sdr": float(raw_score)}


def _si_sdri(prediction_score: float, mixture_score: float) -> float:
    if math.isnan(prediction_score) or math.isnan(mixture_score):
        return 0.0
    diff = prediction_score - mixture_score
    if math.isnan(diff):
        return 0.0
    return float(diff)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score SI-SDR between predicted and reference audio."
    )
    parser.add_argument("--prediction-audio")
    parser.add_argument("--reference-audio")
    parser.add_argument("--mixed-audio")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    has_pair_arg = bool(args.prediction_audio or args.reference_audio or args.mixed_audio)
    has_complete_pair = bool(args.prediction_audio and args.reference_audio)
    if (args.input_jsonl and has_pair_arg) or (not args.input_jsonl and not has_complete_pair):
        parser.error("use either --input-jsonl or --prediction-audio plus --reference-audio")

    if args.input_jsonl:
        rows, mixed_paths, provider_mode = _read_rows(args.input_jsonl)
        if args.json_output:
            with redirect_stdout(sys.stderr):
                trace = (
                    score_si_sdr(rows, provider=build_default_provider())
                    if provider_mode
                    else score_si_sdr(rows, mixed_paths=mixed_paths if any(mixed_paths) else None)
                )
        else:
            trace = (
                score_si_sdr(rows, provider=build_default_provider())
                if provider_mode
                else score_si_sdr(rows, mixed_paths=mixed_paths if any(mixed_paths) else None)
            )
        payload = {
            "node_id": NODE_ID,
            "version": NODE_VERSION,
            "result": trace.details["result"],
            "keys": trace.details["keys"],
        }
        if args.json_output:
            sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        else:
            score = payload["result"]["score"]
            print(score if not math.isinf(score) else "inf")
        return 0

    if args.mixed_audio:
        rows = [("single", args.prediction_audio, args.reference_audio)]
        trace = score_si_sdr(rows, mixed_paths=[args.mixed_audio])
        payload = {
            "node_id": NODE_ID,
            "version": NODE_VERSION,
            "prediction_audio": args.prediction_audio,
            "reference_audio": args.reference_audio,
            "result": trace.details["result"],
        }
    else:
        result = build_default_provider()(args.prediction_audio, args.reference_audio)
        payload = {
            "node_id": NODE_ID,
            "version": NODE_VERSION,
            "prediction_audio": args.prediction_audio,
            "reference_audio": args.reference_audio,
            "result": result,
        }
    if args.json_output:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    else:
        score = payload["result"].get("score", payload["result"].get("si_sdr", ""))
        print(score if isinstance(score, float) and math.isinf(score) else score)
    return 0


def _read_rows(path: str) -> tuple[list[SignalRow], list[str], bool]:
    rows: list[SignalRow] = []
    mixed_paths: list[str] = []
    provider_mode = True
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prediction_audio = str(row.get("prediction_audio", row.get("enhanced_audio", "")))
        key = str(row.get("key") or row.get("sample_id") or "")
        rows.append((key, prediction_audio, str(row["reference_audio"])))
        mixed_audio = str(row.get("mixed_audio") or "")
        mixed_paths.append(mixed_audio)
        if row.get("prediction_audio") or mixed_audio:
            provider_mode = False
    return rows, mixed_paths, provider_mode


if __name__ == "__main__":
    raise SystemExit(main())
