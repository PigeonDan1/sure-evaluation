"""SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) scoring node."""

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

NODE_ID = "scoring/si_sdr"
NODE_VERSION = "v1"
SI_SDR_INTERNAL_STAGES = ("audio_load", "length_align", "si_sdr_compute", "mean_aggregation")

SignalRow = tuple[str, str, str]


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
    # Zero-mean both signals before projection (Le Roux et al., 2019).
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
    score = 10.0 * math.log10(ratio)
    if score == float("inf"):
        return float("inf")
    return float(score)


def _normalize_si_sdr_row(raw_score: float) -> dict[str, Any]:
    if math.isinf(raw_score):
        return {"si_sdr": raw_score}
    return {"si_sdr": float(raw_score)}


def _si_sdri(prediction_score: float, mixture_score: float) -> float:
    """SI-SDR improvement: prediction SI-SDR minus mixture SI-SDR.

    Same-signed infinities (e.g. both ``+inf``) yield ``0.0`` — when prediction
    and mixture are equally ideal (or equally degenerate) there is no measurable
    improvement. All other infinite combinations fall out of plain subtraction
    (``5 - inf == -inf``, ``5 - -inf == inf``).
    """
    if math.isnan(prediction_score) or math.isnan(mixture_score):
        return 0.0
    diff = prediction_score - mixture_score
    if math.isnan(diff):
        return 0.0
    return float(diff)


def score_si_sdr(
    rows: list[SignalRow],
    *,
    provider: Any = None,
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
    if finite_scores:
        mean_score = fmean(finite_scores)
    else:
        mean_score = float("inf")

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score SI-SDR between predicted and reference audio.")
    parser.add_argument("--prediction-audio")
    parser.add_argument("--reference-audio")
    parser.add_argument("--mixed-audio")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    if bool(args.input_jsonl) == bool(args.prediction_audio and args.reference_audio):
        parser.error("exactly one of --input-jsonl or --prediction-audio/--reference-audio is required")

    if args.input_jsonl:
        input_path = Path(args.input_jsonl)
        rows: list[SignalRow] = []
        mixed_paths: list[str] = []
        for line in input_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append((str(row.get("key", "")), str(row["prediction_audio"]), str(row["reference_audio"])))
            mixed_paths.append(str(row.get("mixed_audio") or ""))
        mixed_arg = mixed_paths if any(mixed_paths) else None
        if args.json_output:
            with redirect_stdout(sys.stderr):
                trace = score_si_sdr(rows, mixed_paths=mixed_arg)
        else:
            trace = score_si_sdr(rows, mixed_paths=mixed_arg)
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

    rows = [("single", args.prediction_audio, args.reference_audio)]
    mixed_arg = [args.mixed_audio] if args.mixed_audio else None
    if args.json_output:
        with redirect_stdout(sys.stderr):
            trace = score_si_sdr(rows, mixed_paths=mixed_arg)
    else:
        trace = score_si_sdr(rows, mixed_paths=mixed_arg)
    payload = {
        "node_id": NODE_ID,
        "version": NODE_VERSION,
        "prediction_audio": args.prediction_audio,
        "reference_audio": args.reference_audio,
        "result": trace.details["result"],
    }
    if args.json_output:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    else:
        score = payload["result"]["per_sample"][0]["si_sdr"]
        print(score if not math.isinf(score) else "inf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())