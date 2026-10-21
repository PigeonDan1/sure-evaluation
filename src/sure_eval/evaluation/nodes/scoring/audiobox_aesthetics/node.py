"""AudioBox Aesthetics CE, CU, PC, and PQ scoring."""

from __future__ import annotations
import argparse
from contextlib import contextmanager, redirect_stdout
import json
import os
import sys
from pathlib import Path
from statistics import fmean
from sure_eval.evaluation.core.types import PipelineNodeResult

NODE_ID = "scoring/audiobox_aesthetics"
NODE_VERSION = "v1"
NODE_DIR = Path(__file__).resolve().parent
AXES = ("CE", "CU", "PC", "PQ")


def score_audiobox_aesthetics(rows, *, provider) -> PipelineNodeResult:
    if not rows:
        raise ValueError("AudioBox Aesthetics scoring requires at least one row")
    raw = provider(rows)
    if len(raw) != len(rows):
        raise RuntimeError(f"AudioBox provider returned {len(raw)} results for {len(rows)} rows")
    axes = {}
    for axis in AXES:
        per_sample = []
        for (key, _path), result in zip(rows, raw):
            if axis not in result:
                raise KeyError(f"AudioBox result is missing axis {axis}")
            per_sample.append({"sample_id": key, "score": float(result[axis])})
        axes[axis] = {
            "score": fmean(x["score"] for x in per_sample),
            "num_samples": len(rows),
            "per_sample": per_sample,
        }
    return PipelineNodeResult(
        stage="scoring",
        node_id=NODE_ID,
        version=NODE_VERSION,
        details={"axes": axes},
        internal_stages=("aesthetic_inference", "axis_projection", "mean_aggregation"),
    )


@contextmanager
def _use_soundfile_torchaudio_load():
    import soundfile
    import torch
    import torchaudio

    original_load = torchaudio.load

    def soundfile_load(uri, *args, **kwargs):
        if args or kwargs:
            raise TypeError("AudioBox managed loader only accepts an audio path")
        samples, sample_rate = soundfile.read(uri, dtype="float32", always_2d=True)
        return torch.from_numpy(samples.T.copy()), sample_rate

    torchaudio.load = soundfile_load
    try:
        yield
    finally:
        torchaudio.load = original_load


class AudioBoxProvider:
    def __init__(self, batch_size=32):
        from audiobox_aesthetics.infer import initialize_predictor

        configured = os.environ.get("AUDIOBOX_AESTHETICS_CHECKPOINT")
        candidates = []
        if configured:
            candidates.append(Path(configured).expanduser())
        candidates.append(NODE_DIR / "checkpoints" / "checkpoint.pt")
        checkpoint = next((path for path in candidates if path.is_file()), None)
        if checkpoint is not None:
            print(
                f"[AudioBox] loading local checkpoint: {checkpoint}",
                file=sys.stderr,
                flush=True,
            )
            self.predictor = initialize_predictor(str(checkpoint))
        else:
            print(
                "[AudioBox] local checkpoint unavailable; using upstream online fallback",
                file=sys.stderr,
                flush=True,
            )
            self.predictor = initialize_predictor()
        self.batch_size = batch_size

    def __call__(self, rows):
        from tqdm.auto import tqdm

        output = []
        starts = range(0, len(rows), self.batch_size)
        total_batches = (len(rows) + self.batch_size - 1) // self.batch_size
        with _use_soundfile_torchaudio_load():
            for start in tqdm(
                starts,
                total=total_batches,
                desc="AudioBox scoring",
                unit="batch",
                dynamic_ncols=True,
                file=sys.stderr,
            ):
                output.extend(
                    self.predictor.forward(
                        [{"path": path} for _key, path in rows[start : start + self.batch_size]]
                    )
                )
        return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)
    rows = []
    for line in Path(args.input_jsonl).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append((str(row["key"]), str(row["prediction_audio"])))
    if args.json_output:
        with redirect_stdout(sys.stderr):
            trace = score_audiobox_aesthetics(rows, provider=AudioBoxProvider(args.batch_size))
    else:
        trace = score_audiobox_aesthetics(rows, provider=AudioBoxProvider(args.batch_size))
    print(json.dumps({"node_id": NODE_ID, "version": NODE_VERSION, "axes": trace.details["axes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
