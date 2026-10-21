"""FAD, FD, KL, and Inception Score through AudioLDM evaluation."""

from __future__ import annotations
import argparse
from contextlib import contextmanager, redirect_stdout
import inspect
import json
import os
import tempfile
import sys
import zipfile
from pathlib import Path
from sure_eval.evaluation.core.types import PipelineNodeResult

NODE_ID = "scoring/audioldm_eval"
NODE_VERSION = "v1"
NODE_DIR = Path(__file__).resolve().parent
_TORCHVGGISH_REVISION = "f3f90969ce8fb77a05d984b892ddf662e8466377"
_TORCHVGGISH_ARCHIVE = (
    NODE_DIR / "checkpoints" / "torchvggish" / (f"torchvggish-{_TORCHVGGISH_REVISION}.zip")
)
_TORCH_HUB_DIR = NODE_DIR / "checkpoints" / "runtime_home" / ".cache" / "torch" / "hub"
_TORCHVGGISH_REPO = _TORCH_HUB_DIR / "harritaylor_torchvggish_master"
_VGGISH_CHECKPOINTS = (
    _TORCH_HUB_DIR / "checkpoints" / "vggish-10086976.pth",
    _TORCH_HUB_DIR / "checkpoints" / "vggish_pca_params-970ea276.pth",
)
_RAW_KEYS = {
    "fad": "frechet_audio_distance",
    "fd": "frechet_distance",
    "kl": "kullback_leibler_divergence_softmax",
    "is": "inception_score_mean",
}


def score_audio_distribution(rows, *, metrics, provider) -> PipelineNodeResult:
    if not rows:
        raise ValueError("AudioLDM evaluation requires at least one row")
    raw = provider(rows, metrics=tuple(metrics))
    values = {}
    for metric in metrics:
        key = _RAW_KEYS[metric]
        if metric in raw:
            values[metric] = float(raw[metric])
        elif key in raw:
            values[metric] = float(raw[key])
        else:
            raise KeyError(f"AudioLDM evaluator did not return {key!r}")
    return PipelineNodeResult(
        stage="scoring",
        node_id=NODE_ID,
        version=NODE_VERSION,
        details={"metrics": values, "raw_metrics": dict(raw), "keys": [row[0] for row in rows]},
        internal_stages=("audio_embedding", "distribution_statistics", "metric_aggregation"),
    )


class AudioLDMEvalProvider:
    def __init__(self, device="cuda", num_workers=4):
        checkpoint = Path(
            os.environ.get(
                "AUDIOLDM_EVAL_CNN14_CHECKPOINT",
                NODE_DIR / "checkpoints" / "audioldm_eval" / "ckpt" / "Cnn14_16k_mAP=0.438.pth",
            )
        ).expanduser()
        companion = NODE_DIR / "checkpoints" / "audioldm_eval" / "ckpt" / "Cnn14_mAP=0.431.pth"
        if not checkpoint.is_file():
            raise RuntimeError(
                "AudioLDM 16 kHz CNN14 checkpoint is missing; run "
                "sure-eval env setup --node scoring/audioldm_eval"
            )
        if not companion.is_file():
            raise RuntimeError(
                "AudioLDM CNN14 companion checkpoint is missing; run "
                "sure-eval env setup --node scoring/audioldm_eval"
            )
        os.environ.setdefault("AUDIOLDM_EVAL_CACHE", str(checkpoint.parent.parent))
        self.checkpoint = checkpoint
        self.companion_checkpoint = companion
        self.device, self.num_workers = device, num_workers

    def __call__(self, rows, *, metrics):
        with _audioldm_runtime_home(self.checkpoint, self.companion_checkpoint):
            _prepare_torchvggish_cache()
            with _use_local_torchvggish(), _use_trusted_legacy_torch_load(self.device):
                from audioldm_eval import EvaluationHelper

                try:
                    evaluator = EvaluationHelper(16000, self.device, backbone="cnn14")
                except TypeError:
                    evaluator = EvaluationHelper(16000, self.device)

        with (
            tempfile.TemporaryDirectory(prefix="sure-tta-gen-") as gen,
            tempfile.TemporaryDirectory(prefix="sure-tta-ref-") as ref,
        ):
            for index, (key, prediction, reference) in enumerate(rows):
                suffix = Path(prediction).suffix or ".wav"
                name = f"{index:08d}_{_safe_name(key)}{suffix}"
                os.symlink(Path(prediction).resolve(), Path(gen) / name)
                os.symlink(Path(reference).resolve(), Path(ref) / name)
            parameters = inspect.signature(evaluator.main).parameters
            kwargs = {}
            if "recalculate" in parameters:
                kwargs["recalculate"] = True
            if "num_workers" in parameters:
                kwargs["num_workers"] = self.num_workers
            if "backbone" in parameters:
                kwargs["backbone"] = "cnn14"
            with _use_soundfile_torchaudio_load():
                return evaluator.main(gen, ref, **kwargs)


@contextmanager
def _audioldm_runtime_home(checkpoint: Path, companion_checkpoint: Path):
    """Expose managed checkpoints at the legacy paths used by upstream AudioLDM Eval."""

    runtime_home = NODE_DIR / "checkpoints" / "runtime_home"
    legacy_dir = runtime_home / ".cache" / "audioldm_eval" / "ckpt"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    for source in (checkpoint, companion_checkpoint):
        destination = legacy_dir / source.name
        if destination.is_symlink() and not destination.exists():
            destination.unlink()
        if not destination.exists():
            destination.symlink_to(source.resolve())

    previous_home = os.environ.get("HOME")
    os.environ["HOME"] = str(runtime_home)
    try:
        yield
    finally:
        if previous_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = previous_home


def _prepare_torchvggish_cache() -> None:
    """Materialize the pinned Torch Hub repo without contacting GitHub at runtime."""

    missing = [path for path in (_TORCHVGGISH_ARCHIVE, *_VGGISH_CHECKPOINTS) if not path.is_file()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise RuntimeError(
            f"AudioLDM VGGish assets are missing ({names}); run "
            "sure-eval env setup --node scoring/audioldm_eval"
        )
    if _TORCHVGGISH_REPO.is_dir():
        return

    _TORCH_HUB_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="torchvggish-", dir=_TORCH_HUB_DIR) as temp:
        temp_dir = Path(temp)
        with zipfile.ZipFile(_TORCHVGGISH_ARCHIVE) as archive:
            roots = {Path(name).parts[0] for name in archive.namelist() if Path(name).parts}
            if len(roots) != 1:
                raise RuntimeError("unexpected torchvggish archive layout")
            archive.extractall(temp_dir)
        extracted = temp_dir / roots.pop()
        if not (extracted / "hubconf.py").is_file():
            raise RuntimeError("torchvggish archive is missing hubconf.py")
        try:
            extracted.replace(_TORCHVGGISH_REPO)
        except FileExistsError:
            pass


@contextmanager
def _use_trusted_legacy_torch_load(device=None):
    """Load only the checksum-pinned legacy checkpoints with pickle enabled."""

    import torch

    original_load = torch.load

    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        if str(device).startswith("cpu"):
            kwargs.setdefault("map_location", "cpu")
        return original_load(*args, **kwargs)

    torch.load = trusted_load
    try:
        yield
    finally:
        torch.load = original_load


@contextmanager
def _use_soundfile_torchaudio_load():
    """Keep legacy AudioLDM Eval independent of TorchCodec and system FFmpeg."""

    import soundfile
    import torch
    import torchaudio

    original_load = torchaudio.load

    def soundfile_load(uri, *args, **kwargs):
        if args or kwargs:
            raise TypeError("AudioLDM Eval's managed loader only accepts an audio path")
        samples, sample_rate = soundfile.read(uri, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(samples.T.copy())
        return waveform, sample_rate

    torchaudio.load = soundfile_load
    try:
        yield
    finally:
        torchaudio.load = original_load


@contextmanager
def _use_local_torchvggish():
    """Redirect AudioLDM Eval hard-coded Torch Hub request to the local repo."""

    import torch

    original_load = torch.hub.load

    def local_load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir == "harritaylor/torchvggish":
            kwargs["source"] = "local"
            return original_load(str(_TORCHVGGISH_REPO), model, *args, **kwargs)
        return original_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = local_load
    try:
        yield
    finally:
        torch.hub.load = original_load


def _safe_name(value):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(value)) or "sample"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--metrics", default="fad,fd,kl,is")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)
    rows = []
    for line in Path(args.input_jsonl).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append(
                (str(row["key"]), str(row["prediction_audio"]), str(row["reference_audio"]))
            )
    metrics = tuple(x.strip() for x in args.metrics.split(",") if x.strip())
    if args.json_output:
        with redirect_stdout(sys.stderr):
            trace = score_audio_distribution(
                rows, metrics=metrics, provider=AudioLDMEvalProvider(args.device, args.num_workers)
            )
    else:
        trace = score_audio_distribution(
            rows, metrics=metrics, provider=AudioLDMEvalProvider(args.device, args.num_workers)
        )
    print(
        json.dumps(
            {
                "node_id": NODE_ID,
                "version": NODE_VERSION,
                "metrics": trace.details["metrics"],
                "raw_metrics": trace.details["raw_metrics"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
