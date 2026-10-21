"""LAION-CLAP text-audio alignment scoring."""

from __future__ import annotations
import argparse
from contextlib import contextmanager, redirect_stdout
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from sure_eval.evaluation.core.types import PipelineNodeResult

NODE_ID = "scoring/clap_score"
NODE_VERSION = "v1"
NODE_DIR = Path(__file__).resolve().parent
_HF_HOME = NODE_DIR / "checkpoints" / "huggingface"
_ROBERTA_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
_ROBERTA_SNAPSHOT = _HF_HOME / "hub" / "models--roberta-base" / "snapshots" / _ROBERTA_REVISION
_ROBERTA_FILES = ("config.json", "merges.txt", "tokenizer_config.json", "vocab.json")


def score_clap(rows, *, provider) -> PipelineNodeResult:
    if not rows:
        raise ValueError("CLAP scoring requires at least one row")
    raw = provider(rows)
    if len(raw) != len(rows):
        raise RuntimeError(f"CLAP provider returned {len(raw)} scores for {len(rows)} rows")
    per_sample = [{"sample_id": row[0], "score": float(score)} for row, score in zip(rows, raw)]
    result = {
        "metric_name": "clap_score",
        "score": float(np.mean([x["score"] for x in per_sample])),
        "num_samples": len(rows),
        "aggregation": "mean",
        "per_sample": per_sample,
    }
    return PipelineNodeResult(
        stage="scoring",
        node_id=NODE_ID,
        version=NODE_VERSION,
        details={"result": result},
        internal_stages=(
            "audio_embedding",
            "text_embedding",
            "cosine_similarity",
            "mean_aggregation",
        ),
    )


class _UnusedTokenizer:
    def __call__(self, *args, **kwargs):
        raise RuntimeError("LAION-CLAP requested an unused BERT/BART tokenizer")


@contextmanager
def _use_managed_hf_cache():
    missing = [name for name in _ROBERTA_FILES if not (_ROBERTA_SNAPSHOT / name).is_file()]
    local_assets_ready = not missing
    managed_env = {"HF_HOME": str(_HF_HOME)}
    if local_assets_ready:
        managed_env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
    previous_env = {name: os.environ.get(name) for name in managed_env}
    os.environ.update(managed_env)

    import transformers

    original_bert = transformers.BertTokenizer.from_pretrained
    original_bart = transformers.BartTokenizer.from_pretrained
    original_roberta_tokenizer = transformers.RobertaTokenizer.from_pretrained
    original_roberta_model = transformers.RobertaModel.from_pretrained
    transformers.BertTokenizer.from_pretrained = staticmethod(
        lambda *args, **kwargs: _UnusedTokenizer()
    )
    transformers.BartTokenizer.from_pretrained = staticmethod(
        lambda *args, **kwargs: _UnusedTokenizer()
    )

    def roberta_tokenizer_from_cache(model_id, *args, **kwargs):
        kwargs["revision"] = _ROBERTA_REVISION
        kwargs["local_files_only"] = local_assets_ready
        return original_roberta_tokenizer(model_id, *args, **kwargs)

    transformers.RobertaTokenizer.from_pretrained = staticmethod(roberta_tokenizer_from_cache)

    def roberta_from_config(model_id, *args, **kwargs):
        config = transformers.RobertaConfig.from_pretrained(
            model_id, local_files_only=local_assets_ready, revision=_ROBERTA_REVISION
        )
        return transformers.RobertaModel(config)

    transformers.RobertaModel.from_pretrained = staticmethod(roberta_from_config)
    try:
        yield
    finally:
        transformers.BertTokenizer.from_pretrained = original_bert
        transformers.BartTokenizer.from_pretrained = original_bart
        transformers.RobertaTokenizer.from_pretrained = original_roberta_tokenizer
        transformers.RobertaModel.from_pretrained = original_roberta_model
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _log(message: str) -> None:
    print(f"[CLAP] {message}", file=sys.stderr, flush=True)


class CLAPProvider:
    def __init__(self, device="cuda", batch_size=16):
        checkpoint = os.environ.get("CLAP_MODEL_PATH") or str(
            NODE_DIR / "checkpoints" / "630k-audioset-best.pt"
        )
        if not Path(checkpoint).is_file():
            raise RuntimeError("CLAP_MODEL_PATH must point to a LAION-CLAP checkpoint")
        if batch_size < 1:
            raise ValueError("CLAP batch_size must be positive")
        self.batch_size = batch_size

        setup_started = time.monotonic()
        with _use_managed_hf_cache():
            stage_started = time.monotonic()
            _log("setup 1/3: importing laion_clap")
            import laion_clap

            _log(f"setup 1/3 complete ({time.monotonic() - stage_started:.1f}s)")
            stage_started = time.monotonic()
            _log("setup 2/3: building model")
            self.model = laion_clap.CLAP_Module(
                enable_fusion=False, device=device, tmodel="roberta"
            )
            _log(f"setup 2/3 complete ({time.monotonic() - stage_started:.1f}s)")
            stage_started = time.monotonic()
            _log("setup 3/3: loading checkpoint")
            self.model.load_ckpt(ckpt=checkpoint, verbose=False)
            self.model.eval()
            _log(f"setup 3/3 complete ({time.monotonic() - stage_started:.1f}s)")
        _log(f"setup complete ({time.monotonic() - setup_started:.1f}s)")

    def __call__(self, rows):
        from tqdm.auto import tqdm

        scores = []
        total_batches = (len(rows) + self.batch_size - 1) // self.batch_size
        starts = range(0, len(rows), self.batch_size)
        for start in tqdm(
            starts,
            total=total_batches,
            desc="CLAP scoring",
            unit="batch",
            dynamic_ncols=True,
            file=sys.stderr,
        ):
            batch = rows[start : start + self.batch_size]
            texts = [row[2] for row in batch]
            paths = [row[1] for row in batch]
            text = self.model.get_text_embedding(texts, use_tensor=False)
            audio = self.model.get_audio_embedding_from_filelist(x=paths, use_tensor=False)
            scores.append(
                np.sum(audio * text, axis=1)
                / (np.linalg.norm(audio, axis=1) * np.linalg.norm(text, axis=1))
            )
        return np.concatenate(scores)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)
    rows = []
    for line in Path(args.input_jsonl).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append((str(row["key"]), str(row["prediction_audio"]), str(row["caption"])))
    if args.json_output:
        with redirect_stdout(sys.stderr):
            trace = score_clap(rows, provider=CLAPProvider(args.device))
    else:
        trace = score_clap(rows, provider=CLAPProvider(args.device))
    print(
        json.dumps({"node_id": NODE_ID, "version": NODE_VERSION, "result": trace.details["result"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
