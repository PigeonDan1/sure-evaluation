"""Node-local inference entrypoint for the released ETC canonicalizer."""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

from .model_runtime import canonicalize_batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    try:
        texts = json.load(sys.stdin)["texts"]
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise ValueError("texts must be a list of strings")
        payload = {"predictions": canonicalize_batch(texts, model=args.model)}
    except Exception as exc:
        print(f"ETC canonicalizer failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
