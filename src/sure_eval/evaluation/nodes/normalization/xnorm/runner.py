"""Node-local process entrypoint for XNorm file normalization."""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import types

from sure_eval.evaluation.core.types import KeyTextFiles

if "cdifflib" not in sys.modules:
    _cdifflib = types.ModuleType("cdifflib")
    _cdifflib.CSequenceMatcher = difflib.SequenceMatcher
    sys.modules["cdifflib"] = _cdifflib

from .node import _normalize_key_text_files_in_process


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-file", required=True)
    parser.add_argument("--hyp-file", required=True)
    parser.add_argument("--language", required=True)
    args = parser.parse_args()
    files, trace = _normalize_key_text_files_in_process(
        KeyTextFiles(ref_file=args.ref_file, hyp_file=args.hyp_file), args.language
    )
    print(json.dumps({
        "normalized_files": {"ref_file": files.ref_file, "hyp_file": files.hyp_file},
        "trace": {
            "details": trace.details,
            "internal_stages": list(trace.internal_stages),
        },
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
