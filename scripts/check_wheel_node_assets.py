#!/usr/bin/env python3
"""Verify that node-local runtime assets are included in a built wheel."""

from __future__ import annotations

import argparse
import posixpath
import zipfile
from pathlib import Path

import yaml

REQUIRED_NODE_STATIC_ASSETS = {
    "sure_eval/evaluation/nodes/normalization/whisper_norm/manifest.yaml": (
        "sure_eval/evaluation/nodes/normalization/whisper_norm/normalization_impl/english.json",
        "sure_eval/evaluation/nodes/normalization/whisper_norm/normalization_impl/LICENSE.openai-whisper",
    ),
}


def missing_node_assets(wheel: Path) -> list[str]:
    missing: list[str] = []
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        node_env_files = sorted(name for name in names if name.endswith("/node_env.yaml"))
        if not node_env_files:
            return ["wheel contains no node_env.yaml files"]
        packaged_runtime = sorted(
            name for name in names if "/nodes/" in name and "/runtime/" in name
        )
        if packaged_runtime:
            missing.append(f"wheel contains node runtime asset: {packaged_runtime[0]}")

        for trigger, required_assets in REQUIRED_NODE_STATIC_ASSETS.items():
            if trigger not in names:
                continue
            for asset in required_assets:
                if asset not in names:
                    missing.append(f"{trigger}: missing {posixpath.basename(asset)}")

        for node_env_file in node_env_files:
            payload = yaml.safe_load(archive.read(node_env_file)) or {}
            runtime = payload.get("runtime") or {}
            node_dir = posixpath.dirname(node_env_file)
            referenced_assets = [
                runtime.get("project"),
                runtime.get("build_script"),
                runtime.get("post_setup_script"),
            ]
            if runtime.get("frozen"):
                referenced_assets.append("uv.lock")

            for asset in referenced_assets:
                if not asset:
                    continue
                member = posixpath.normpath(posixpath.join(node_dir, str(asset)))
                if member not in names:
                    missing.append(f"{node_env_file}: missing {asset}")

    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()

    missing = missing_node_assets(args.wheel)
    if missing:
        for item in missing:
            print(item)
        return 1
    print(f"Node runtime assets verified: {args.wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
