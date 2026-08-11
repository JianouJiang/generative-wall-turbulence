#!/usr/bin/env python3
"""Verify that the exact manifest is dependency-closed in a clean tree."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="development/nodes/node_010/clean_release_verification.json",
    )
    parser.add_argument("--keep-tree", action="store_true")
    args = parser.parse_args()

    manifest_path = ROOT / "SUBMISSION_RELEASE_MANIFEST.json"
    checksum_path = ROOT / "FIGURE_DATA_SHA256SUMS"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    temporary = Path(tempfile.mkdtemp(prefix="gwt-clean-release-"))
    clean_root = temporary / "release"
    clean_root.mkdir()
    try:
        copied = []
        for entry in manifest["entries"]:
            source = ROOT / entry["path"]
            destination = clean_root / entry["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(entry["path"])
        shutil.copy2(manifest_path, clean_root / manifest_path.name)
        shutil.copy2(checksum_path, clean_root / checksum_path.name)

        environment = os.environ.copy()
        environment["GWT_CLEAN_RELEASE"] = "1"
        completed = subprocess.run(
            ["bash", "codes/reproduce_all.sh"],
            cwd=clean_root,
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        pass_marker = "REPRODUCE_ALL: ALL_PASS" in completed.stdout
        payload = {
            "schema": "gwt-clean-release-verification-v1",
            "source_manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_entry_count": len(manifest["entries"]),
            "copied_entry_count": len(copied),
            "temporary_tree": str(clean_root) if args.keep_tree else "deleted after verification",
            "command": "bash codes/reproduce_all.sh",
            "returncode": completed.returncode,
            "pass_marker": pass_marker,
            "all_pass": completed.returncode == 0 and pass_marker,
            "stdout_tail": completed.stdout.splitlines()[-30:],
            "stderr_tail": completed.stderr.splitlines()[-30:],
        }
        output = ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not payload["all_pass"]:
            raise SystemExit(
                "CLEAN_RELEASE_FAIL\n"
                + "\n".join(payload["stdout_tail"] + payload["stderr_tail"])
            )
        try:
            output_label = output.relative_to(ROOT)
        except ValueError:
            output_label = output
        print(f"CLEAN_RELEASE_PASS entries={len(copied)} output={output_label}")
    finally:
        if not args.keep_tree:
            shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    main()
