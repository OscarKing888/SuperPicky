#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incremental patch tool for SuperPicky release folders.

Features:
- Compare two directories and extract added/modified files into a patch folder
- Generate manifest (added / modified / deleted / unchanged counts)
- Optionally create a zip archive
- Apply a generated patch to a target directory (including deletions)

This is designed to be cross-platform (Windows / macOS / Linux).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from zipfile import ZIP_DEFLATED, ZipFile


MANIFEST_NAME = "patch_manifest.json"
PAYLOAD_DIRNAME = "payload"


@dataclass(frozen=True)
class FileEntry:
    rel_path: str
    abs_path: Path
    size: int
    mtime_ns: int


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel_str(path: Path, root: Path) -> str:
    # Use POSIX separators in manifests for cross-platform stability.
    return path.relative_to(root).as_posix()


def _iter_files(root: Path) -> Dict[str, FileEntry]:
    files: Dict[str, FileEntry] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        rel = _rel_str(path, root)
        files[rel] = FileEntry(
            rel_path=rel,
            abs_path=path,
            size=stat.st_size,
            mtime_ns=getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)),
        )
    return files


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _files_equal(a: FileEntry, b: FileEntry) -> bool:
    if a.size != b.size:
        return False
    # Use hash compare for correctness. mtime can collide across rapid writes.
    return _sha256_file(a.abs_path) == _sha256_file(b.abs_path)


def _copy_rel_files(src_root: Path, rel_paths: Iterable[str], dst_root: Path) -> None:
    for rel in rel_paths:
        src = src_root / Path(rel)
        dst = dst_root / Path(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _zip_dir(src_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for path in src_dir.rglob("*"):
            if path.is_dir():
                continue
            arcname = path.relative_to(src_dir).as_posix()
            zf.write(path, arcname)


def _print_summary(prefix: str, manifest: dict) -> None:
    summary = manifest.get("summary", {})
    print(
        f"{prefix}: added={summary.get('added', 0)}, "
        f"modified={summary.get('modified', 0)}, "
        f"deleted={summary.get('deleted', 0)}, "
        f"unchanged={summary.get('unchanged', 0)}"
    )


def make_patch(
    old_dir: Path,
    new_dir: Path,
    out_dir: Path,
    *,
    zip_output: bool = False,
    zip_path: Path | None = None,
) -> Tuple[Path, dict]:
    if not old_dir.is_dir():
        raise FileNotFoundError(f"旧目录不存在: {old_dir}")
    if not new_dir.is_dir():
        raise FileNotFoundError(f"新目录不存在: {new_dir}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload_dir = out_dir / PAYLOAD_DIRNAME
    payload_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Patch] Scanning old: {old_dir}")
    old_files = _iter_files(old_dir)
    print(f"[Patch] Scanning new: {new_dir}")
    new_files = _iter_files(new_dir)

    old_keys = set(old_files)
    new_keys = set(new_files)

    added = sorted(new_keys - old_keys)
    deleted = sorted(old_keys - new_keys)
    maybe_common = sorted(old_keys & new_keys)

    modified: List[str] = []
    unchanged: List[str] = []

    for rel in maybe_common:
        if _files_equal(old_files[rel], new_files[rel]):
            unchanged.append(rel)
        else:
            modified.append(rel)

    changed = added + modified
    _copy_rel_files(new_dir, changed, payload_dir)

    manifest = {
        "format_version": 1,
        "generated_at_utc": _utc_now_iso(),
        "tool": "scripts_dev/incremental_patch_tool.py",
        "python_version": sys.version.split()[0],
        "source": {
            "old_dir": str(old_dir.resolve()),
            "new_dir": str(new_dir.resolve()),
        },
        "summary": {
            "old_files": len(old_files),
            "new_files": len(new_files),
            "added": len(added),
            "modified": len(modified),
            "deleted": len(deleted),
            "unchanged": len(unchanged),
            "payload_files": len(changed),
        },
        "files": {
            "added": added,
            "modified": modified,
            "deleted": deleted,
        },
    }

    _write_json(out_dir / MANIFEST_NAME, manifest)

    if zip_output:
        if zip_path is None:
            zip_path = out_dir.with_suffix(".zip")
        print(f"[Patch] Creating zip: {zip_path}")
        _zip_dir(out_dir, zip_path)
        manifest["zip_path"] = str(zip_path.resolve())

    _print_summary("[Patch] Done", manifest)
    return out_dir, manifest


def apply_patch(
    patch_dir: Path,
    target_dir: Path,
    *,
    dry_run: bool = False,
    skip_delete: bool = False,
) -> dict:
    manifest_path = patch_dir / MANIFEST_NAME
    payload_dir = patch_dir / PAYLOAD_DIRNAME

    if not manifest_path.is_file():
        raise FileNotFoundError(f"补丁清单不存在: {manifest_path}")
    if not payload_dir.is_dir():
        raise FileNotFoundError(f"补丁 payload 目录不存在: {payload_dir}")
    if not target_dir.is_dir():
        raise FileNotFoundError(f"目标目录不存在: {target_dir}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    added = manifest.get("files", {}).get("added", [])
    modified = manifest.get("files", {}).get("modified", [])
    deleted = manifest.get("files", {}).get("deleted", [])
    changed = list(added) + list(modified)

    print(f"[Patch] Applying to: {target_dir}")
    print(f"[Patch] Files to copy: {len(changed)}")
    print(f"[Patch] Files to delete: {0 if skip_delete else len(deleted)}")

    copied = 0
    removed = 0
    missing_payload = []

    for rel in changed:
        src = payload_dir / Path(rel)
        dst = target_dir / Path(rel)
        if not src.is_file():
            missing_payload.append(rel)
            continue
        if dry_run:
            copied += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    if not skip_delete:
        for rel in deleted:
            dst = target_dir / Path(rel)
            if dry_run:
                if dst.exists():
                    removed += 1
                continue
            try:
                if dst.is_file():
                    dst.unlink()
                    removed += 1
            except FileNotFoundError:
                continue

    result = {
        "copied": copied,
        "deleted": removed,
        "missing_payload": missing_payload,
        "dry_run": dry_run,
        "skip_delete": skip_delete,
    }
    print(
        f"[Patch] Apply done: copied={copied}, deleted={removed}, "
        f"missing_payload={len(missing_payload)}"
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or apply incremental patch between release directories."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_make = sub.add_parser("make", help="Compare two directories and create patch output")
    p_make.add_argument("--old", dest="old_dir", required=True, help="Previous version directory")
    p_make.add_argument("--new", dest="new_dir", required=True, help="Current version directory")
    p_make.add_argument("--out", dest="out_dir", required=True, help="Patch output directory")
    p_make.add_argument(
        "--zip",
        action="store_true",
        help="Create zip archive from patch output directory",
    )
    p_make.add_argument(
        "--zip-path",
        dest="zip_path",
        default="",
        help="Optional zip output path (default: <out>.zip)",
    )

    p_apply = sub.add_parser("apply", help="Apply a generated patch to a target directory")
    p_apply.add_argument("--patch", dest="patch_dir", required=True, help="Patch directory")
    p_apply.add_argument("--target", dest="target_dir", required=True, help="Target install directory")
    p_apply.add_argument("--dry-run", action="store_true", help="Preview only")
    p_apply.add_argument("--skip-delete", action="store_true", help="Do not delete files listed in manifest")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "make":
            zip_path = Path(args.zip_path) if args.zip_path else None
            make_patch(
                Path(args.old_dir),
                Path(args.new_dir),
                Path(args.out_dir),
                zip_output=args.zip,
                zip_path=zip_path,
            )
            return 0

        if args.command == "apply":
            result = apply_patch(
                Path(args.patch_dir),
                Path(args.target_dir),
                dry_run=args.dry_run,
                skip_delete=args.skip_delete,
            )
            if result["missing_payload"]:
                print("[Patch] Error: patch payload is incomplete")
                return 2
            return 0

        parser.print_help()
        return 1
    except Exception as e:
        print(f"[Patch] Failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
