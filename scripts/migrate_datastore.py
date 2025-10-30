#!/usr/bin/env python3
"""
Datastore Path Migration Utility
================================

Safely copy or move data from the legacy base path (typically "data/") to the
new datastore root (typically "./data/insight_store").

Usage examples:

  # Auto-detect from current config (source=paths.data_dir, dest=datastore.root_path)
  PYTHONPATH=src python scripts/migrate_datastore.py --mode copy --dry-run

  # Execute copy
  PYTHONPATH=src python scripts/migrate_datastore.py --mode copy

  # Move instead of copy
  PYTHONPATH=src python scripts/migrate_datastore.py --mode move

  # Override paths explicitly
  PYTHONPATH=src python scripts/migrate_datastore.py \
      --source ./data \
      --dest ./data/insight_store \
      --mode copy
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Tuple, Optional


def _detect_paths() -> Tuple[Path, Path]:
    """Detect legacy source and new destination paths using current config.

    - source: config.paths.data_dir (default: ./data)
    - dest:   config.datastore.root_path (default: ./data/insight_store)
    """
    try:
        # Lazy import via PYTHONPATH=src
        from insightspike.config.loader import load_config  # type: ignore
        from insightspike.config.models import DataStoreConfig  # type: ignore

        cfg = load_config()
        source = Path(str(getattr(getattr(cfg, "paths", None), "data_dir", "./data")))
        dest = Path(str(getattr(getattr(cfg, "datastore", None), "root_path", DataStoreConfig().root_path)))
        return source, dest
    except Exception:
        # Fallbacks if config load fails
        return Path("./data"), Path("./data/insight_store")


def _is_inside(path: Path, directory: Path) -> bool:
    """Return True if 'path' is inside 'directory' (or equals it)."""
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except Exception:
        return False


def _copy_tree(src: Path, dst: Path, *, exclude: Optional[Path] = None) -> None:
    """Copy directory tree from src to dst (merging if dst exists).

    If 'exclude' is provided and lies within 'src', that subtree is skipped.
    This prevents recursive copy when dst is a subdirectory of src.
    """
    dst.mkdir(parents=True, exist_ok=True)
    exclude_resolved = exclude.resolve() if exclude else None
    src_resolved = src.resolve()

    for root, dirs, files in os.walk(src):
        root_path = Path(root).resolve()
        # Skip excluded subtree
        if exclude_resolved and _is_inside(root_path, exclude_resolved):
            continue

        # Prune traversal into excluded subtree
        if exclude_resolved:
            dirs[:] = [
                d for d in dirs
                if not _is_inside((root_path / d), exclude_resolved)
            ]

        rel = root_path.relative_to(src_resolved)
        target_dir = dst / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for d in dirs:
            (target_dir / d).mkdir(parents=True, exist_ok=True)
        for f in files:
            src_file = root_path / f
            # Skip files inside excluded subtree (safety)
            if exclude_resolved and _is_inside(src_file, exclude_resolved):
                continue
            dst_file = target_dir / f
            shutil.copy2(src_file, dst_file)


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate datastore files between base paths")
    parser.add_argument("--source", type=str, default=None, help="Source path (legacy base), defaults to config.paths.data_dir")
    parser.add_argument("--dest", type=str, default=None, help="Destination path (new root), defaults to config.datastore.root_path")
    parser.add_argument("--mode", choices=["copy", "move"], default="copy", help="Copy or move files")
    parser.add_argument("--dry-run", action="store_true", help="Print intended operations without changing files")
    parser.add_argument("--force", action="store_true", help="Proceed even if destination is non-empty (merge)")

    args = parser.parse_args()

    src_default, dst_default = _detect_paths()
    src = Path(args.source) if args.source else src_default
    dst = Path(args.dest) if args.dest else dst_default

    print(f"Detected source: {src}")
    print(f"Detected destination: {dst}")
    print(f"Mode: {args.mode}; Dry-run: {args.dry_run}; Force: {args.force}")

    if not src.exists():
        print(f"[WARN] Source path does not exist: {src}")
        return 0

    if src.resolve() == dst.resolve():
        print("[INFO] Source and destination are the same. Nothing to do.")
        return 0

    # Safety: if destination exists and not empty, require --force to merge
    if dst.exists():
        # Check non-empty
        try:
            non_empty = any(dst.iterdir())
        except Exception:
            non_empty = True
        if non_empty and not args.force and args.mode == "copy":
            print(f"[ERROR] Destination exists and is non-empty: {dst}. Use --force to merge or choose an empty directory.")
            return 2

    if args.dry_run:
        print("[DRY-RUN] Would perform the following operations:")
        for root, dirs, files in os.walk(src):
            rel = Path(root).relative_to(src)
            target_dir = dst / rel
            print(f"  mkdir -p {target_dir}")
            for d in dirs:
                print(f"  mkdir -p {target_dir / d}")
            for f in files:
                print(f"  {args.mode} {Path(root)/f} -> {target_dir/f}")
        return 0

    # Execute
    dst.mkdir(parents=True, exist_ok=True)
    if args.mode == "copy":
        # Exclude destination when it is a subdirectory of source
        exclude = dst if _is_inside(dst, src) else None
        _copy_tree(src, dst, exclude=exclude)
        print(f"[OK] Copied contents from '{src}' to '{dst}'")
    else:
        # move: best-effort dir move (merging if dst exists)
        # shutil.move won't merge deep trees safely; do manual move per entry
        for entry in src.iterdir():
            # Skip destination dir if it's directly under source
            if entry.resolve() == dst.resolve():
                continue
            target = dst / entry.name
            if entry.is_dir():
                # Ensure we don't recurse into destination
                exclude = dst if _is_inside(dst, entry) else None
                _copy_tree(entry, target, exclude=exclude)
                shutil.rmtree(entry)
            else:
                shutil.move(str(entry), str(target))
        # Optionally remove src if empty
        try:
            src.rmdir()
        except OSError:
            pass
        print(f"[OK] Moved contents from '{src}' to '{dst}'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
