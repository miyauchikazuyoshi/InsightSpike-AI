#!/usr/bin/env python3
"""
List large tracked files to help decide Releases/LFS moves.

Usage:
  python scripts/audit_large_files.py --min-mb 10 --top 200 > docs/asset_audit/LARGE_FILES.md
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import List, Tuple


def git_ls_files() -> List[Path]:
    out = subprocess.check_output(["git", "ls-files"], text=True)
    return [Path(line) for line in out.splitlines() if line.strip()]


def human_mb(nbytes: int) -> float:
    return nbytes / (1024 * 1024)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mb", type=float, default=10.0)
    ap.add_argument("--top", type=int, default=200)
    args = ap.parse_args()

    files = git_ls_files()
    sized: List[Tuple[float, Path]] = []
    for p in files:
        if not p.exists() or p.is_dir():
            continue
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        sized.append((human_mb(sz), p))

    sized.sort(reverse=True)
    sized = [t for t in sized if t[0] >= args.min_mb][: args.top]

    print("# Large tracked files (>= {:.1f} MB)".format(args.min_mb))
    print()
    print("| Size (MB) | Path |")
    print("|---:|:---|")
    for mb, p in sized:
        print(f"| {mb:.2f} | {p} |")

    print()
    print("## Next steps")
    print("- Consider moving heavy generated artifacts to GitHub Releases or Git LFS.")
    print("- For history cleanup, evaluate BFG Repo-Cleaner or git filter-repo (with caution).")


if __name__ == "__main__":
    main()

