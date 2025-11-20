#!/usr/bin/env python3
"""
Legacy docs index builder stub.

The original project used this script to maintain an index under docs/development/.
That directory no longer exists in this fork, but GitHub Actions still calls this
script via docs_lint.yml. To keep CI green without reintroducing the old system,
we provide a no-op stub that exits successfully.
"""
from __future__ import annotations

import pathlib


def main() -> None:
    root = pathlib.Path("docs/development")
    if root.is_dir():
        # In case a user reintroduces docs/development, list files as a tiny sanity check.
        files = sorted(p.name for p in root.glob("*.md"))
        print(f"[build_plan_index] docs/development/ present, {len(files)} md files")
    else:
        print("[build_plan_index] docs/development/ not present; nothing to do")


if __name__ == "__main__":
    main()

