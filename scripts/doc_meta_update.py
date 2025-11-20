#!/usr/bin/env python3
"""
Legacy doc meta update stub.

The original project used this script to update front matter and indices for
docs/development/*.md. The current repository does not ship those docs, but
GitHub Actions still invokes this script.

To avoid CI failures while keeping behaviour benign, this stub simply prints a
short message and exits with status 0.
"""
from __future__ import annotations


def main() -> None:
    print("[doc_meta_update] No docs/development/ tree; stub doing nothing.")


if __name__ == "__main__":
    main()

