#!/usr/bin/env python3
"""
Legacy documentation scheduler stub.

The original project generated a decision schedule from docs/development/*.md.
That directory is not part of this repository, but the docs_lint workflow still
invokes this script. We emit a tiny placeholder JSONL entry so downstream steps
have predictable output.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone


def main() -> None:
    schedule_entry = {
        "status": "noop",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": "docs/development/ absent; schedule generation skipped",
    }
    print(json.dumps(schedule_entry))


if __name__ == "__main__":
    main()

