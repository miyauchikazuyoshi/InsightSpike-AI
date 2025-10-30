#!/usr/bin/env python3
"""Convert QA JSONL into alignment input JSON for run_insight_vector_alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict


def load_qa_pairs(path: Path) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            q = rec.get("question") or rec.get("query")
            a = rec.get("answer") or rec.get("response") or rec.get("ground_truth")
            if not q or not a:
                continue
            pairs.append({"question": q, "response": a})
    return pairs


def save_alignment_json(pairs: List[Dict[str, str]], output_path: Path) -> None:
    payload = {"results": pairs}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare alignment input JSON from QA JSONL.")
    parser.add_argument("--qa-jsonl", type=Path, required=True, help="QA log in JSONL (question/answer).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "inputs" / "alignment_inputs.json",
        help="Output JSON path for alignment script.",
    )
    args = parser.parse_args()

    pairs = load_qa_pairs(args.qa_jsonl)
    if not pairs:
        raise SystemExit(f"No valid QA pairs found in {args.qa_jsonl}")
    save_alignment_json(pairs, args.output)
    print(f"Wrote alignment inputs for {len(pairs)} QA pairs → {args.output}")


if __name__ == "__main__":
    main()
