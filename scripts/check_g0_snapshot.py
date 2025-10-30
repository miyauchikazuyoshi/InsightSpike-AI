#!/usr/bin/env python3
"""
Quick sanity checker for maze run outputs.

Loads a run JSON (either raw output from run_experiment.py or the derived
visualization run_data.json) and verifies that:

* g0_history aligns with g0_components[*]['g0']
* gmin_history aligns with g0_components[*]['gmin'] when present
* IG-related fields are populated when ig_value != 0

Intended for lightweight regression checks after navigator/geDIG changes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

FLOAT_TOL = 1e-6


def _load_detail(payload: Dict[str, Any], index: int) -> Dict[str, Any]:
    if "detail" in payload:
        return payload["detail"]
    details = payload.get("details")
    if not isinstance(details, list) or not details:
        raise ValueError("Input JSON does not contain 'detail' or non-empty 'details'.")
    if index < 0 or index >= len(details):
        raise IndexError(f"detail index {index} out of range (size={len(details)}).")
    return details[index]


def _iter_pairs(detail: Dict[str, Any], limit: int | None = None) -> Iterable[Tuple[int, Dict[str, Any]]]:
    g0_hist = detail.get("g0_history") or []
    g0_components = detail.get("g0_components") or []
    total = min(len(g0_hist), len(g0_components))
    if limit is not None:
        total = min(total, limit)
    for idx in range(total):
        yield idx, {
            "g0": g0_hist[idx],
            "gmin": (detail.get("gmin_history") or [None])[idx] if detail.get("gmin_history") else None,
            "components": g0_components[idx],
        }


def _check(detail: Dict[str, Any], limit: int | None = None) -> Tuple[int, List[str]]:
    mismatches = 0
    messages: List[str] = []
    for idx, bundle in _iter_pairs(detail, limit):
        comp = bundle["components"]
        g0_cmp = comp.get("g0")
        g0_hist = bundle["g0"]
        if g0_cmp is None or math.isnan(g0_cmp):
            mismatches += 1
            messages.append(f"[step {idx}] missing component g0 value")
        elif abs(float(g0_hist) - float(g0_cmp)) > FLOAT_TOL:
            mismatches += 1
            messages.append(f"[step {idx}] g0 mismatch (history={g0_hist:.6f}, component={g0_cmp:.6f})")

        gmin_cmp = comp.get("gmin")
        gmin_hist = bundle["gmin"]
        if gmin_hist is not None and gmin_cmp is not None:
            if abs(float(gmin_hist) - float(gmin_cmp)) > FLOAT_TOL:
                mismatches += 1
                messages.append(
                    f"[step {idx}] gmin mismatch (history={gmin_hist:.6f}, component={gmin_cmp:.6f})"
                )

        ig_value = comp.get("ig_value")
        if ig_value not in (None, 0, 0.0):
            ent_before = comp.get("entropy_before")
            ent_after = comp.get("entropy_after")
            if ent_before is None or ent_after is None:
                mismatches += 1
                messages.append(f"[step {idx}] ig_value={ig_value} but entropy fields missing")
            elif abs(float(ent_before) - float(ent_after)) < FLOAT_TOL:
                messages.append(f"[step {idx}] warning: entropy_before ~= entropy_after despite ig_value={ig_value}")

    return mismatches, messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Check g0/gmin/IG fields for maze run outputs.")
    parser.add_argument("input", type=Path, help="Path to run JSON (raw output or run_data.json).")
    parser.add_argument("--detail-index", type=int, default=0, help="Index for multi-seed 'details' array.")
    parser.add_argument("--limit", type=int, default=None, help="Only inspect the first N steps.")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    detail = _load_detail(payload, args.detail_index)

    mismatches, messages = _check(detail, args.limit)
    for msg in messages:
        print(msg)

    if mismatches > 0:
        print(f"FAIL: detected {mismatches} mismatches.")
        raise SystemExit(1)

    total_steps = len(detail.get("g0_history") or [])
    inspected = min(args.limit or total_steps, total_steps)
    print(f"OK: inspected {inspected} steps (total available: {total_steps}).")


if __name__ == "__main__":
    main()
