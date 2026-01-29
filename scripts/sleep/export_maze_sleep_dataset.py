#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _infer_run_id(step_log_path: Path) -> str:
    # Common naming: *_steps.json
    stem = step_log_path.stem
    if stem.endswith("_steps"):
        return stem[: -len("_steps")]
    return stem


def _episode_id(run_id: str, seed: int, step: int) -> str:
    return f"maze:{run_id}:{int(seed)}:{int(step)}"


def _euclid(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _standardize(vectors: List[List[float]]) -> Tuple[List[float], List[float]]:
    if not vectors:
        return [], []
    dim = len(vectors[0])
    mean = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            mean[i] += float(v[i])
    mean = [m / max(len(vectors), 1) for m in mean]
    var = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            d = float(v[i]) - mean[i]
            var[i] += d * d
    std = [math.sqrt(v / max(len(vectors), 1)) for v in var]
    std = [s if s > 1e-9 else 1.0 for s in std]
    return mean, std


def _zscore(v: List[float], mean: List[float], std: List[float]) -> List[float]:
    return [(float(x) - mean[i]) / std[i] for i, x in enumerate(v)]


@dataclass
class Proposal:
    episode_id: str
    run_id: str
    seed: int
    step: int
    vector: List[float]
    label: int
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Sleep datasets from maze step logs (+ optional DG ledger).")
    parser.add_argument("--step-log", type=Path, nargs="+", required=True, help="Maze steps JSON file(s) (list of dicts).")
    parser.add_argument("--dg-ledger", type=Path, default=None, help="Optional DG ledger JSONL emitted by --dg-ledger-log.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/sleep/maze"), help="Output directory.")
    parser.add_argument("--hard-negatives", type=int, default=3, help="Hard negatives per anchor (DG triplets).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- episodes / affordance ----
    episodes_out: List[Dict[str, Any]] = []
    affordance_out: List[Dict[str, Any]] = []
    missing_vectors = 0
    total_steps = 0

    for step_path in args.step_log:
        payload = _read_json(step_path)
        if isinstance(payload, dict) and "steps" in payload and isinstance(payload["steps"], list):
            steps = payload["steps"]
        elif isinstance(payload, list):
            steps = payload
        else:
            raise SystemExit(f"Unsupported step-log format: {step_path}")

        default_run_id = _infer_run_id(step_path)
        for row in steps:
            if not isinstance(row, dict):
                continue
            total_steps += 1

            run_id = str(row.get("run_id") or default_run_id)
            seed = int(row.get("seed", 0))
            step = int(row.get("step", 0))
            episode_id = str(row.get("episode_id") or _episode_id(run_id, seed, step))
            vector = row.get("episode_vector") or []

            if not isinstance(vector, list) or not vector:
                missing_vectors += 1
                continue

            moved = bool(row.get("moved", False))
            episodes_out.append(
                {
                    "domain": "maze",
                    "type": "episode",
                    "episode_id": episode_id,
                    "run_id": run_id,
                    "seed": seed,
                    "step": step,
                    "vector": vector,
                    "position_pre": row.get("position_pre"),
                    "position_post": row.get("position_post"),
                    "moved": moved,
                    "action": row.get("action"),
                    "gate": {"g0": row.get("g0"), "gmin": row.get("gmin"), "best_hop": row.get("best_hop")},
                }
            )

            affordance_out.append(
                {
                    "domain": "maze",
                    "type": "affordance",
                    "episode_id": episode_id,
                    "run_id": run_id,
                    "seed": seed,
                    "step": step,
                    "action": row.get("action"),
                    "label": "passable" if moved else "blocked",
                    "moved": moved,
                    "vector": vector,
                }
            )

    episodes_path = args.out_dir / "episodes.jsonl"
    affordance_path = args.out_dir / "affordance.jsonl"
    n_episodes = _write_jsonl(episodes_path, episodes_out)
    n_aff = _write_jsonl(affordance_path, affordance_out)

    # ---- DG proposals + triplets ----
    proposals_out: List[Dict[str, Any]] = []
    triplets_out: List[Dict[str, Any]] = []
    proposals: List[Proposal] = []

    if args.dg_ledger:
        for ev in _iter_jsonl(args.dg_ledger):
            if ev.get("event") != "dg_decision":
                continue
            gate = ev.get("gate") or {}
            metrics = ev.get("metrics") or {}
            staged = ev.get("staged_edges") or []
            committed = ev.get("committed_edges") or []

            run_id = str(ev.get("run_id") or "unknown")
            seed = int(ev.get("seed", 0))
            step = int(ev.get("step", 0))
            episode_id = str(ev.get("episode_id") or _episode_id(run_id, seed, step))

            # Proposal feature vector (small + numeric): suitable for contrastive baselines.
            vec = [
                float(gate.get("g0", 0.0)),
                float(gate.get("gmin", 0.0)),
                float(metrics.get("delta_ged", 0.0)),
                float(metrics.get("delta_ig", 0.0)),
                float(gate.get("best_hop", 0.0)),
                float(len(staged)),
                float(len(committed)),
            ]
            label = 1 if str(ev.get("decision", "")).lower() == "commit" else 0
            reason = str(ev.get("reason") or "")
            proposals.append(Proposal(episode_id=episode_id, run_id=run_id, seed=seed, step=step, vector=vec, label=label, reason=reason))

            proposals_out.append(
                {
                    "domain": "maze",
                    "type": "dg_proposal",
                    "episode_id": episode_id,
                    "run_id": run_id,
                    "seed": seed,
                    "step": step,
                    "label": label,
                    "reason": reason,
                    "vector": vec,
                    "gate": gate,
                    "metrics": metrics,
                    "staged_edges_count": len(staged),
                    "committed_edges_count": len(committed),
                }
            )

        if proposals:
            vectors = [p.vector for p in proposals]
            mean, std = _standardize(vectors)
            zvecs = [_zscore(p.vector, mean, std) for p in proposals]

            commit_idx = [i for i, p in enumerate(proposals) if p.label == 1]
            reject_idx = [i for i, p in enumerate(proposals) if p.label == 0]

            for i in commit_idx:
                # positive: nearest other commit
                pos = None
                pos_dist = float("inf")
                for j in commit_idx:
                    if i == j:
                        continue
                    d = _euclid(zvecs[i], zvecs[j])
                    if d < pos_dist:
                        pos_dist = d
                        pos = j
                if pos is None:
                    continue

                # hard negatives: nearest rejects
                negs: List[Tuple[float, int]] = []
                for j in reject_idx:
                    d = _euclid(zvecs[i], zvecs[j])
                    negs.append((d, j))
                negs.sort(key=lambda x: x[0])
                chosen_negs = [j for _, j in negs[: max(0, int(args.hard_negatives))]]

                triplets_out.append(
                    {
                        "domain": "maze",
                        "type": "dg_triplet",
                        "anchor_episode_id": proposals[i].episode_id,
                        "positive_episode_id": proposals[pos].episode_id,
                        "negative_episode_ids": [proposals[j].episode_id for j in chosen_negs],
                        "anchor_vector": proposals[i].vector,
                        "positive_vector": proposals[pos].vector,
                        "negative_vectors": [proposals[j].vector for j in chosen_negs],
                        "standardize": {"mean": mean, "std": std},
                    }
                )

    proposals_path = args.out_dir / "dg_proposals.jsonl"
    triplets_path = args.out_dir / "dg_triplets.jsonl"
    n_props = _write_jsonl(proposals_path, proposals_out) if proposals_out else 0
    n_trip = _write_jsonl(triplets_path, triplets_out) if triplets_out else 0

    stats = {
        "inputs": {"step_logs": [str(p) for p in args.step_log], "dg_ledger": str(args.dg_ledger) if args.dg_ledger else None},
        "counts": {
            "steps_seen": total_steps,
            "episodes_written": n_episodes,
            "affordance_written": n_aff,
            "dg_proposals_written": n_props,
            "dg_triplets_written": n_trip,
            "missing_episode_vectors": missing_vectors,
        },
    }
    stats_path = args.out_dir / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

