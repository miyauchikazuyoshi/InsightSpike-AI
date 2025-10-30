#!/usr/bin/env python
"""Analyze rehydration dynamics from a persisted evicted catalog.

Usage:
  python analyze_rehydration.py --catalog /path/to/evicted_catalog.jsonl --sample 20

Outputs simple summary (counts, position distribution, top directions) to stdout.
"""
from __future__ import annotations
import argparse, json, collections, sys, os
from typing import Dict, Any

def load_catalog(path: str, limit: int | None = None):
    items = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                pos = rec.get('position')
                if isinstance(pos, list) and len(pos)==2:
                    rec['position'] = tuple(pos)
                items.append(rec)
            except Exception:
                continue
            if limit is not None and len(items) >= limit:
                break
    return items

def summarize(records: list[Dict[str, Any]]) -> Dict[str, Any]:
    pos_counter = collections.Counter(rec['position'] for rec in records if 'position' in rec)
    dir_counter = collections.Counter(rec.get('direction') for rec in records if rec.get('direction') is not None)
    visit_stats = [rec.get('visit_count',0) for rec in records]
    out: Dict[str, Any] = {
        'records': len(records),
        'unique_positions': len(pos_counter),
        'top_positions': pos_counter.most_common(5),
        'top_directions': dir_counter.most_common(5),
        'avg_visit_count': (sum(visit_stats)/len(visit_stats)) if visit_stats else 0.0,
    }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--sample', type=int, default=0, help='Random sample print size')
    args = ap.parse_args()
    if not os.path.isfile(args.catalog):
        print(f"Catalog not found: {args.catalog}", file=sys.stderr)
        sys.exit(2)
    records = load_catalog(args.catalog, args.limit)
    summ = summarize(records)
    print(json.dumps(summ, ensure_ascii=False, indent=2))
    if args.sample > 0:
        import random
        sample = random.sample(records, k=min(args.sample, len(records)))
        print("\n# Sample Records")
        for rec in sample:
            print(json.dumps(rec, ensure_ascii=False))

if __name__ == '__main__':
    main()
