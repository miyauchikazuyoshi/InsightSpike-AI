"""Analyze loop (shortcut) events and multihop depth characteristics.

Runs the complex maze with dynamic thresholding and extracts:
  - Count of shortcut_candidate events
  - Distribution of multihop depth (max hop reached) per shortcut
  - Average score trajectory across hops
  - Example events (first 5)
"""
from __future__ import annotations
import os, sys
from typing import Dict, List
import numpy as np
import re

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'navigation'))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
for p in [PARENT_DIR, BASE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from complex_maze_branch_probe_v2 import build_maze  # type: ignore
from navigation.maze_navigator import MazeNavigator  # type: ignore

def main():
    maze, start, goal = build_maze()
    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy='simple',
        use_escalation=True,
        escalation_threshold='dynamic',
        dynamic_escalation=True,
        dynamic_offset=0.06,
        gedig_threshold=0.25,
        backtrack_threshold=-0.05,
    )
    nav.run(max_steps=3500)
    shortcut_events = [e for e in nav.event_log if e['type']=='shortcut_candidate']
    print(f"Total shortcut_candidate events: {len(shortcut_events)}")
    # Gather multihop per structural record where shortcut flag set
    depths: List[int] = []
    hop_vectors: List[List[float]] = []
    for r in nav.gedig_structural:
        if r.get('shortcut') and r.get('multihop'):
            mh: Dict[object, float] = r['multihop']  # type: ignore
            # keys may be 'hop0','hop1', or integers already
            parsed: List[tuple[int, float]] = []
            for k,v in mh.items():
                if isinstance(k,int):
                    idx = k
                else:
                    ks = str(k)
                    m = re.search(r'(\d+)', ks)
                    if m:
                        idx = int(m.group(1))
                    else:
                        # fallback: place unknown keys at end in insertion order
                        idx = 10_000 + len(parsed)
                parsed.append((idx, v))
            parsed.sort(key=lambda t: t[0])
            hop_scores = [val for _,val in parsed]
            if hop_scores:
                depths.append(len(hop_scores))
                hop_vectors.append(hop_scores)
    # raw profile (unclipped) も試行
    raw_vectors: List[List[float]] = []
    for r in nav.gedig_structural:
        if r.get('shortcut') and r.get('multihop_raw'):
            mh: Dict[object, float] = r['multihop_raw']  # type: ignore
            parsed: List[tuple[int, float]] = []
            for k,v in mh.items():
                if isinstance(k,int):
                    idx = k
                else:
                    ks = str(k); m = re.search(r'(\d+)', ks)
                    idx = int(m.group(1)) if m else 10_000 + len(parsed)
                parsed.append((idx, v))
            parsed.sort(key=lambda t: t[0])
            raw_scores = [val for _,val in parsed]
            if raw_scores:
                raw_vectors.append(raw_scores)
    if depths:
        print(f"Multihop depth stats: min={min(depths)} max={max(depths)} mean={sum(depths)/len(depths):.2f}")
        # Pad hop vectors for mean profile
        max_d = max(depths)
        arr = np.zeros((len(hop_vectors), max_d)) * np.nan
        for i, vec in enumerate(hop_vectors):
            arr[i,:len(vec)] = vec
        mean_profile = np.nanmean(arr, axis=0)
        print("Mean hop score profile:", [round(float(x),4) for x in mean_profile])
    if raw_vectors:
        max_r = max(len(v) for v in raw_vectors)
        arr_r = np.zeros((len(raw_vectors), max_r)) * np.nan
        for i, vec in enumerate(raw_vectors):
            arr_r[i,:len(vec)] = vec
        mean_raw = np.nanmean(arr_r, axis=0)
        print("Mean raw hop gain profile (unscaled, unclipped):", [round(float(x),6) for x in mean_raw])
    if shortcut_events:
        print("First 5 shortcut events:")
        for ev in shortcut_events[:5]:
            print(ev)

if __name__ == '__main__':
    main()
