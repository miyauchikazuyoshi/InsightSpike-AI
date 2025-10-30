#!/usr/bin/env python3
"""Multi-seed variability of dead-end phase quantiles.
Runs deadend_heuristic_probe logic across N seeds by adding noise to weights.
Outputs distribution of terminal p25 / p10 per mini maze.
"""
from __future__ import annotations
import os, sys, json, argparse, datetime, random
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from deadend_mini_mazes import MAZE_BUILDERS
from navigation.maze_navigator import MazeNavigator  # type: ignore

from deadend_heuristic_probe import PhaseAnnotator, compute_phase_stats, _roc_auc  # reuse


def run_once(builder, seed:int, temperature:float, gedig_threshold:float, backtrack_threshold:float, max_steps:int, noise:float):
    maze,start,goal,name,desc=builder()
    rng=np.random.default_rng(seed)
    base_weights=np.array([1.0,1.0,0.0,0.0,3.0,2.0,0.1,0.0])
    if noise>0:
        perturb=rng.normal(0, noise, size=base_weights.shape)
        weights=base_weights + perturb
    else:
        weights=base_weights
    nav=MazeNavigator(maze,start,goal,weights=weights,temperature=temperature,gedig_threshold=gedig_threshold,backtrack_threshold=backtrack_threshold,wiring_strategy='simple',simple_mode=True,backtrack_debounce=True)
    nav.run(max_steps=max_steps)
    annot=PhaseAnnotator(maze)
    records=[]
    for step,pos in enumerate(nav.path):
        phase=annot.label(step,pos, nav.path[:step+1])
        g=nav.gedig_history[step] if step < len(nav.gedig_history) else 0.0
        records.append({'step':step,'phase':phase,'gedig':g})
    stats=compute_phase_stats(records)
    t_stats=stats.get('terminal',{})
    return name, {'terminal_p10': t_stats.get('p10'), 'terminal_p25': t_stats.get('p25'), 'seed': seed}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[101,202,303,404,505])
    ap.add_argument('--noise', type=float, default=0.15, help='Gaussian noise std added to weights')
    ap.add_argument('--max_steps', type=int, default=120)
    ap.add_argument('--gedig_threshold', type=float, default=0.3)
    ap.add_argument('--backtrack_threshold', type=float, default=-0.2)
    ap.add_argument('--temperature', type=float, default=0.1)
    args=ap.parse_args()

    rows=[]
    for seed in args.seeds:
        for builder in MAZE_BUILDERS:
            name, rec = run_once(builder, seed, args.temperature, args.gedig_threshold, args.backtrack_threshold, args.max_steps, args.noise)
            rows.append({'maze': name, **rec})
            print(f"seed {seed} maze {name} p25={rec['terminal_p25']} p10={rec['terminal_p10']}")

    # aggregate
    import statistics, collections
    agg=collections.defaultdict(list)
    for r in rows:
        agg[(r['maze'],'p25')].append(r['terminal_p25'])
        agg[(r['maze'],'p10')].append(r['terminal_p10'])
    summary=[]
    for (maze, kind), vals in agg.items():
        vals=[v for v in vals if v is not None]
        if not vals: continue
        summary.append({'maze': maze,'kind': kind,'mean': float(statistics.fmean(vals)),'stdev': float(statistics.pstdev(vals)) if len(vals)>1 else 0.0,'min': float(min(vals)),'max': float(max(vals))})

    ts=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir=os.path.join('results','maze_report', f'deadend_variability_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir,'variability.json'),'w') as f:
        json.dump({'params': vars(args),'rows': rows,'summary': summary}, f, indent=2)
    print(f'Variability results -> {out_dir}')

if __name__=='__main__':
    main()
