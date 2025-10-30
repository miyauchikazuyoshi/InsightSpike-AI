#!/usr/bin/env python3
"""Dead-end / branch heuristic probe.

Runs Simple Mode navigator on a set of tiny crafted mazes capturing canonical phases:
  - branch_entry: 分岐進入直後 (degree>2 or first reduction from hub)
  - deepening: 分岐一本化後にまだ未探索伸長
  - terminal: 行き止まりセル (degree<=1 且つ 新規進展なし / 先が壁)
  - retreat: backtrack 相当 (path 上で過去位置へ後退中)

Outputs per maze:
  results/maze_report/deadend_probe_<timestamp>/<maze_name>/
    - trace.csv (step,x,y,gedig,phase,delta)
    - phase_stats.json (phase別 mean/std/min/max, 自然発生閾値候補)
    - plot.png (geDIG vs step with phases colored)
    - summary.md

"""
from __future__ import annotations
import os, sys, csv, json, argparse, datetime, math
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from navigation.maze_navigator import MazeNavigator  # type: ignore
from deadend_mini_mazes import MAZE_BUILDERS

# Simple structural helpers

def degree(maze: np.ndarray, pos):
    x,y=pos; d=0
    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        nx,ny=x+dx,y+dy
        if 0<=nx<maze.shape[1] and 0<=ny<maze.shape[0] and maze[ny,nx]==0:
            d+=1
    return d

class PhaseAnnotator:
    def __init__(self, maze):
        self.maze=maze
        self.prev_pos=None
        self.prev_deg=None
        self.visited=set()
        self.last_forward=True

    def label(self, step:int, pos, path_prefix):
        d=degree(self.maze,pos)
        phase='deepening'
        if step==0:
            phase='branch_entry'
        elif pos in self.visited:
            # moving along previous path; if heading toward earlier index => retreat
            prev_idx = next((i for i,p in enumerate(path_prefix[:-1]) if p==pos), None)
            if prev_idx is not None and prev_idx < len(path_prefix)-2:
                phase='retreat'
        elif d>=3:
            phase='branch_entry'
        elif d<=1:
            phase='terminal'
        # record
        self.visited.add(pos)
        self.prev_pos=pos; self.prev_deg=d
        return phase


def run_probe(maze, start, goal, temperature, gedig_threshold, backtrack_threshold, max_steps):
    weights=np.array([1.0,1.0,0.0,0.0,3.0,2.0,0.1,0.0])
    nav=MazeNavigator(maze, start, goal, weights=weights, temperature=temperature,
                      gedig_threshold=gedig_threshold, backtrack_threshold=backtrack_threshold,
                      wiring_strategy='simple', simple_mode=True, backtrack_debounce=True)
    nav.run(max_steps=max_steps)
    return nav


def compute_phase_stats(records):
    by={}  # phase -> list of gedig
    for r in records:
        by.setdefault(r['phase'], []).append(r['gedig'])
    stats={}
    for ph, vals in by.items():
        if not vals: continue
        arr=np.array(vals)
        stats[ph]={
            'count': int(arr.size),
            'mean': float(arr.mean()),
            'std': float(arr.std(ddof=0)),
            'min': float(arr.min()),
            'max': float(arr.max()),
            'p10': float(np.percentile(arr,10)),
            'p25': float(np.percentile(arr,25)),
            'p50': float(np.percentile(arr,50)),
            'p75': float(np.percentile(arr,75)),
            'p90': float(np.percentile(arr,90)),
        }
    # Natural candidate threshold suggestions (terminal低域 vs others)
    if 'terminal' in stats:
        term_p25=stats['terminal']['p25']
        others=[stats[p]['p50'] for p in stats if p!='terminal']
        if others:
            gap = min(others) - term_p25
        else:
            gap = 0.0
        stats['suggested_thresholds']={
            'terminal_p25': term_p25,
            'terminal_p10': stats['terminal']['p10'],
            'mid_gap_estimate': term_p25 + gap*0.5 if gap>0 else term_p25,
        }
    return stats


def _roc_auc(scores, labels):
    # Manual ROC AUC (labels 0/1)
    paired=sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    pos=sum(labels); neg=len(labels)-pos
    if pos==0 or neg==0:
        return None
    tp=0; fp=0; prev_score=None; auc=0.0; prev_tp_rate=0.0; prev_fp_rate=0.0
    for s,l in paired:
        if l==1: tp+=1
        else: fp+=1
        tp_rate=tp/pos; fp_rate=fp/neg
        # trapezoid
        auc += (fp_rate - prev_fp_rate) * (tp_rate + prev_tp_rate) / 2.0
        prev_tp_rate=tp_rate; prev_fp_rate=fp_rate
    return auc

def _best_linear_combo_auc(gedig_vals, deltas, labels):
    # search simple grid a*gedig + b*(-delta)
    grid=[0.0,0.25,0.5,0.75,1.0]
    best={'auc':None,'a':None,'b':None}
    # precompute transformed deltas (negative delta bigger near terminal)
    neg_delta=[-d for d in deltas]
    for a in grid:
        for b in grid:
            if a==0 and b==0:
                continue
            scores=[a*g + b*nd for g,nd in zip(gedig_vals, neg_delta)]
            auc=_roc_auc(scores, labels)
            if auc is None: continue
            if best['auc'] is None or auc>best['auc']:
                best={'auc':round(auc,4),'a':a,'b':b}
    return best


def plot_trace(out_png, records, backtrack_threshold):
    steps=[r['step'] for r in records]
    vals=[r['gedig'] for r in records]
    phases=[r['phase'] for r in records]
    colors={'branch_entry':'tab:blue','deepening':'tab:green','terminal':'tab:red','retreat':'tab:orange'}
    plt.figure(figsize=(7,3.2))
    for ph in colors:
        xs=[s for s,p in zip(steps,phases) if p==ph]
        ys=[v for v,p in zip(vals,phases) if p==ph]
        if xs:
            plt.scatter(xs, ys, s=18, c=colors[ph], label=ph, alpha=0.85, marker='o')
    plt.plot(steps, vals, color='gray', alpha=0.3, linewidth=1)
    plt.axhline(0.0, color='black', linewidth=0.7)
    if backtrack_threshold is not None:
        plt.axhline(backtrack_threshold, color='magenta', linestyle='--', linewidth=0.8, label='bt_thresh')
    plt.xlabel('step'); plt.ylabel('geDIG'); plt.legend(ncol=4, fontsize=8)
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()


def write_summary(out_md, maze_name, desc, stats, note):
    with open(out_md,'w') as f:
        f.write(f"# Dead-end Probe: {maze_name}\n\n")
        f.write(desc+'\n\n')
        if 'suggested_thresholds' in stats:
            sug=stats['suggested_thresholds']
            f.write('## Suggested (Phenomenological) Thresholds\n')
            f.write(f"- terminal p25: {sug['terminal_p25']:.4f}\n")
            f.write(f"- terminal p10: {sug['terminal_p10']:.4f}\n")
            f.write(f"- mid-gap estimate: {sug['mid_gap_estimate']:.4f}\n\n")
        f.write('## Phase Statistics\n')
        for ph, d in stats.items():
            if ph=='suggested_thresholds':
                continue
            f.write(f"### {ph}\n")
            f.write(', '.join(f"{k}={d[k]:.4f}" for k in ['count','mean','std','min','p10','p25','p50','p75','p90','max'] if k in d)+'\n')
        if 'classification' in stats:
            cls=stats['classification']
            f.write('\n## Terminal Classification\n')
            if 'auc_geDIG' in cls:
                f.write(f"auc_geDIG: {cls.get('auc_geDIG')}\\n")
            if 'auc_neg_delta' in cls:
                f.write(f"auc_neg_delta: {cls.get('auc_neg_delta')}\\n")
            if 'best_linear_combo' in cls:
                bl=cls['best_linear_combo']
                f.write(f"best_linear_combo_auc: {bl.get('auc')} (a={bl.get('a')}, b={bl.get('b')})\\n")
            f.write(f"suggested_threshold_mid_gap: {stats.get('suggested_thresholds',{}).get('mid_gap_estimate')}\\n")
        f.write('\n'+note+'\n')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--max_steps', type=int, default=400)
    ap.add_argument('--gedig_threshold', type=float, default=0.3)
    ap.add_argument('--backtrack_threshold', type=float, default=-0.2)
    ap.add_argument('--temperature', type=float, default=0.1)
    ap.add_argument('--note', type=str, default='Phenomenological dead-end probe.')
    args=ap.parse_args()

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join('results','maze_report', f'deadend_probe_{timestamp}')
    os.makedirs(base_dir, exist_ok=True)

    for builder in MAZE_BUILDERS:
        maze,start,goal,name,desc=builder()
        nav=run_probe(maze,start,goal,args.temperature,args.gedig_threshold,args.backtrack_threshold,args.max_steps)
        annot=PhaseAnnotator(maze)
        records=[]
        for step,pos in enumerate(nav.path):
            phase=annot.label(step,pos, nav.path[:step+1])
            gedig_val = nav.gedig_history[step] if step < len(nav.gedig_history) else 0.0
            records.append({'step':step,'x':pos[0],'y':pos[1],'gedig':gedig_val,'phase':phase})
        # delta
        for i in range(1,len(records)):
            records[i]['delta']=records[i]['gedig']-records[i-1]['gedig']
        records[0]['delta']=0.0
        stats=compute_phase_stats(records)
        # AUC: terminal (1) vs others (0)
        gedig_scores=[r['gedig'] for r in records]
        labels=[1 if r['phase']=='terminal' else 0 for r in records]
        auc_raw=_roc_auc(gedig_scores, labels)
        delta_scores=[r['delta'] for r in records]
        auc_delta=_roc_auc([-d for d in delta_scores], labels) if len(delta_scores)>1 else None
        best_lin=_best_linear_combo_auc(gedig_scores, delta_scores, labels)
        stats['classification']={}
        if auc_raw is not None:
            stats['classification']['auc_geDIG']=round(auc_raw,4)
        if auc_delta is not None:
            stats['classification']['auc_neg_delta']=round(auc_delta,4)
        if best_lin['auc'] is not None:
            stats['classification']['best_linear_combo']=best_lin
        out_dir=os.path.join(base_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        # CSV
        with open(os.path.join(out_dir,'trace.csv'),'w',newline='') as f:
            w=csv.DictWriter(f, fieldnames=['step','x','y','gedig','delta','phase'])
            w.writeheader(); w.writerows(records)
        # JSON stats
        with open(os.path.join(out_dir,'phase_stats.json'),'w') as f:
            json.dump(stats, f, indent=2)
        # Plot
        plot_trace(os.path.join(out_dir,'plot.png'), records, args.backtrack_threshold)
        # Summary
        write_summary(os.path.join(out_dir,'summary.md'), name, desc, stats, args.note)
        print(f'Finished {name}: steps={len(nav.path)} terminal_p25={stats.get("suggested_thresholds",{}).get("terminal_p25")}')

    print(f'Probe outputs in {base_dir}')

if __name__=='__main__':
    main()
