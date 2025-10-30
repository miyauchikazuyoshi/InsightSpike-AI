"""Secondary summarization / figure stub (optional).
Currently aggregates are produced in run_batch.py; this script can extend with CSV or plots.
"""
from __future__ import annotations
import json
import pathlib
from typing import Dict, Any, List

try:
    import matplotlib.pyplot as plt  # optional
except Exception:  # pragma: no cover
    plt = None


def load_json(p: pathlib.Path):
    return json.loads(p.read_text())


def plot_success_rate(by_policy: Dict[str, Any], out_dir: pathlib.Path):
    if plt is None:
        print("matplotlib not available; skipping plots")
        return
    policies = list(by_policy.keys())
    vals = [by_policy[p]['success_rate'] for p in policies]
    plt.figure(figsize=(6,4))
    plt.bar(policies, vals, color='#4C72B0')
    plt.ylabel('Success Rate')
    plt.title('Success Rate by Policy (R1)')
    plt.ylim(0,1)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    out_file = out_dir / 'success_rate.png'
    plt.savefig(out_file)
    print('Saved', out_file)

def plot_mean_steps(by_policy: Dict[str, Any], out_dir: pathlib.Path):
    if plt is None:
        return
    policies = list(by_policy.keys())
    vals = [by_policy[p]['mean_steps_success'] for p in policies]
    plt.figure(figsize=(6,4))
    plt.bar(policies, vals, color='#55A868')
    plt.ylabel('Mean Steps (Success Only)')
    plt.title('Steps to Goal (R1)')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    out_file = out_dir / 'mean_steps.png'
    plt.savefig(out_file)
    print('Saved', out_file)

def plot_auc_vs_eff(by_policy: Dict[str, Any], out_dir: pathlib.Path):
    if plt is None:
        return
    policies = list(by_policy.keys())
    aucs = [by_policy[p]['gedig_auc_mean'] for p in policies]
    effs = [by_policy[p]['path_efficiency_mean'] for p in policies]
    plt.figure(figsize=(5,5))
    plt.scatter(aucs, effs, s=80, c='#C44E52')
    for i,p in enumerate(policies):
        plt.text(aucs[i]+0.005, effs[i]+0.005, p, fontsize=9)
    plt.xlabel('Mean geDIG AUC')
    plt.ylabel('Mean Path Efficiency (Success)')
    plt.title('AUC vs Path Efficiency (R1)')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.tight_layout()
    out_file = out_dir / 'gedig_auc_vs_efficiency.png'
    plt.savefig(out_file)
    print('Saved', out_file)

def write_csv(by_policy: Dict[str, Any], out_dir: pathlib.Path):
    rows: List[str] = ["policy,success_rate,mean_steps_success,path_efficiency_mean,exploration_ratio_mean,gedig_auc_mean,gedig_auc_stdev"]
    for p,data in by_policy.items():
        rows.append(
            f"{p},{data['success_rate']},{data['mean_steps_success']},{data['path_efficiency_mean']},{data['exploration_ratio_mean']},{data['gedig_auc_mean']},{data['gedig_auc_stdev']}"
        )
    (out_dir / 'by_policy.csv').write_text('\n'.join(rows))
    print('Saved', out_dir / 'by_policy.csv')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args()
    summary_dir = pathlib.Path(args.results_dir) / 'summary'
    by_policy_p = summary_dir / 'by_policy.json'
    if not by_policy_p.exists():
        print('by_policy.json not found; run batch first')
        return
    by_policy = load_json(by_policy_p)
    fig_dir = summary_dir.parent / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    write_csv(by_policy, summary_dir)
    if not args.no_plots:
        plot_success_rate(by_policy, fig_dir)
        plot_mean_steps(by_policy, fig_dir)
        plot_auc_vs_eff(by_policy, fig_dir)
    else:
        print('Plot generation skipped (--no-plots)')

if __name__ == '__main__':
    main()
