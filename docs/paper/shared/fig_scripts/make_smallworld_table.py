#!/usr/bin/env python3
"""
Compute Small-world metrics from exported doc co-occurrence edges and write a LaTeX table.

Input (default): experiments/rag-dynamic-db-v3/insight_eval/results/exports/doc_cooccurrence_edges.csv
Columns: src_id, dst_id, weight

Output: docs/paper/templates/tab_smallworld.tex with avg degree, clustering coef, avg shortest path length
"""
from __future__ import annotations
from pathlib import Path
import sys
import csv


def load_edges(p: Path):
    edges = []
    if p.exists():
        with p.open("r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    u = row.get("src_id") or row.get("source") or row.get("src")
                    v = row.get("dst_id") or row.get("target") or row.get("dst")
                    w = float(row.get("weight") or 1.0)
                except Exception:
                    continue
                if u is None or v is None:
                    continue
                edges.append((str(u), str(v), w))
    return edges


def compute_metrics(edges):
    try:
        import networkx as nx
    except Exception:
        return None
    G = nx.Graph()
    for u, v, w in edges:
        if u == v:
            continue
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)
    if G.number_of_nodes() == 0:
        return None
    # Largest connected component
    if not nx.is_connected(G):
        cc = max(nx.connected_components(G), key=len)
        H = G.subgraph(cc).copy()
    else:
        H = G
    n = H.number_of_nodes()
    m = H.number_of_edges()
    avg_deg = 2 * m / n if n else 0.0
    try:
        clustering = nx.average_clustering(H)
    except Exception:
        clustering = float("nan")
    try:
        spl = nx.average_shortest_path_length(H)
    except Exception:
        spl = float("nan")
    return dict(n=n, m=m, avg_deg=avg_deg, clustering=clustering, avg_path=spl)


def write_table(tex_path: Path, metrics: dict | None):
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    with tex_path.open("w") as f:
        f.write("% Auto-generated Small-world metrics table\n")
        f.write("\\begin{table}[H]\n\\centering\n\\small\n")
        f.write("\\begin{tabular}{lrr}\\toprule\n")
        f.write("指標 & 値 & 備考 \\\\ \\midrule\n")
        if metrics:
            f.write(f"ノード数 $n$ & {metrics['n']} & 最大連結成分 \\\\ \n")
            f.write(f"エッジ数 $m$ & {metrics['m']} & 無向単純グラフ \\\\ \n")
            f.write(f"平均次数 $\\bar{{k}}$ & {metrics['avg_deg']:.2f} & $2m/n$ \\\\ \n")
            f.write(f"クラスタ係数 $C$ & {metrics['clustering']:.3f} & NetworkX.avg\\_clustering \\\\ \n")
            f.write(f"平均最短距離 $L$ & {metrics['avg_path']:.3f} & 最大連結成分 \\\\ \\bottomrule\n")
        else:
            f.write("データ未検出 & - & エクスポート後に再生成してください。\\\\ \\bottomrule\n")
        f.write("\\end{tabular}\n\\caption{Small‑world 指標（RAG グラフ近似）。共起エッジから構成した無向グラフに基づく推定。}\\label{tab:smallworld}\n\\end{table}\n")


def main(argv):
    repo = Path(__file__).resolve().parents[3]
    in_edges = repo / "experiments/rag-dynamic-db-v3/insight_eval/results/exports/doc_cooccurrence_edges.csv"
    out_tex = repo / "docs/paper/templates/tab_smallworld.tex"
    edges = load_edges(in_edges)
    metrics = compute_metrics(edges)
    write_table(out_tex, metrics)
    print(f"✅ Wrote {out_tex}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
