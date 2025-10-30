#!/usr/bin/env python3
"""
Export CSVs for appendix from insight alignment JSON.

Inputs:
  - JSON path (default: results/outputs/rag_insight_alignment_ragv3.json)

Outputs (under results/exports/):
  - questions_prompts_answers.csv: question, prompt, answer, s, s0, delta_s, provider, model
  - docs_per_question.csv: q_idx, rank, text, c_value, similarity
  - doc_cooccurrence_edges.csv: src_id, dst_id, weight (co-occur in same question), nodes.csv with id, text_snippet
  - LaTeX preview: docs/paper/templates/tab_appendix_rag_csv_preview.tex (top-5 rows excerpts)
"""
from __future__ import annotations
from pathlib import Path
import json
import csv
import argparse
import hashlib


def doc_id(text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json", nargs="?", default=None)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    in_json = Path(args.json) if args.json else (repo / "experiments/rag-dynamic-db-v3/insight_eval/results/outputs/rag_insight_alignment_ragv3.json")
    outdir = repo / "experiments/rag-dynamic-db-v3/insight_eval/results/exports"
    outdir.mkdir(parents=True, exist_ok=True)
    js = json.loads(in_json.read_text())

    provider = js.get("provider")
    model = js.get("model")
    records = js.get("records") or []

    # 1) Q/P/A CSV
    qpa = outdir / "questions_prompts_answers.csv"
    with qpa.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["q_idx", "question", "prompt", "answer", "s", "s0", "delta_s", "provider", "model"])
        for i, r in enumerate(records):
            w.writerow([i, r.get("question"), r.get("prompt"), r.get("answer"), r.get("s"), r.get("s0"), r.get("delta_s"), provider, model])

    # 2) Docs per question
    dpq = outdir / "docs_per_question.csv"
    with dpq.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["q_idx", "rank", "text", "c_value", "similarity", "doc_id"])
        for i, r in enumerate(records):
            docs = r.get("docs") or []
            for rank, d in enumerate(docs, 1):
                text = d.get("text", "")
                cid = doc_id(text)
                w.writerow([i, rank, text, d.get("c_value"), d.get("similarity"), cid])

    # 3) Co-occurrence graph (nodes, edges)
    nodes_path = outdir / "nodes.csv"
    edges_path = outdir / "doc_cooccurrence_edges.csv"
    # Build co-occurrence counts per question
    from collections import defaultdict
    node_text = {}
    edge_count = defaultdict(int)
    for r in records:
        docs = r.get("docs") or []
        ids = []
        for d in docs:
            t = d.get("text", "")
            if not t:
                continue
            cid = doc_id(t)
            node_text[cid] = t
            ids.append(cid)
        # count unordered pairs
        ids = list(dict.fromkeys(ids))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = sorted((ids[i], ids[j]))
                edge_count[(a, b)] += 1
    # write nodes
    with nodes_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "text_snippet"])
        for cid, text in node_text.items():
            w.writerow([cid, text[:160].replace("\n", " ")])
    # write edges
    with edges_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_id", "dst_id", "weight"])
        for (a, b), wgt in edge_count.items():
            w.writerow([a, b, wgt])

    # 4) Tiny LaTeX preview table (top-5 Q/P/A rows)
    def latex_escape(s: str) -> str:
        repl = {
            "\\": r"\\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{" : r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        out = []
        for ch in s:
            out.append(repl.get(ch, ch))
        return "".join(out)

    tex_preview = repo / "docs/paper/templates/tab_appendix_rag_csv_preview.tex"
    with tex_preview.open("w") as f:
        f.write("% Auto-generated appendix CSV preview (top-5)\n")
        f.write("\\begin{table}[H]\\centering\\small\\begin{tabular}{rl}\\toprule\n")
        f.write("列 & ファイルパス \\\\ \\midrule\n")
        paths = [
            ("質問/プロンプト/回答", "experiments/rag-dynamic-db-v3/insight_eval/results/exports/questions_prompts_answers.csv"),
            ("ドキュメント(質問別)", "experiments/rag-dynamic-db-v3/insight_eval/results/exports/docs_per_question.csv"),
            ("共起グラフ(エッジ)", "experiments/rag-dynamic-db-v3/insight_eval/results/exports/doc_cooccurrence_edges.csv"),
        ]
        for label, path in paths:
            f.write(f"{latex_escape(label)} & {latex_escape(path)} \\\\ \n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\\caption{RAG 実験CSVの出力先一覧（本文§付録参照）。}\\label{tab:rag_csv_paths}\\end{table}\n")
        # small sample of Q/P/A
        f.write("\\begin{table}[H]\\centering\\small\\begin{tabular}{r p{0.42\\linewidth} p{0.42\\linewidth}}\\toprule\n")
        f.write("q\\_idx & 質問 & 回答(先頭160文字) \\\\ \\midrule\n")
        for i, r in enumerate(records[:5]):
            ans = latex_escape((r.get("answer") or "").replace("\n", " ")[:160])
            q = latex_escape((r.get("question") or "").replace("\n", " ")[:160])
            f.write(f"{i} & {q} & {ans} \\\\ \n")
        f.write("\\bottomrule\\end{tabular}\\caption{Q/Aサンプル(上位5件)。完全版はCSV参照。}\\label{tab:rag_csv_sample}\\end{table}\n")

    print(f"✅ Wrote {qpa}\n✅ Wrote {dpq}\n✅ Wrote {nodes_path}\n✅ Wrote {edges_path}\n✅ Wrote {tex_preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
