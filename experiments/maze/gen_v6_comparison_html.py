#!/usr/bin/env python3
"""Generate comparison HTML for baseline vs extended (graph-persistent DG) experiments."""
import json, glob, os, html, sys

OUTDIR = "results/graph_persistent_dg/v6_perseed"
HTML_OUT = os.path.join(OUTDIR, "comparison.html")


def load_seed_data(mode, seed):
    path = os.path.join(OUTDIR, f"{mode}_seed{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    s = d.get("summary", {})
    ws = d.get("warmup_summary", {})
    cur = d.get("curriculum", {}).get("per_seed", {}).get(str(seed), {})
    w = cur.get("warmup", {})
    e = cur.get("eval", {})
    r = d.get("runs", [{}])[0]
    wr = d.get("warmup_runs", [{}])[0]
    accepted = r.get("accepted_series", [])
    g0 = r.get("g0_series", [])
    gmin = r.get("gmin_series", [])
    kstar = r.get("k_star_series", [])
    best_hop = r.get("multihop_best_hop", [])
    return {
        "seed": seed,
        "warmup_success": w.get("success", False),
        "warmup_steps": w.get("steps", 0),
        "eval_success": e.get("success", False),
        "eval_steps": e.get("steps", 0),
        "eval_edges": int(s.get("avg_edges", 0)),
        "warmup_edges": int(ws.get("avg_edges", 0)),
        "dead_end_steps": r.get("dead_end_steps", 0),
        "warmup_dead_ends": wr.get("dead_end_steps", 0),
        "dg_accepted": sum(1 for a in accepted if a),
        "dg_total": len(accepted),
        "g0_mean": sum(g0) / max(len(g0), 1),
        "gmin_mean": sum(gmin) / max(len(gmin), 1),
        "kstar_mean": sum(kstar) / max(len(kstar), 1),
        "best_hop_mean": sum(best_hop) / max(len(best_hop), 1),
        "g0_series": g0,
        "gmin_series": gmin,
        "kstar_series": kstar,
        "best_hop_series": best_hop,
    }


def main():
    seeds = set()
    for p in glob.glob(os.path.join(OUTDIR, "*_seed*.json")):
        name = os.path.basename(p)
        try:
            seed = int(name.split("seed")[1].split(".")[0])
            seeds.add(seed)
        except Exception:
            pass
    seeds = sorted(seeds)

    baseline_data = {}
    extended_data = {}
    for s in seeds:
        b = load_seed_data("baseline", s)
        e = load_seed_data("extended", s)
        if b:
            baseline_data[s] = b
        if e:
            extended_data[s] = e

    # Summary stats
    both_seeds = sorted(set(baseline_data.keys()) & set(extended_data.keys()))
    ext_only_seeds = sorted(set(extended_data.keys()) - set(baseline_data.keys()))

    b_success = sum(1 for s in both_seeds if baseline_data[s]["eval_success"])
    e_success_both = sum(1 for s in both_seeds if extended_data[s]["eval_success"])
    e_success_all = sum(1 for s in extended_data if extended_data[s]["eval_success"])
    b_total = len(both_seeds)
    e_total = len(extended_data)

    b_avg_steps = sum(baseline_data[s]["eval_steps"] for s in both_seeds) / max(b_total, 1)
    e_avg_steps = sum(extended_data[s]["eval_steps"] for s in both_seeds) / max(b_total, 1)

    b_avg_dead = sum(baseline_data[s]["dead_end_steps"] for s in both_seeds) / max(b_total, 1)
    e_avg_dead = sum(extended_data[s]["dead_end_steps"] for s in both_seeds) / max(b_total, 1)

    # Build per-seed comparison JSON for charts
    chart_data = []
    for s in sorted(set(list(baseline_data.keys()) + list(extended_data.keys()))):
        row = {"seed": s}
        if s in baseline_data:
            d = baseline_data[s]
            row["b_success"] = d["eval_success"]
            row["b_steps"] = d["eval_steps"]
            row["b_edges"] = d["eval_edges"]
            row["b_dead"] = d["dead_end_steps"]
            row["b_dg_rate"] = d["dg_accepted"] / max(d["dg_total"], 1)
            row["b_kstar"] = d["kstar_mean"]
            row["b_hop"] = d["best_hop_mean"]
            row["b_warmup_ok"] = d["warmup_success"]
            row["b_warmup_steps"] = d["warmup_steps"]
        if s in extended_data:
            d = extended_data[s]
            row["e_success"] = d["eval_success"]
            row["e_steps"] = d["eval_steps"]
            row["e_edges"] = d["eval_edges"]
            row["e_dead"] = d["dead_end_steps"]
            row["e_dg_rate"] = d["dg_accepted"] / max(d["dg_total"], 1)
            row["e_kstar"] = d["kstar_mean"]
            row["e_hop"] = d["best_hop_mean"]
            row["e_warmup_ok"] = d["warmup_success"]
            row["e_warmup_steps"] = d["warmup_steps"]
        chart_data.append(row)

    chart_json = json.dumps(chart_data)

    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8"/>
<title>Graph-Persistent DG: Baseline vs Extended Comparison</title>
<style>
:root {{ color-scheme: light; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Hiragino Sans', sans-serif;
  margin: 0; padding: 28px 36px 48px; background: #f4f6fb; color: #1f2937;
}}
h1 {{ margin: 0 0 8px; font-size: 1.8rem; }}
.subtitle {{ color: #6b7280; font-size: 0.9rem; margin-bottom: 24px; }}
.top-summary {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); margin-bottom: 28px; }}
.card {{
  background: #fff; border: 1px solid #dbe1f1; border-radius: 14px;
  padding: 16px 18px; box-shadow: 0 6px 14px rgba(30,50,90,0.07);
}}
.card-title {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; color: #5c6c83; margin-bottom: 6px; }}
.card-value {{ font-size: 1.5rem; font-weight: 600; color: #13213a; }}
.card-sub {{ font-size: 0.78rem; color: #6b7280; margin-top: 4px; }}
.improve {{ color: #059669; }}
.worse {{ color: #dc2626; }}
canvas {{ display: block; width: 100%; background: #fff; border: 1px solid #dde3f0; border-radius: 14px; box-shadow: inset 0 1px 1px rgba(30,60,90,0.05); }}
.chart-section {{ margin-bottom: 28px; }}
.chart-section h2 {{ font-size: 1.1rem; margin: 0 0 10px; }}
.dual {{ display: grid; gap: 18px; grid-template-columns: 1fr 1fr; }}
.legend {{ display: flex; gap: 18px; font-size: 0.82rem; color: #5c6c83; margin-bottom: 8px; }}
.legend span::before {{ content: ''; display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 5px; vertical-align: middle; }}
.legend .baseline::before {{ background: #94a3b8; }}
.legend .extended::before {{ background: #3b82f6; }}
.legend .ext-fail::before {{ background: #fbbf24; }}
table.seed-table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
.seed-table th, .seed-table td {{ padding: 5px 8px; border-bottom: 1px solid #edf1fb; text-align: center; }}
.seed-table th {{ background: #f8fafc; color: #475569; font-weight: 500; position: sticky; top: 0; }}
.seed-table tr:hover {{ background: #f0f4ff; }}
.seed-table .ok {{ color: #059669; font-weight: 600; }}
.seed-table .fail {{ color: #dc2626; font-weight: 600; }}
.seed-table .improved {{ background: #ecfdf5; }}
.seed-table .regressed {{ background: #fef2f2; }}
.table-wrap {{ max-height: 600px; overflow-y: auto; border: 1px solid #dbe1f1; border-radius: 14px; }}
</style>
</head>
<body>
<h1>Graph-Persistent DG: Baseline vs Extended</h1>
<div class="subtitle">Wake-Sleep-Wake architecture | 25x25 maze | max_hops=15 sp-cand-topk=5 | reward: novel=+0.2 revisit=-0.4</div>

<div class="top-summary">
  <div class="card">
    <div class="card-title">Seeds Compared</div>
    <div class="card-value">{len(both_seeds)}</div>
    <div class="card-sub">Extended only: {len(ext_only_seeds)}</div>
  </div>
  <div class="card">
    <div class="card-title">Success Rate (Eval)</div>
    <div class="card-value">{e_success_both}/{b_total} vs {b_success}/{b_total}</div>
    <div class="card-sub">Extended vs Baseline</div>
  </div>
  <div class="card">
    <div class="card-title">Avg Eval Steps</div>
    <div class="card-value">{e_avg_steps:.0f} vs {b_avg_steps:.0f}</div>
    <div class="card-sub">Extended vs Baseline</div>
  </div>
  <div class="card">
    <div class="card-title">Avg Dead-Ends</div>
    <div class="card-value">{e_avg_dead:.1f} vs {b_avg_dead:.1f}</div>
    <div class="card-sub">Extended vs Baseline (eval)</div>
  </div>
  <div class="card">
    <div class="card-title">Extended Total</div>
    <div class="card-value">{e_success_all}/{e_total}</div>
    <div class="card-sub">All extended seeds</div>
  </div>
</div>

<div class="chart-section">
<h2>Eval Steps per Seed</h2>
<div class="legend"><span class="baseline">Baseline (8D)</span><span class="extended">Extended (10D+propagated)</span></div>
<canvas id="stepsChart" height="250"></canvas>
</div>

<div class="dual">
<div class="chart-section">
<h2>DG Acceptance Rate</h2>
<canvas id="dgChart" height="220"></canvas>
</div>
<div class="chart-section">
<h2>Best Hop (Multi-hop Usage)</h2>
<canvas id="hopChart" height="220"></canvas>
</div>
</div>

<div class="dual">
<div class="chart-section">
<h2>Dead-End Steps (Eval)</h2>
<canvas id="deadChart" height="220"></canvas>
</div>
<div class="chart-section">
<h2>Eval Edges</h2>
<canvas id="edgeChart" height="220"></canvas>
</div>
</div>

<div class="chart-section">
<h2>Per-Seed Detail Table</h2>
<div class="table-wrap">
<table class="seed-table">
<thead><tr>
<th>Seed</th>
<th colspan="2">Warmup</th>
<th colspan="2">Eval Success</th>
<th colspan="2">Eval Steps</th>
<th colspan="2">Dead-Ends</th>
<th colspan="2">Edges</th>
<th colspan="2">DG Rate</th>
<th colspan="2">Best Hop</th>
</tr>
<tr>
<th></th>
<th>Base</th><th>Ext</th>
<th>Base</th><th>Ext</th>
<th>Base</th><th>Ext</th>
<th>Base</th><th>Ext</th>
<th>Base</th><th>Ext</th>
<th>Base</th><th>Ext</th>
<th>Base</th><th>Ext</th>
</tr></thead>
<tbody id="seedTableBody"></tbody>
</table>
</div>
</div>

<script>
const DATA = {chart_json};

function getCanvas(id) {{
  const c = document.getElementById(id);
  const dpr = window.devicePixelRatio || 1;
  const rect = c.getBoundingClientRect();
  c.width = rect.width * dpr;
  c.height = rect.height * dpr;
  const ctx = c.getContext('2d');
  ctx.scale(dpr, dpr);
  return {{ ctx, w: rect.width, h: rect.height }};
}}

function drawBarChart(id, bKey, eKey, opts = {{}}) {{
  const {{ ctx, w, h }} = getCanvas(id);
  const pad = {{ t: 20, b: 40, l: 50, r: 20 }};
  const pw = w - pad.l - pad.r;
  const ph = h - pad.t - pad.b;

  let maxVal = opts.maxVal || 0;
  DATA.forEach(d => {{
    if (d[bKey] != null) maxVal = Math.max(maxVal, d[bKey]);
    if (d[eKey] != null) maxVal = Math.max(maxVal, d[eKey]);
  }});
  if (maxVal === 0) maxVal = 1;
  maxVal *= 1.1;

  const n = DATA.length;
  const barW = Math.max(2, pw / n / 2.5);

  // Grid
  ctx.strokeStyle = '#e5e7eb'; ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {{
    const y = pad.t + ph * (1 - i / 4);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillStyle = '#9ca3af'; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
    ctx.fillText((maxVal * i / 4).toFixed(opts.decimals || 0), pad.l - 4, y + 3);
  }}

  DATA.forEach((d, i) => {{
    const x = pad.l + (i + 0.5) * pw / n;
    // Baseline bar
    if (d[bKey] != null) {{
      const bh = (d[bKey] / maxVal) * ph;
      ctx.fillStyle = d.b_success === false ? '#fca5a5' : '#94a3b8';
      ctx.fillRect(x - barW, pad.t + ph - bh, barW, bh);
    }}
    // Extended bar
    if (d[eKey] != null) {{
      const eh = (d[eKey] / maxVal) * ph;
      ctx.fillStyle = d.e_success === false ? '#fbbf24' : '#3b82f6';
      ctx.fillRect(x, pad.t + ph - eh, barW, eh);
    }}
    // Seed label
    if (n <= 30 || i % 5 === 0) {{
      ctx.fillStyle = '#6b7280'; ctx.font = '9px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText(d.seed, x, h - pad.b + 14);
    }}
  }});
}}

function buildTable() {{
  const tbody = document.getElementById('seedTableBody');
  DATA.forEach(d => {{
    const tr = document.createElement('tr');
    const bOk = d.b_success, eOk = d.e_success;
    if (eOk && !bOk) tr.classList.add('improved');
    if (bOk && !eOk) tr.classList.add('regressed');

    const cell = (v, cls) => `<td class="${{cls || ''}}">${{v != null ? v : '-'}}</td>`;
    const okFail = (v) => v == null ? '-' : (v ? '<span class="ok">OK</span>' : '<span class="fail">FAIL</span>');
    const warmupLabel = (ok, steps) => ok == null ? '-' : (ok ? `<span class="ok">${{steps}}</span>` : `<span class="fail">${{steps}}</span>`);
    const pct = (v) => v != null ? (v * 100).toFixed(0) + '%' : '-';
    const f1 = (v) => v != null ? v.toFixed(1) : '-';

    tr.innerHTML = `
      <td><b>${{d.seed}}</b></td>
      <td>${{warmupLabel(d.b_warmup_ok, d.b_warmup_steps)}}</td>
      <td>${{warmupLabel(d.e_warmup_ok, d.e_warmup_steps)}}</td>
      <td>${{okFail(d.b_success)}}</td>
      <td>${{okFail(d.e_success)}}</td>
      <td>${{d.b_steps != null ? d.b_steps : '-'}}</td>
      <td>${{d.e_steps != null ? d.e_steps : '-'}}</td>
      <td>${{d.b_dead != null ? d.b_dead : '-'}}</td>
      <td>${{d.e_dead != null ? d.e_dead : '-'}}</td>
      <td>${{d.b_edges != null ? d.b_edges : '-'}}</td>
      <td>${{d.e_edges != null ? d.e_edges : '-'}}</td>
      <td>${{pct(d.b_dg_rate)}}</td>
      <td>${{pct(d.e_dg_rate)}}</td>
      <td>${{f1(d.b_hop)}}</td>
      <td>${{f1(d.e_hop)}}</td>
    `;
    tbody.appendChild(tr);
  }});
}}

drawBarChart('stepsChart', 'b_steps', 'e_steps');
drawBarChart('dgChart', 'b_dg_rate', 'e_dg_rate', {{ maxVal: 1.05, decimals: 1 }});
drawBarChart('hopChart', 'b_hop', 'e_hop', {{ decimals: 1 }});
drawBarChart('deadChart', 'b_dead', 'e_dead');
drawBarChart('edgeChart', 'b_edges', 'e_edges');
buildTable();
</script>
</body>
</html>"""

    with open(HTML_OUT, "w") as f:
        f.write(html_content)
    print(f"Wrote {HTML_OUT}")
    print(f"  Seeds with both: {len(both_seeds)}")
    print(f"  Extended only: {len(ext_only_seeds)}")
    print(f"  Baseline success: {b_success}/{b_total}")
    print(f"  Extended success: {e_success_both}/{b_total} (of compared), {e_success_all}/{e_total} (all)")


if __name__ == "__main__":
    main()
