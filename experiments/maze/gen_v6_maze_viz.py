#!/usr/bin/env python3
"""Generate interactive maze visualization HTML for v6 per-seed comparison.

Usage:
    python gen_v6_maze_viz.py --seed 4 \
        --result-dir results/graph_persistent_dg/v6_perseed \
        --output results/graph_persistent_dg/v6_perseed/maze_viz_seed4.html
"""
import argparse
import json
import os

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def build_html(seed, result_dir, output_path):
    ext_full = load_json(os.path.join(result_dir, f"extended_seed{seed}_full.json"))
    ext_steps = load_json(os.path.join(result_dir, f"extended_seed{seed}_steps.json"))
    base_full = load_json(os.path.join(result_dir, f"baseline_seed{seed}_full.json"))
    base_steps = load_json(os.path.join(result_dir, f"baseline_seed{seed}_steps.json"))

    maze_src = ext_full or base_full
    if not maze_src:
        print(f"ERROR: No full JSON found for seed {seed}")
        return
    maze_data = maze_src["maze_data"][str(seed)]

    def split_phases(steps):
        if steps is None:
            return [], []
        warmup = [s for s in steps if s["episode_phase"] == "warmup"]
        eval_ = [s for s in steps if s["episode_phase"] == "eval"]
        return warmup, eval_

    ext_warmup, ext_eval = split_phases(ext_steps)
    base_warmup, base_eval = split_phases(base_steps)

    def slim_steps(steps):
        out = []
        visited = {}
        for s in steps:
            pos = tuple(s["position"])
            pos_key = f"{pos[0]},{pos[1]}"
            visit_n = visited.get(pos_key, 0) + 1
            visited[pos_key] = visit_n
            novel = visit_n == 1

            # Extract dim9 per direction from ranked_candidates
            d9 = {}
            d8 = {}
            rc = s.get("ranked_candidates", [])
            for c in rc:
                if str(c.get("origin", "")).startswith("obs"):
                    av = c.get("abs_vector", [])
                    al = c.get("action_label", "")
                    if al:
                        d9[al] = round(av[9], 4) if len(av) > 9 else 0
                        d8[al] = round(av[8], 4) if len(av) > 8 else 0

            out.append({
                "step": s["step"],
                "pos": s["position"],
                "action": s.get("action", ""),
                "moved": s.get("moved", True),
                "hop": s.get("best_hop", 0),
                "dged": round(s.get("delta_ged", 0), 8),
                "dig": round(s.get("delta_ig", 0), 8),
                "g0": round(s.get("g0", 0), 8),
                "gmin": round(s.get("gmin", 0), 8),
                "ag": s.get("ag_fire", False),
                "dg": s.get("dg_fire", False),
                "dead": s.get("is_dead_end", False),
                "edges": s.get("graph_edges", 0) if isinstance(s.get("graph_edges"), (int, float)) else len(s.get("graph_edges", [])),
                "novel": novel,
                "visits": visit_n,
                "d9": d9,  # dim9 (propagated) per direction
                "d8": d8,  # dim8 (reward) per direction
            })
        return out

    def get_summary(full_json):
        if not full_json:
            return None
        run = full_json["runs"][0] if full_json.get("runs") else None
        warmup_run = full_json["warmup_runs"][0] if full_json.get("warmup_runs") else None
        return {
            "success": run["success"] if run else False,
            "steps": run["steps"] if run else 0,
            "edges": run["edges"] if run else 0,
            "warmup_success": warmup_run["success"] if warmup_run else False,
            "warmup_steps": warmup_run["steps"] if warmup_run else 0,
        }

    sz = maze_data.get("size", len(maze_data["layout"]))
    if isinstance(sz, list):
        sz = sz[0]

    data_js = {
        "seed": seed,
        "maze": {
            "layout": maze_data["layout"],
            "start": maze_data["start_pos"],
            "goal": maze_data["goal_pos"],
            "size": sz,
        },
        "extended": {
            "warmup": slim_steps(ext_warmup),
            "eval": slim_steps(ext_eval),
            "summary": get_summary(ext_full),
        },
        "baseline": {
            "warmup": slim_steps(base_warmup),
            "eval": slim_steps(base_eval),
            "summary": get_summary(base_full),
        },
    }

    data_json = json.dumps(data_js, separators=(",", ":"))

    html = TEMPLATE.replace("__DATA_JSON__", data_json).replace("__SEED__", str(seed))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Written: {output_path}")
    sz_bytes = os.path.getsize(output_path)
    print(f"Size: {sz_bytes:,} bytes ({sz_bytes/1024:.1f} KB)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8"/>
<title>Maze Viz — Seed __SEED__ — Baseline vs Extended</title>
<style>
:root { color-scheme: light; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Hiragino Sans', 'Noto Sans JP', sans-serif;
  margin: 0; padding: 24px 28px 48px; background: #f4f6fb; color: #1f2937;
}
h1 { margin: 0 0 16px; font-size: 1.8rem; }
h3 { margin: 0 0 8px; font-size: 1.05rem; }
.top-summary { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); margin-bottom: 16px; }
.card {
  background: #fff; border: 1px solid #dbe1f1; border-radius: 14px;
  padding: 14px 16px; box-shadow: 0 6px 14px rgba(30,50,90,0.07);
}
.card-title { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; color: #5c6c83; margin-bottom: 4px; }
.card-value { font-size: 1.4rem; font-weight: 600; color: #13213a; }
.card-value.ok { color: #16a34a; }
.card-value.fail { color: #dc2626; }
.controls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.controls label { font-size: 0.82rem; color: #475569; }
.controls select { padding: 3px 6px; border-radius: 6px; border: 1px solid #cbd5e1; font-size: 0.82rem; }
.controls input[type=range] { flex: 1; min-width: 180px; }
.controls button {
  border: none; background: #1c7ed6; color: #fff;
  padding: 5px 12px; border-radius: 999px; font-size: 0.82rem; cursor: pointer;
}
.controls button.paused { background: #394867; }
.controls .step-btn { background: #475569; }
.controls span { font-size: 0.8rem; color: #4b5563; min-width: 76px; text-align: right; }
.maze-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
.maze-panel { text-align: center; }
canvas { display: block; margin: 0 auto; background: #fff; border: 1px solid #dde3f0; border-radius: 10px; }
.step-detail { display: grid; gap: 8px; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); margin-top: 10px; }
.step-box { background: #fff; border: 1px solid #dbe1f1; border-radius: 8px; padding: 8px 10px; }
.step-box h4 { margin: 0 0 2px; font-size: 0.72rem; color: #495365; text-transform: uppercase; letter-spacing: 0.04em; }
.step-box span { font-size: 0.92rem; font-weight: 600; color: #101828; }
.step-box span.pos { color: #16a34a; }
.step-box span.neg { color: #dc2626; }
.legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 0.78rem; color: #5c677d; margin: 6px 0; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 3px; vertical-align: middle; }
.event-log { max-height: 180px; overflow-y: auto; font-size: 0.78rem; font-family: monospace; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px; margin-top: 8px; }
.event-log .ev-novel { color: #16a34a; }
.event-log .ev-revisit { color: #e67700; }
.event-log .ev-dead { color: #dc2626; font-weight: 600; }
</style>
</head>
<body>
<h1>Maze Seed __SEED__ — Baseline vs Extended (10D)</h1>
<div class="top-summary" id="summary-cards"></div>

<div class="card" style="margin-bottom:14px; padding:12px 16px;">
  <div class="controls">
    <label>Phase:</label>
    <select id="phase-sel">
      <option value="warmup">Warmup (Wake1)</option>
      <option value="eval" selected>Eval (Wake2)</option>
    </select>
    <button id="play-btn" type="button">▶ Play</button>
    <button id="prev-btn" type="button" class="step-btn">⟵</button>
    <button id="next-btn" type="button" class="step-btn">⟶</button>
    <input type="range" id="step-slider" min="0" max="0" value="0"/>
    <span id="step-counter">0 / 0</span>
    <label>Speed:<select id="speed-sel">
      <option value="100">x4</option>
      <option value="200">x2</option>
      <option value="500" selected>x1</option>
      <option value="1000">x0.5</option>
    </select></label>
    <label><input type="checkbox" id="overlay-toggle"/> Bias overlay</label>
  </div>
</div>

<div class="maze-row">
  <div class="maze-panel card">
    <h3 id="title-base">Baseline (8D standard)</h3>
    <canvas id="maze-base" width="560" height="560"></canvas>
    <div class="step-detail" id="detail-base"></div>
    <div class="event-log" id="events-base"></div>
  </div>
  <div class="maze-panel card">
    <h3 id="title-ext">Extended (10D + Sleep propagation)</h3>
    <canvas id="maze-ext" width="560" height="560"></canvas>
    <div class="step-detail" id="detail-ext"></div>
    <div class="event-log" id="events-ext"></div>
  </div>
</div>

<div class="legend">
  <span><i style="background:#d0ebff;border:1px solid #74c0fc;"></i>Start</span>
  <span><i style="background:#ffdce5;border:1px solid #ffa8c5;"></i>Goal</span>
  <span><i style="background:#16a34a;"></i>Novel (1st visit)</span>
  <span><i style="background:#e67700;"></i>Revisit</span>
  <span><i style="background:#dc2626;"></i>Dead-end</span>
  <span><i style="background:#f59f00;"></i>Current pos</span>
  <span><i style="background:rgba(34,197,94,0.5);"></i>dim9 bias (overlay)</span>
</div>

<div class="maze-row">
  <div class="card">
    <h3>Edge Count Growth</h3>
    <canvas id="chart-edges" width="560" height="180"></canvas>
  </div>
  <div class="card">
    <h3>Best Hop (k*)</h3>
    <canvas id="chart-hop" width="560" height="180"></canvas>
  </div>
</div>

<script>
const D = __DATA_JSON__;
const DIR_DELTA = {up:[-1,0], down:[1,0], left:[0,-1], right:[0,1]};

let phase = 'eval', stepIdx = 0, playing = false, playTimer = null;
const $ = id => document.getElementById(id);
const phaseSel=$('phase-sel'), slider=$('step-slider'), counter=$('step-counter');
const playBtn=$('play-btn'), prevBtn=$('prev-btn'), nextBtn=$('next-btn');
const speedSel=$('speed-sel'), overlayTgl=$('overlay-toggle');
const canvasBase=$('maze-base'), canvasExt=$('maze-ext');
const ctxBase=canvasBase.getContext('2d'), ctxExt=canvasExt.getContext('2d');

function getSteps(mode) { return D[mode][phase] || []; }
function maxSteps() { return Math.max(getSteps('baseline').length, getSteps('extended').length) - 1; }

// ---------- Summary ----------
function renderSummary() {
  const el = $('summary-cards');
  const bs = D.baseline.summary, es = D.extended.summary;
  function c(t,v,cls) { return `<div class="card"><div class="card-title">${t}</div><div class="card-value ${cls||''}">${v}</div></div>`; }
  let h = c('Seed', D.seed) + c('Maze', D.maze.size+'x'+D.maze.size);
  if (bs) {
    h += c('Base Warmup', (bs.warmup_success?'OK':'FAIL')+'('+bs.warmup_steps+')', bs.warmup_success?'ok':'fail');
    h += c('Base Eval', (bs.success?'OK':'FAIL')+'('+bs.steps+')', bs.success?'ok':'fail');
  }
  if (es) {
    h += c('Ext Warmup', (es.warmup_success?'OK':'FAIL')+'('+es.warmup_steps+')', es.warmup_success?'ok':'fail');
    h += c('Ext Eval', (es.success?'OK':'FAIL')+'('+es.steps+')', es.success?'ok':'fail');
  }
  el.innerHTML = h;
}

// ---------- Maze drawing ----------
function drawMaze(ctx, canvas, steps, curStep, isExtended) {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const layout = D.maze.layout;
  const rows = layout.length, cols = layout[0].length;
  const pad = 20;
  const cs = Math.min((W - pad*2) / cols, (H - pad*2) / rows);
  const ox = (W - cs*cols)/2, oy = (H - cs*rows)/2;

  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);

  // Cells
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      ctx.fillStyle = layout[r][c] === 1 ? '#1f2933' : '#f8fafc';
      ctx.fillRect(ox+c*cs, oy+r*cs, cs, cs);
    }

  // Start/Goal
  ctx.fillStyle='#d0ebff'; ctx.fillRect(ox+D.maze.start[1]*cs, oy+D.maze.start[0]*cs, cs, cs);
  ctx.fillStyle='#ffdce5'; ctx.fillRect(ox+D.maze.goal[1]*cs, oy+D.maze.goal[0]*cs, cs, cs);

  if (!steps || !steps.length) {
    ctx.fillStyle='#94a3b8'; ctx.font='16px sans-serif'; ctx.textAlign='center';
    ctx.fillText('No data', W/2, H/2); return;
  }
  const cap = Math.min(curStep, steps.length-1);

  // --- Novel / Revisit / Dead-end cell coloring ---
  const cellState = {}; // key -> {visits, dead}
  for (let i = 0; i <= cap; i++) {
    const s = steps[i], k = s.pos[0]+','+s.pos[1];
    if (!cellState[k]) cellState[k] = {visits:0, dead:false};
    cellState[k].visits++;
    if (s.dead) cellState[k].dead = true;
  }
  for (const [key, st] of Object.entries(cellState)) {
    const [r,c] = key.split(',').map(Number);
    if (st.dead) {
      ctx.fillStyle = 'rgba(220, 38, 38, 0.25)';
    } else if (st.visits === 1) {
      ctx.fillStyle = 'rgba(22, 163, 74, 0.15)';
    } else {
      const alpha = Math.min(0.45, 0.12 + 0.33 * Math.min(st.visits / 8, 1));
      ctx.fillStyle = `rgba(230, 119, 0, ${alpha})`;
    }
    ctx.fillRect(ox+c*cs, oy+r*cs, cs, cs);
  }

  // --- Bias overlay (dim9 propagation gradient) ---
  if (overlayTgl.checked && isExtended) {
    // Build a map of max dim9 seen at each cell from all steps up to cap
    const dim9map = {};
    for (let i = 0; i <= cap; i++) {
      const s = steps[i], d9 = s.d9 || {};
      for (const [dir, val] of Object.entries(d9)) {
        if (val > 0) {
          const delta = DIR_DELTA[dir];
          if (!delta) continue;
          const tr = s.pos[0]+delta[0], tc = s.pos[1]+delta[1];
          const tk = tr+','+tc;
          dim9map[tk] = Math.max(dim9map[tk]||0, val);
        }
      }
    }
    for (const [key, val] of Object.entries(dim9map)) {
      const [r,c] = key.split(',').map(Number);
      const alpha = Math.min(0.55, val * 0.55);
      ctx.fillStyle = `rgba(34, 197, 94, ${alpha})`;
      ctx.fillRect(ox+c*cs, oy+r*cs, cs, cs);
    }
  }

  // Grid
  ctx.strokeStyle='#e2e8f0'; ctx.lineWidth=0.5;
  for (let r=0;r<=rows;r++){ctx.beginPath();ctx.moveTo(ox,oy+r*cs);ctx.lineTo(ox+cols*cs,oy+r*cs);ctx.stroke();}
  for (let c=0;c<=cols;c++){ctx.beginPath();ctx.moveTo(ox+c*cs,oy);ctx.lineTo(ox+c*cs,oy+rows*cs);ctx.stroke();}

  // Path line
  ctx.strokeStyle='rgba(76,110,245,0.5)'; ctx.lineWidth=Math.max(1.5, cs*0.18);
  ctx.lineJoin='round'; ctx.lineCap='round'; ctx.beginPath();
  ctx.moveTo(ox+(D.maze.start[1]+0.5)*cs, oy+(D.maze.start[0]+0.5)*cs);
  for (let i=0;i<=cap;i++){const p=steps[i].pos; ctx.lineTo(ox+(p[1]+0.5)*cs, oy+(p[0]+0.5)*cs);}
  ctx.stroke();

  // --- Direction arrows showing dim9 bias at current step ---
  if (isExtended) {
    const s = steps[cap], d9 = s.d9 || {};
    const cx0 = ox + (s.pos[1]+0.5)*cs, cy0 = oy + (s.pos[0]+0.5)*cs;
    for (const [dir, val] of Object.entries(d9)) {
      if (val <= 0) continue;
      const delta = DIR_DELTA[dir];
      if (!delta) continue;
      const len = cs * 0.4 * val;
      const ex = cx0 + delta[1]*len, ey = cy0 + delta[0]*len;
      ctx.strokeStyle = `rgba(34,197,94,${0.4+val*0.6})`;
      ctx.lineWidth = Math.max(2, cs*0.12);
      ctx.beginPath(); ctx.moveTo(cx0, cy0); ctx.lineTo(ex, ey); ctx.stroke();
      // Arrowhead
      const angle = Math.atan2(delta[0], delta[1]);
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex - cs*0.15*Math.cos(angle-0.5), ey - cs*0.15*Math.sin(angle-0.5));
      ctx.lineTo(ex - cs*0.15*Math.cos(angle+0.5), ey - cs*0.15*Math.sin(angle+0.5));
      ctx.closePath(); ctx.fillStyle=ctx.strokeStyle; ctx.fill();
    }
  }

  // Current pos marker
  const cur = steps[cap].pos;
  const cxp = ox+(cur[1]+0.5)*cs, cyp = oy+(cur[0]+0.5)*cs;
  ctx.beginPath(); ctx.arc(cxp, cyp, cs*0.32, 0, Math.PI*2);
  ctx.fillStyle = steps[cap].dead ? '#dc2626' : '#f59f00';
  ctx.fill(); ctx.strokeStyle='#fff'; ctx.lineWidth=2; ctx.stroke();

  // S/G labels
  ctx.font=`bold ${Math.max(9,cs*0.4)}px sans-serif`; ctx.textAlign='center'; ctx.textBaseline='middle';
  ctx.fillStyle='#1971c2'; ctx.fillText('S', ox+(D.maze.start[1]+0.5)*cs, oy+(D.maze.start[0]+0.5)*cs);
  ctx.fillStyle='#c2255c'; ctx.fillText('G', ox+(D.maze.goal[1]+0.5)*cs, oy+(D.maze.goal[0]+0.5)*cs);
}

// ---------- Step detail ----------
function renderDetail(id, steps, idx, isExtended) {
  const el = $(id);
  if (!steps||!steps.length) { el.innerHTML='<div class="step-box"><h4>Status</h4><span>No data</span></div>'; return; }
  const cap = Math.min(idx, steps.length-1), s = steps[cap];
  const d9 = s.d9 || {};
  const d9str = Object.entries(d9).filter(([,v])=>v>0).map(([k,v])=>`${k}:${v.toFixed(3)}`).join(' ') || 'none';
  const boxes = [
    ['Step', `${s.step} / ${steps.length-1}`],
    ['Position', `(${s.pos[0]}, ${s.pos[1]})`],
    ['Action', s.action + (s.moved?'':' <span class="neg">blocked</span>')],
    ['Type', s.dead ? '<span class="neg">Dead-end</span>' : s.novel ? '<span class="pos">Novel</span>' : '<span class="neg">Revisit</span>'],
    ['Visits', s.visits],
    ['Best Hop', s.hop],
    ['Edges', s.edges],
    ['g_min', s.gmin.toFixed(6)],
  ];
  if (isExtended) boxes.push(['dim9 bias', d9str]);
  el.innerHTML = boxes.map(([t,v])=>`<div class="step-box"><h4>${t}</h4><span>${v}</span></div>`).join('');
}

// ---------- Event log ----------
function renderEvents(id, steps, idx) {
  const el = $(id);
  if (!steps||!steps.length) { el.innerHTML=''; return; }
  const cap = Math.min(idx, steps.length-1);
  let html = '';
  const start = Math.max(0, cap-20);
  for (let i=start; i<=cap; i++) {
    const s = steps[i];
    let cls = 'ev-novel', label = 'novel';
    if (s.dead) { cls='ev-dead'; label='DEAD-END'; }
    else if (!s.novel) { cls='ev-revisit'; label=`revisit(${s.visits})`; }
    html += `<div class="${cls}">step ${s.step}: (${s.pos[0]},${s.pos[1]}) ${s.action} → ${label}</div>`;
  }
  el.innerHTML = html;
  el.scrollTop = el.scrollHeight;
}

// ---------- Charts ----------
function drawChart(canvasId, baseSteps, extSteps, field, label, curStep) {
  const canvas = $(canvasId), ctx = canvas.getContext('2d');
  const W=canvas.width, H=canvas.height;
  ctx.clearRect(0,0,W,H);
  const pad={l:50,r:14,t:20,b:24};
  const pW=W-pad.l-pad.r, pH=H-pad.t-pad.b;
  const bD=baseSteps.map(s=>s[field]||0), eD=extSteps.map(s=>s[field]||0);
  const all=[...bD,...eD]; if(!all.length) return;
  const maxX=Math.max(bD.length,eD.length)||1;
  let minY=Math.min(...all), maxY=Math.max(...all);
  if(minY===maxY){minY-=1;maxY+=1;}
  const xs=pW/maxX, ys=pH/(maxY-minY);
  ctx.fillStyle='#fff'; ctx.fillRect(0,0,W,H);
  // Current step
  ctx.strokeStyle='#e2e8f0'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l+curStep*xs,pad.t); ctx.lineTo(pad.l+curStep*xs,pad.t+pH); ctx.stroke();
  function draw(data,color,mx){if(!data.length)return;ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.beginPath();
    const lim=Math.min(mx+1,data.length);for(let i=0;i<lim;i++){const x=pad.l+i*xs,y=pad.t+pH-(data[i]-minY)*ys;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}ctx.stroke();}
  draw(bD,'rgba(148,163,184,0.8)',curStep); draw(eD,'rgba(76,110,245,0.9)',curStep);
  ctx.strokeStyle='#94a3b8'; ctx.lineWidth=1; ctx.beginPath();
  ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+pH); ctx.lineTo(pad.l+pW,pad.t+pH); ctx.stroke();
  ctx.fillStyle='#475569'; ctx.font='11px sans-serif';
  ctx.textAlign='right'; ctx.fillText(maxY.toFixed(1),pad.l-4,pad.t+10); ctx.fillText(minY.toFixed(1),pad.l-4,pad.t+pH);
  ctx.textAlign='center'; ctx.fillText(label,W/2,14);
  ctx.textAlign='left';
  ctx.fillStyle='#94a3b8';ctx.fillRect(pad.l+8,pad.t+4,16,3);ctx.fillStyle='#475569';ctx.fillText('base',pad.l+28,pad.t+10);
  ctx.fillStyle='#4c6ef5';ctx.fillRect(pad.l+70,pad.t+4,16,3);ctx.fillStyle='#475569';ctx.fillText('ext',pad.l+90,pad.t+10);
}

// ---------- Render all ----------
function render() {
  const bS=getSteps('baseline'), eS=getSteps('extended');
  drawMaze(ctxBase, canvasBase, bS, stepIdx, false);
  drawMaze(ctxExt, canvasExt, eS, stepIdx, true);
  renderDetail('detail-base', bS, stepIdx, false);
  renderDetail('detail-ext', eS, stepIdx, true);
  renderEvents('events-base', bS, stepIdx);
  renderEvents('events-ext', eS, stepIdx);
  drawChart('chart-edges', bS, eS, 'edges', 'Edge Count', stepIdx);
  drawChart('chart-hop', bS, eS, 'hop', 'Best Hop (k*)', stepIdx);
  counter.textContent = `${stepIdx} / ${Math.max(maxSteps(),0)}`;
}

// ---------- Controls ----------
function updateSlider() { const m=Math.max(maxSteps(),0); slider.max=m; if(stepIdx>m)stepIdx=m; slider.value=stepIdx; }
phaseSel.addEventListener('change',()=>{ phase=phaseSel.value; stepIdx=0; stopPlay(); updateSlider(); render(); });
slider.addEventListener('input',()=>{ stepIdx=parseInt(slider.value); render(); });
playBtn.addEventListener('click',()=>{ playing?stopPlay():startPlay(); });
prevBtn.addEventListener('click',()=>{ if(stepIdx>0){stepIdx--;slider.value=stepIdx;render();} });
nextBtn.addEventListener('click',()=>{ if(stepIdx<maxSteps()){stepIdx++;slider.value=stepIdx;render();} });
overlayTgl.addEventListener('change',()=>{ render(); });
function startPlay(){playing=true;playBtn.textContent='⏸';playBtn.classList.add('paused');tick();}
function stopPlay(){playing=false;playBtn.textContent='▶';playBtn.classList.remove('paused');if(playTimer){clearTimeout(playTimer);playTimer=null;}}
function tick(){if(!playing)return;if(stepIdx<maxSteps()){stepIdx++;slider.value=stepIdx;render();playTimer=setTimeout(tick,parseInt(speedSel.value));}else stopPlay();}
document.addEventListener('keydown',e=>{
  if(e.key==='ArrowRight'||e.key==='l'){if(stepIdx<maxSteps()){stepIdx++;slider.value=stepIdx;render();}}
  else if(e.key==='ArrowLeft'||e.key==='h'){if(stepIdx>0){stepIdx--;slider.value=stepIdx;render();}}
  else if(e.key===' '){e.preventDefault();playing?stopPlay():startPlay();}
});

renderSummary(); updateSlider(); render();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--result-dir", default="results/graph_persistent_dg/v6_perseed")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = os.path.join(args.result_dir, f"maze_viz_seed{args.seed}.html")
    build_html(args.seed, args.result_dir, args.output)

if __name__ == "__main__":
    main()
