#!/usr/bin/env python3
"""
Build a simple HTML gallery for run folders containing run_summary.json.

Usage:
  python experiments/maze-navigation-enhanced/scripts/build_gallery.py [BASE_DIR]

If BASE_DIR is omitted, defaults to docs/images/gedegkaisetsu.
Outputs an index.html in the base directory, listing subfolders with quick stats and links.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def find_runs(base: Path) -> list[Path]:
    runs: list[Path] = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        if (p / 'run_summary.json').exists():
            runs.append(p)
    runs.sort(key=lambda q: q.name)
    return runs


def load_summary(p: Path) -> dict[str, Any] | None:
    f = p / 'run_summary.json'
    try:
        return json.loads(f.read_text(encoding='utf-8'))
    except Exception:
        return None


def coverage(maze: list[list[int]] | None, path_xy: list[list[int]] | None) -> float | None:
    if not maze or not path_xy:
        return None
    open_cells = sum(1 for row in maze for v in row if v == 0)
    uniq = len({tuple(p) for p in path_xy})
    return uniq / max(1, open_cells)


def build_html(rows: list[dict[str, Any]], base: Path) -> str:
    head = f"""<!DOCTYPE html><html lang=\"ja\"><meta charset=\"utf-8\"><title>GeDIG DFS Gallery</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:16px}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #e9ecef;padding:6px 8px;font-size:14px}}th{{background:#f8f9fa;text-align:left}}
a{{text-decoration:none;color:#0b7285}} .muted{{opacity:.7}}</style>
<h2>GeDIG DFS Gallery</h2>
<p class=\"muted\">Base: {base.as_posix()}</p>
<table><thead><tr><th>Folder</th><th>Size</th><th>Seed</th><th>Steps</th><th>Goal</th><th>Coverage</th><th>geDIG min</th><th>ΔSP&lt;0</th></tr></thead><tbody>
"""
    body = []
    for r in rows:
        link = (r['path'] / 'index.html').as_posix()
        cov = r.get('coverage')
        body.append(
            "<tr>"
            f"<td><a href='{link}'>{r['name']}</a></td>"
            f"<td>{r.get('size','')}</td>"
            f"<td>{r.get('seed','')}</td>"
            f"<td>{r.get('steps','')}</td>"
            f"<td>{'✓' if r.get('goal') else ''}</td>"
            f"<td>{('%.3f' % cov) if isinstance(cov, (int,float)) else ''}</td>"
            f"<td>{r.get('gedig_min','')}</td>"
            f"<td>{r.get('sp_neg_count','')}</td>"
            "</tr>"
        )
    tail = "</tbody></table>\n"
    return head + "\n".join(body) + tail


def main(argv: list[str]) -> None:
    base = Path(argv[1]).resolve() if len(argv) > 1 else Path('docs/images/gedegkaisetsu').resolve()
    base.mkdir(parents=True, exist_ok=True)
    runs = find_runs(base)
    rows: list[dict[str, Any]] = []
    for r in runs:
        js = load_summary(r)
        if not js:
            continue
        g = [v for v in (js.get('gedig_t') or []) if isinstance(v, (int, float))]
        sp = [v for v in (js.get('sp_delta_t') or []) if isinstance(v, (int, float))]
        rows.append({
            'name': r.name,
            'path': r,
            'size': js.get('size'),
            'seed': js.get('seed'),
            'steps': js.get('steps'),
            'goal': js.get('goal_reached'),
            'coverage': coverage(js.get('maze'), js.get('path')),
            'gedig_min': ("%.6f" % min(g)) if g else '',
            'sp_neg_count': sum(1 for x in sp if x < 0),
        })
    html = build_html(rows, base)
    out = base / 'index.html'
    out.write_text(html, encoding='utf-8')
    print(f"[Saved] {out}")


if __name__ == '__main__':
    main(sys.argv)
