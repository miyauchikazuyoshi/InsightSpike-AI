# geDIG v0.1.0 — Public Preview

Date: YYYY‑MM‑DD

## Highlights
- English‑first README with academic tone and clear project intent
- GitHub Pages (docs/) landing page and reviewer call
- Paper (v4): PSZ/SLO definition; AG/DG 60‑seed aggregation for 25×25/s500; threats section
- Reproduction scripts: run 60‑seed batch → aggregate → TeX update
- Repo hygiene: Code of Conduct; large‑file audit + relocation plan

## Repro Entrypoints
- Maze batch (25×25, s500):
  - L3: `python scripts/run_maze_batch_and_update.py --mode l3 --seeds 60 --workers 4 --update-tex`
  - Eval: `python scripts/run_maze_batch_and_update.py --mode eval --seeds 60 --workers 4 --update-tex`
- Aggregates: `docs/paper/data/maze_25x25_*_s500.json`
- Paper build (JA): `make paper-build-ja`

## Assets (Release attachments)
- Option A (light): aggregated JSONs + rendered PDF/figures subset
- Option B (full): add a ZIP containing HTML viewers + a small subset of step logs/SQLite

## Known Limitations
- Eval 60‑seed still running (table uses seed=0 grid run for Eval column)
- Large artifacts not fully migrated to Releases/LFS yet
- Phase 2 is design‑only in this release

## Next
- Run Eval 60‑seed and update the Eval column
- Add SE/95% CI to tables (aggregation script extension)
- Migrate heavy artifacts to a `v0.1-assets` release bundle

