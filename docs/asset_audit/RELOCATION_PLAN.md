# Releases / LFS Relocation Plan

This document lists patterns that are good candidates for Releases or Git LFS.

## Candidates

- Generated HTML step viewers: `experiments/**/results/**/*.html`
- SQLite snapshots: `experiments/**/results/**/*.sqlite`
- Large model/cache blobs under `experiments/**/cache/`
- High‑resolution figures under `docs/paper/figures/*.png` (if very large)
- Long JSON step logs: `*_steps.json` (keep aggregated CSV/JSON in repo)

## Suggested approach

1) Stop tracking new large artifacts
   - Ensure `.gitignore` covers common results and cache paths (already in place).
2) Move existing heavy artifacts to a GitHub Release (attach as assets) or Git LFS
   - Create a `v0.1-assets` Release for reproducibility bundles (HTML viewers, SQLite, raw step logs)
3) Link Releases from README and Docs
   - Add a “Reproducibility bundle” link where tables/figures are referenced
4) If the git history is large due to past commits
   - Evaluate BFG Repo‑Cleaner or `git filter-repo` to prune large files from history (with backups)
