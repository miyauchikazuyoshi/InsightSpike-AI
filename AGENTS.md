# AGENTS.md — Codex Cloud Guidance

Scope: Applies to the entire repository.

## Goal
Make this repo easy to test in constrained cloud agents (no GPU, restricted network, ephemeral filesystem). Prefer fast, deterministic checks over heavy runs.

## Quick Commands
- Install (pip): `python -m venv .venv && source .venv/bin/activate && pip install -e . && pip install -U pytest`
- Smoke tests (fast, no network): `bash scripts/codex_smoke.sh`
- Minimal pytest (skip heavy): `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 INSIGHTSPIKE_LITE_MODE=1 INSIGHTSPIKE_MIN_IMPORT=1 pytest -q -m "not slow and not integration and not performance and not benchmark and not calibration"`
- Make target: `make codex-smoke`

## Environment Defaults (cloud-safe)
- `INSIGHTSPIKE_LITE_MODE=1` and `INSIGHTSPIKE_MIN_IMPORT=1`
  - Forces lightweight code paths and fallback embedder (no torch/sentence-transformers).
  - When embedding backends are not installed, the system degrades to a simple CPU embedder and Numpy index; examples and smoke tests still run.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
  - Prevents unexpected slow/fragile third‑party pytest plugins.
- Optional when running CLI in cloud: set LLM to mock
  - `INSIGHTSPIKE_LLM__PROVIDER=mock` and `INSIGHTSPIKE_LLM__MODEL=mock`

## What to Run in Codex Cloud
1) Preferred: `make codex-smoke` (or `bash scripts/codex_smoke.sh`)
   - Executes a tiny, representative subset of tests: AB logger + SQLite + geDIG pure API.
   - No external downloads, no GPUs, finishes quickly.
2) Broader but still light: run pytest with marker filter shown above to exclude `slow`, `integration`, `performance`, `benchmark`, `calibration`.

## Do Not Do (cloud)
- Do not install optional heavy extras (e.g., torch/pyg) or run GPU code.
- Do not rely on live network LLMs. Use mock provider or examples that don’t call external APIs.
- Do not write outside the repository. Use `tmp/` or `results/`.

## IO & Paths
- Temporary outputs: `tmp/` and `results/` are allowed for writes.
- Avoid writing into top-level `data/` unless tests explicitly require `data/insight_store` (default datastore path). Keep changes local.

## CLIs and Examples
- Public minimal examples (no network):
  - `python examples/public_quick_start.py`
  - `python examples/hello_insight.py`
- CLI entrypoints:
  - `python -m insightspike.__main__`
  - `python -m insightspike.cli.spike --help`

## Notes for Agents
- Prefer the smoke script over full test matrix to keep runs deterministic.
- If you need a preset, you can use `cloud`, `development`, or `experiment`.
  - `cloud` preset now exists (mock LLM, no heavy features, safe logging).
- Large dirs to ignore for scanning/heavy work: `logs/`, `results/`, `node_modules/`, `experiments/`, `docs/paper/`.

## Troubleshooting
- Slow import or stalls: keep `INSIGHTSPIKE_LITE_MODE=1` and `INSIGHTSPIKE_MIN_IMPORT=1` set.
- PyYAML missing for config files: use JSON or avoid explicit config loading in smoke runs.
- If pytest plugins cause issues, ensure `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is set.
