# InsightSpike-AI Makefile
# =======================

.PHONY: install install-dev test clean setup-models

# Install the package
install:
	pip install -e .
	@echo "Skipping model setup (no scripts/setup_models.py)"

# Install with development dependencies
install-dev:
	pip install -e ".[dev]"
	@echo "Skipping model setup (no scripts/setup_models.py)"

# Setup models only
setup-models:
	@echo "No model setup script present. Nothing to do."

# Run tests
test:
	pytest tests/

# Ultra-lightweight A/B logger selftest (no heavy deps)
selftest-ab:
	INSIGHTSPIKE_MIN_IMPORT=1 python scripts/selftest_ab_logger.py

# Fast targeted pytest without external plugins, minimal import
pytest-fast:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 INSIGHTSPIKE_MIN_IMPORT=1 INSIGHTSPIKE_LITE_MODE=1 pytest -k gedig_ab_logger -q

# SQLite-focused unit tests (lightweight, no external deps)
pytest-sqlite:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 INSIGHTSPIKE_MIN_IMPORT=1 INSIGHTSPIKE_LITE_MODE=1 PYTHONPATH=src \
		pytest -k "datastore_sqlite" -q

# geDIG Pure API test (lightweight)
pytest-gedig-pure:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 INSIGHTSPIKE_MIN_IMPORT=1 INSIGHTSPIKE_LITE_MODE=1 PYTHONPATH=src \
		pytest -k "gedig_pure_api" -q

# Codex Cloud smoke: tiny, deterministic subset
.PHONY: codex-smoke
codex-smoke:
	bash scripts/codex_smoke.sh

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Clean LaTeX artifacts (root and docs/paper)
.PHONY: clean-latex
clean-latex:
	find . -maxdepth 1 -type f \( -name "*.aux" -o -name "*.fls" -o -name "*.fdb_latexmk" -o -name "*.xdv" -o -name "*.out" \) -delete
	find docs/paper -type f \( -name "*.aux" -o -name "*.fls" -o -name "*.fdb_latexmk" -o -name "*.xdv" -o -name "*.out" \) -delete || true

# Tidy top-level stray paper artifacts by moving them under docs/paper
.PHONY: tidy-top
tidy-top:
	@mkdir -p docs/paper
	@for f in geDIG_paper_ja.aux geDIG_paper_ja.fdb_latexmk geDIG_paper_ja.fls geDIG_paper_ja.log; do \
	  if [ -f "$$f" ]; then echo "Moving $$f -> docs/paper/"; mv -f "$$f" docs/paper/; fi; \
	done

# Quick setup for new users
quickstart: install setup-models
	@echo "✅ InsightSpike-AI is ready to use!"
	@echo "Try running: python examples/basic_usage.py"

# ---------------------------------------------------------------------
# Dockerized maze experiments (Linux/OpenBLAS for stability)
# ---------------------------------------------------------------------
.PHONY: maze-docker-build maze-run-smoke maze-run-full

maze-docker-build:
	docker build -f docker/maze/Dockerfile -t insightspike-maze .

# Mount the experiments directory so results persist to host
maze-run-smoke:
	docker run --rm \
	  -v $(PWD)/experiments/maze-navigation-enhanced:/app/experiments/maze-navigation-enhanced \
	  -w /app \
	  insightspike-maze \
	  bash -lc "PYTHONPATH=experiments/maze-navigation-enhanced/src \
	    python experiments/maze-navigation-enhanced/src/analysis/clean_maze_run.py --sizes 11 --seeds 2"

maze-run-full:
	docker run --rm \
	  -v $(PWD)/experiments/maze-navigation-enhanced:/app/experiments/maze-navigation-enhanced \
	  -w /app \
	  insightspike-maze \
	  bash -lc "PYTHONPATH=experiments/maze-navigation-enhanced/src \
	    python experiments/maze-navigation-enhanced/src/analysis/clean_maze_run.py --sizes 15 25 --seeds 10"

# Parameterized single-batch runner
# Usage examples:
#  make maze-run SIZE=15 SEEDS=6 OFFSET=0 FAST=1
#  make maze-run SIZE=25 SEEDS=6 OFFSET=6 FAST=1
.PHONY: maze-run
maze-run:
	docker run --rm \
	  -e MAZE_FAST_MODE=$(FAST) \
	  -e MAZE_MAX_STEPS_FACTOR=$(FACTOR) \
	  -e MAZE_WIRING_WINDOW=$(WINDOW) \
	  -e MAZE_SPATIAL_GATE=$(DIST) \
	  -e MAZE_EARLY_ACCEPT_MARGIN=$(MARGIN) \
	  -v $(PWD)/experiments/maze-navigation-enhanced:/app/experiments/maze-navigation-enhanced \
	  -w /app \
	  insightspike-maze \
	  bash -lc "PYTHONPATH=experiments/maze-navigation-enhanced/src \
	    python experiments/maze-navigation-enhanced/src/analysis/clean_maze_run.py --sizes $(SIZE) --seeds $(SEEDS) --seed-offset $(OFFSET) --checkpoint"

.PHONY: maze-viz
maze-viz:
	docker run --rm \
	  -e MAZE_FAST_MODE=$(FAST) \
	  -v $(PWD)/experiments/maze-navigation-enhanced:/app/experiments/maze-navigation-enhanced \
	  -w /app \
	  insightspike-maze \
	  bash -lc "PYTHONPATH=experiments/maze-navigation-enhanced/src \
	    python experiments/maze-navigation-enhanced/src/analysis/visualize_maze_run.py --size $(SIZE) --seed $(SEED) --strategy $(STRAT) --fast"

# ------------------------------------------------------------
# Local preset + calibration + stats (no Docker)
# ------------------------------------------------------------
.PHONY: maze-preset maze-calibrate maze-stats maze-suite

# Apply a preset by exporting ENV via the loader (effective for this shell only)
maze-preset:
	PYTHONPATH=experiments/maze-navigation-enhanced/src \
	python -c "from utils.preset_loader import load_preset, apply_env; cfg = load_preset(preset_name='$(PRESET)') if '$(PRESET)' else load_preset(preset_name='25x25'); apply_env(cfg); print('Applied preset:', (cfg.get('preset') or 'default'), 'size=', (cfg.get('maze') or {}).get('size'))"

# Grid calibration for (k, tau, tau_bt)
maze-calibrate:
	PYTHONPATH=experiments/maze-navigation-enhanced/src \
	python experiments/maze-navigation-enhanced/src/analysis/calibrate_ktau.py \
	  --size $(or $(SIZE),25) --seeds $(or $(SEEDS),16) \
	  --k-grid $(or $(KGRID),0.08 0.10 0.12 0.15) \
	  --tau-grid $(or $(TAUGRID),-0.22 -0.18 -0.15 -0.12) \
	  --tau-bt-grid $(or $(BTGRID),-0.30 -0.25 -0.22 -0.18) \
	  --max-steps-factor $(or $(FACTOR),4.0)

# Statistical summary printout
maze-stats:
	PYTHONPATH=experiments/maze-navigation-enhanced/src \
	python experiments/maze-navigation-enhanced/src/analysis/stats_summary.py \
	  --size $(or $(SIZE),25) --seeds $(or $(SEEDS),16) --max-steps-factor $(or $(FACTOR),4.0)

# One-shot: preset -> calibrate -> stats
maze-suite: maze-preset maze-calibrate maze-stats

# ------------------------------------------------------------
# One-click reproduction helpers
# ------------------------------------------------------------
.PHONY: reproduce-maze reproduce-rag

reproduce-maze:
	@echo "[maze] preset→calibration→stats (25x25, seeds=32)"
	$(MAKE) maze-suite PRESET=25x25 SIZE=25 SEEDS=32 FACTOR=4.0

reproduce-rag:
	@echo "[rag] lightweight k, tau calibration on subset"
	PYTHONPATH=experiments/rag-dynamic-db-v3/src \
	python experiments/rag-dynamic-db-v3/src/calibrate_k_tau.py --subset 100 || true

.PHONY: rag-figs
rag-figs:
	@echo "[rag] Generating paper figures (performance & PSZ)"
	PYTHONPATH=experiments/rag-dynamic-db-v3/src MPLCONFIGDIR=results/mpl \
	python experiments/rag-dynamic-db-v3/src/plot_paper_figs.py

# ------------------------------------------------------------
# Maze tuning (15x15)
# ------------------------------------------------------------
.PHONY: maze-tune-15
maze-tune-15:
	PYTHONPATH=experiments/maze-navigation-enhanced/src \
	python experiments/maze-navigation-enhanced/src/analysis/tune_15x15.py --seeds $(or $(SEEDS),12) --fast $(FAST) --grid-small

.PHONY: maze-unified-15
maze-unified-15:
	PYTHONPATH=experiments/maze-navigation-enhanced/src \
	python experiments/maze-navigation-enhanced/src/analysis/unified_runner.py --preset 15x15 --seeds $(or $(SEEDS),20) --fast $(FAST) --compare-simple
