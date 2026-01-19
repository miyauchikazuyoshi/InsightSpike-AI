# InsightSpike-AI Makefile
# =======================

.PHONY: install install-dev test clean setup-models

# Install the package
install:
	pip install -e .
	@echo "Skipping model setup (no scripts/setup_models.py)"

# Install with development dependencies
install-dev:
	# Unified to pure pip for portability (Poetry groups not used here)
	pip install -e .
	python -m pip install -U \
		pytest pytest-cov pytest-asyncio \
		black isort flake8 mypy \
		types-pyyaml types-requests pillow
	@echo "Dev dependencies installed via pip."

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

# Move known stray artifacts under results/ for tidiness (non-destructive)
.PHONY: tidy-strays
tidy-strays:
	@mkdir -p results/strays
	@if [ -f None ]; then echo "Moving top-level 'None' -> results/strays/None.json"; mv -f None results/strays/None.json; fi

# Quick setup for new users
quickstart: install setup-models
	@echo "✅ InsightSpike-AI is ready to use!"
	@echo "Try running: python examples/public_quick_start.py"

# Coverage run (lightweight env flags)
.PHONY: coverage
coverage:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 INSIGHTSPIKE_MIN_IMPORT=1 INSIGHTSPIKE_LITE_MODE=1 \
		pytest -q --cov=insightspike --cov-report=term-missing tests/

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
# Local (no Docker) quick maze run
# ------------------------------------------------------------
# Usage examples:
#  make maze-run-local SIZE=25 STEPS=300 SEED=0 OUT=results/maze-local
#  make maze-run-local SIZE=25 STEPS=500 SEED=17 FAST=1 OUT=results/maze-local
.PHONY: maze-run-local
maze-run-local:
	@mkdir -p $(or $(OUT),results/maze-local)
	python experiments/maze-query-hub-prototype/run_experiment_query.py \
	  --preset paper --maze-size $(or $(SIZE),25) --max-steps $(or $(STEPS),300) \
	  --output $(or $(OUT),results/maze-local)/maze_s$(or $(SIZE),25)_t$(or $(STEPS),300)_seed$(or $(SEED),0)_summary.json \
	  --step-log $(or $(OUT),results/maze-local)/maze_s$(or $(SIZE),25)_t$(or $(STEPS),300)_seed$(or $(SEED),0)_steps.json \
	  --seed-start $(or $(SEED),0) --seeds 1
	@echo "Local run complete → $(or $(OUT),results/maze-local)"

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
# Default LLM provider for reproducibility (override with LLM_PROVIDER=openai)
LLM_PROVIDER ?= mock

.PHONY: reproduce-maze reproduce-rag reproduce-maze15 reproduce-hotpotqa reproduce-analogy

reproduce-maze:
	@echo "[maze] preset→calibration→stats (25x25, seeds=32)"
	$(MAKE) maze-suite PRESET=25x25 SIZE=25 SEEDS=32 FACTOR=4.0

reproduce-rag:
	@echo "[rag] lightweight k, tau calibration on subset"
	PYTHONPATH=experiments/rag-dynamic-db-v3/src \
	python experiments/rag-dynamic-db-v3/src/calibrate_k_tau.py --subset 100 || true

reproduce-maze15:
	@mkdir -p docs/paper/data
	@echo "[maze15] L3-only paper preset (15x15, steps=250, seeds=12)"
	INSIGHTSPIKE_PRESET=paper PYTHONPATH=src \
	python experiments/maze-query-hub-prototype/tools/run_paper_l3_only.py \
	  --maze-size 15 --max-steps 250 --seeds 12 \
	  --out-root experiments/maze-query-hub-prototype/results/l3_only \
	  --namespace l3only_15x15_s250 \
	  --maze-snapshot-out docs/paper/data/maze_15x15.json
	python experiments/maze-query-hub-prototype/tools/export_paper_maze.py \
	  --summary experiments/maze-query-hub-prototype/results/l3_only/_15x15_s250_l3only_summary.json \
	  --steps experiments/maze-query-hub-prototype/results/l3_only/_15x15_s250_l3only_steps.json \
	  --out-json docs/paper/data/maze_15x15_l3_s250.json \
	  --out-csv docs/paper/data/maze_15x15_l3_s250.csv \
	  --compression-base mem

reproduce-hotpotqa:
	@if [ ! -f experiments/hotpotqa-benchmark/data/hotpotqa_sample_100.jsonl ]; then \
	  echo "[hotpotqa] Missing data: experiments/hotpotqa-benchmark/data/hotpotqa_sample_100.jsonl"; \
	  echo "[hotpotqa] Run: python experiments/hotpotqa-benchmark/scripts/download_data.py"; \
	  exit 1; \
	fi
	@if [ "$(LLM_PROVIDER)" = "openai" ] && [ -z "$$OPENAI_API_KEY" ]; then \
	  echo "[hotpotqa] OPENAI_API_KEY is not set (LLM_PROVIDER=openai)"; \
	  exit 1; \
	fi
	@mkdir -p docs/paper/data
	@echo "[hotpotqa] geDIG sample run (provider=$(LLM_PROVIDER))"
	INSIGHTSPIKE_LLM_PROVIDER=$(LLM_PROVIDER) LLM_PROVIDER=$(LLM_PROVIDER) \
	python experiments/hotpotqa-benchmark/scripts/run_gedig.py \
	  --data experiments/hotpotqa-benchmark/data/hotpotqa_sample_100.jsonl \
	  --output experiments/hotpotqa-benchmark/results \
	  --limit 100
	@LATEST=$$(ls -1t experiments/hotpotqa-benchmark/results/gedig_*_summary.json | head -n1); \
	  cp $$LATEST docs/paper/data/hotpotqa_sample_summary.json; \
	  echo "[hotpotqa] Summary -> docs/paper/data/hotpotqa_sample_summary.json"

reproduce-analogy:
	@mkdir -p docs/paper/data
	@echo "[analogy] Cross-domain QA (SS enabled/disabled)"
	INSIGHTSPIKE_LITE_MODE=1 python -m experiments.structural_similarity.cross_domain_qa.qa_evaluation
	@cp experiments/structural_similarity/cross_domain_qa/results/comparison_results.json \
	  docs/paper/data/analogy_comparison.json

.PHONY: rag-figs
rag-figs:
	@echo "[rag] Generating paper figures (performance & PSZ)"
	PYTHONPATH=experiments/rag-dynamic-db-v3/src MPLCONFIGDIR=results/mpl \
	python experiments/rag-dynamic-db-v3/src/plot_paper_figs.py

# ------------------------------------------------------------
# Exp II–IV (lite) one-click helpers
# ------------------------------------------------------------
.PHONY: exp23-paper exp23-psz-sweep exp23-export-figs exp23-export-tables exp23-all paper-en paper-ja

# Calibrate gates on val and run test; real latency enabled.
exp23-paper:
	INSIGHTSPIKE_LITE_MODE=1 INSIGHTSPIKE_MIN_IMPORT=1 EXP_LITE_REAL_LATENCY=1 \
	python -m experiments.exp2to4_lite.src.run_suite \
	  --config experiments/exp2to4_lite/configs/exp23_paper.yaml --calibrate
	@# Generate figures for the latest paper result
	@LATEST=$$(ls -1t experiments/exp2to4_lite/results/exp23_paper_*.json | grep -v _alignment.json | head -n1); \
	python -c "from experiments.exp2to4_lite.src.viz import plot_psz_curves, plot_gating_timeseries; from pathlib import Path; import sys; res=Path(sys.argv[1]); out=Path('docs/paper/figures'); plot_psz_curves(res,out); plot_gating_timeseries(res,out)" $$LATEST
	@# Alignment JSON + table
	@LATEST=$$(ls -1t experiments/exp2to4_lite/results/exp23_paper_*.json | grep -v _alignment.json | head -n1); \
	python -m experiments.exp2to4_lite.src.alignment --results $$LATEST || true
	python -m experiments.exp2to4_lite.src.export_alignment_tex || true
	python -m experiments.exp2to4_lite.src.export_resources_tex || true
	python -m experiments.exp2to4_lite.src.export_tables_tex || true

# Sweep parameters toward PSZ band (50 queries, percentile acceptance example)
exp23-psz-sweep:
	INSIGHTSPIKE_LITE_MODE=1 INSIGHTSPIKE_MIN_IMPORT=1 EXP_LITE_REAL_LATENCY=1 \
	python -m experiments.exp2to4_lite.src.psz_sweep \
	  --config experiments/exp2to4_lite/configs/exp23_psz_target.yaml \
	  --accept-mode percentile --accept-percentile 0.02

# Export figures for latest paper + latest PSZ-target results
exp23-export-figs:
	@PAPER=$$(ls -1t experiments/exp2to4_lite/results/exp23_paper_*.json | grep -v _alignment.json | head -n1); \
	  PSZ=$$(ls -1t experiments/exp2to4_lite/results/exp23_psz_target_*.json | head -n1); \
	python -c "from experiments.exp2to4_lite.src.viz import plot_psz_curves, plot_gating_timeseries; from pathlib import Path; import sys; out=Path('docs/paper/figures'); [plot_psz_curves(Path(p),out) or plot_gating_timeseries(Path(p),out) for p in sys.argv[1:] if Path(p).exists()]" $$PAPER $$PSZ

# Export LaTeX tables (summary, ablation, resources, alignment)
exp23-export-tables:
	python -m experiments.exp2to4_lite.src.export_tables_tex || true
	python -m experiments.exp2to4_lite.src.export_resources_tex || true
	python -m experiments.exp2to4_lite.src.export_alignment_tex || true

# Build PDFs for both EN/JA v4 drafts
paper-en:
	cd docs/paper && xelatex -interaction=nonstopmode geDIG_onegauge_improved_v4_en.tex >/dev/null || true && \
	  xelatex -interaction=nonstopmode geDIG_onegauge_improved_v4_en.tex >/dev/null || true

paper-ja:
	cd docs/paper && xelatex -interaction=nonstopmode geDIG_onegauge_improved_v4.tex >/dev/null || true && \
	  xelatex -interaction=nonstopmode geDIG_onegauge_improved_v4.tex >/dev/null || true

# Latexmk variants (with BibTeX)
.PHONY: paper-en-bib paper-ja-bib
paper-en-bib:
	cd docs/paper && latexmk -xelatex -bibtex -file-line-error -interaction=nonstopmode geDIG_onegauge_improved_v4_en.tex && latexmk -c

paper-ja-bib:
	cd docs/paper && latexmk -xelatex -bibtex -file-line-error -interaction=nonstopmode geDIG_onegauge_improved_v4.tex && latexmk -c

# End-to-end: run paper, sweep PSZ, export assets, and compile PDFs
exp23-all: exp23-paper exp23-psz-sweep exp23-export-figs exp23-export-tables paper-en paper-ja

# ------------------------------------------------------------
# Paper helpers: maze batch aggregation (25x25 / s500)
# ------------------------------------------------------------
.PHONY: paper-maze-agg-l3 paper-maze-agg-eval paper-build-ja

paper-maze-agg-l3:
	python scripts/aggregate_maze_batch.py \
	  --dir experiments/maze-query-hub-prototype/results/batch_25x25 \
	  --prefix paper25_25x25_s500_seed \
	  --out docs/paper/data/maze_25x25_l3_s500.json

paper-maze-agg-eval:
	python scripts/aggregate_maze_batch.py \
	  --dir experiments/maze-query-hub-prototype/results/batch_25x25 \
	  --prefix paper25_25x25_s500_eval_seed \
	  --out docs/paper/data/maze_25x25_eval_s500.json

paper-build-ja:
	cd docs/paper && xelatex -interaction=nonstopmode geDIG_onegauge_improved_v4.tex >/dev/null || true && \
	  xelatex -interaction=nonstopmode geDIG_onegauge_improved_v4.tex >/dev/null || true

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
