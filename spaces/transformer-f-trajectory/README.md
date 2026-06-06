---
title: geDIG F-Trajectory Demo
emoji: 📈
colorFrom: indigo
colorTo: orange
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
license: apache-2.0
---

# geDIG · F-Trajectory across Transformer layers

Live companion to the JSAI 2026 poster
**「動的知識グラフの探索・統合を制御する統一ゲージの提案」**.

Visualises how the unified gauge

```
F = ΔEPC − λ(ΔH + γ·ΔSP)
```

evolves layer-by-layer as a sentence flows through BERT. Same equation as the
maze experiment — different domain. The poster's central claim
(*one gauge, multiple domains*) you can see in 10 seconds.

## Quick start

```bash
cd spaces/transformer-f-trajectory
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (Optional) Pre-compute presets so the demo switches instantly.
python compute_presets.py --model bert-base-uncased --device mps    # ~30s on M-series Mac

streamlit run app.py
```

Then open <http://localhost:8501>.

## Architecture

```
spaces/transformer-f-trajectory/
├── app.py                  # Streamlit app (single page, 2 tabs)
├── compute_presets.py      # CLI: pre-compute trajectories for preset sentences
├── preset_sentences.json   # Curated demo sentences by category
├── presets.json            # ← generated; cached trajectories
├── requirements.txt
└── lib/
    └── f_trajectory.py     # Thin wrapper over the research code at
                            # experiments/transformer/inference_f_trajectory/
```

The actual F computation lives in
`experiments/transformer/inference_f_trajectory/gedig_hidden.py`
(the research code). This demo only adds a Streamlit UI and Plotly charts.

## What the demo shows

### Tab 1 — Presets (instant switch)
12 hand-picked sentences across six categories:

- **Simple** — short, unambiguous
- **Complex** — multi-clause, long
- **Ambiguous** — classic parser-confusing constructions
- **Garden-path** — "The horse raced past the barn fell."
- **Named entity** — proper-noun anchoring
- **Question** — interrogative form

For each, the user sees:

- **Cumulative F across layers** — the phase-transition signature: low F early
  (exploration), monotonic rise late (structuring).
- **Per-layer ΔEPC, ΔH, ΔSP** — what drives F at each step. Shows that
  F is a balance, not a sum.

### Tab 2 — Custom input (live BERT)
Type any sentence. BERT runs locally (1-2s on Apple Silicon `mps`,
3-5s on CPU). Same charts.

### Sidebar controls
- Model: `bert-base-uncased` (12 layers, default) or `distilbert-base-uncased` (6 layers).
- Device: cpu / mps / cuda.
- λ, γ sliders — affect custom input. Presets are fixed at λ=1.0, γ=0.5.

## How it ties to the poster

| Poster claim | What the demo shows |
| --- | --- |
| F is a *single scalar* combining structure + information | Σ F curve is one line you read in 2 sec |
| Phase transition: shallow = explore, deep = structure | Cumulative F bends upward in the deep layers |
| Same F across spatial KG (maze) and semantic KG (Transformer) | Same equation; only the underlying graph changes |
| ΔEPC and ΔH+γΔSP are independent dimensions | Component chart shows them moving independently |

## Notes for the poster session

- **Pre-compute presets the night before.** The Cached `presets.json` makes
  the switcher instant; do not rely on live BERT in front of an audience.
- **Disable wifi.** Demo is fully local once `presets.json` exists and the
  model has been pulled once. Avoids streaming hiccups.
- **Have a fallback laptop.** Streamlit + transformers should be stable on
  any recent Mac. Keep `presets.json` checked in as the last-resort static
  fallback.

## Deployment (optional)

The structure follows the HuggingFace Spaces convention so the demo can
be deployed as a public Space later. The frontmatter at the top of this
README is the Spaces config.

## License

Apache 2.0 (same as the parent repository).
