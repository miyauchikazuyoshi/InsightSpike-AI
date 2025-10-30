# RAG Insight Vector Alignment (Supplemental Experiment)

This experiment computes alignment between an "insight vector" (constructed by similarity‑weighted multi‑hop message passing over KB embeddings) and the LLM answer embedding. It generates Δs = cos(z_ins, z_ans) − cos(z0, z_ans) histograms and summary stats.

- Inputs: a JSON with question/response pairs (e.g., final_optimal_prompt_results.json)
- KB: uses `data/insight_store/knowledge_base/initial/insight_dataset.txt` from the root repo by default
- Output: figures to `results/figures/` and JSON stats to `results/outputs/`

## Quick Start

1) Place your QA results JSON here (or symlink):

```
cp /path/to/final_optimal_prompt_results.json inputs/final_optimal_prompt_results.json
```

2) Run the alignment:

```
python3 run_alignment.py \
  --inputs inputs/final_optimal_prompt_results.json \
  --outfig results/figures/rag_insight_alignment.pdf
```

3) Rebuild the paper (optional):

```
cd docs/paper && latexmk -xelatex -interaction=nonstopmode geDIG_paper_restructured_draft_xe.tex
```

## Using Live LLM (optional)
If you want to regenerate answers via LLM, set API keys as env vars and use the provided stub (see `run_llm_generate.py`).

- OpenAI: `export OPENAI_API_KEY=...`
- Anthropic: `export ANTHROPIC_API_KEY=...`

This project intentionally does not hard‑code API keys. Use env vars or a local `.env` (never commit secrets).

## Notes
- The script relies on the repo embedder if available; otherwise falls back to sentence‑transformers.
- For robustness, consider running with `--residual_ans` to compare `ans - ques` against `z_ins`.

