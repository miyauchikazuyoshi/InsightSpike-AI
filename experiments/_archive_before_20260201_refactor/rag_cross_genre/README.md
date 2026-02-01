# Cross-Genre RAG Experiment

This experiment builds a cross-genre dataset where each query requires
combining knowledge from multiple domains (e.g., science + economics).
It reuses the exp2to4_lite RAG pipeline for evaluation.

## Quick Start (synthetic)

1) Generate a small dataset:

```bash
python experiments/rag_cross_genre/scripts/generate_dataset.py \
  --output experiments/rag_cross_genre/data/cross_genre_sample.jsonl \
  --num-queries 200 \
  --docs-per-query 6
```

2) Run the experiment:

```bash
python -m experiments.exp2to4_lite.src.run_experiment \
  --config experiments/rag_cross_genre/configs/exp23_cross_genre.yaml
```

Results will be written under `experiments/rag_cross_genre/results/`.

## Import Mode (question_answer backup)

If you have a knowledge base and a question set with `expected_tags`
(for example from the 0803 question_answer backup), you can convert it:

```bash
python experiments/rag_cross_genre/scripts/generate_dataset.py \
  --kb /path/to/knowledge_500.json \
  --question-set /path/to/questions_100.json \
  --output experiments/rag_cross_genre/data/cross_genre_import.jsonl \
  --distractors 3 \
  --min-support-domains 2 \
  --support-per-query 2 \
  --selection-mode token
```

The importer maps question tags to KB tags (see `TAG_ALIASES`) and enforces
cross-genre support by requiring at least two distinct domains.

For semantic selection, use the embedding mode (requires `sentence-transformers`):

```bash
.venv/bin/python experiments/rag_cross_genre/scripts/generate_dataset.py \
  --kb /path/to/knowledge_500.json \
  --question-set /path/to/questions_100.json \
  --output experiments/rag_cross_genre/data/cross_genre_import.jsonl \
  --distractors 3 \
  --min-support-domains 2 \
  --support-per-query 2 \
  --selection-mode embedding \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

Then point a config at the new dataset path.

## Public Dataset Mode (HotpotQA)

HotpotQA is a widely used multi-hop QA dataset. This script streams the
distractor split from Hugging Face and builds a cross-genre dataset by
classifying support docs into coarse domains.

```bash
.venv/bin/python experiments/rag_cross_genre/scripts/prepare_hotpotqa.py \
  --output experiments/rag_cross_genre/data/hotpotqa_cross_genre.jsonl \
  --split train \
  --max-questions 50000 \
  --min-domains 2 \
  --min-non-other 1 \
  --include-other \
  --fallback-embedding \
  --embedding-threshold 0.25 \
  --fallback-min-score 2 \
  --distractors 3
```

You can also pass a local HotpotQA JSON via `--input`.

## Notes

- The dataset is JSONL with fields: `query`, `ground_truth`, `documents`.
- `documents` are merged into a single corpus across all queries.
- The evaluation metric uses token overlap with `ground_truth`.
- Import mode adds `expected_tags`, `mapped_support_tags`, `mapped_support_domains`,
  plus `support_tag`/`support_domain`/`support_score` in document metadata for audit.
