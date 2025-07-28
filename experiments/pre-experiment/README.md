# Pre-Experiment: Vector Space Analysis and Integration Methods

This directory contains preliminary experiments that investigate fundamental properties of vector embeddings and integration strategies for the InsightSpike system.

## Overview

These experiments were conducted to:
1. Understand how questions and answers relate in vector space
2. Identify challenges with specific content types (e.g., arithmetic expressions)
3. Compare different vector integration strategies
4. Provide empirical justification for InsightSpike's design decisions

## Key Findings

### 1. Question-Answer Vector Space Relationships
- Questions and answers maintain ≈0.8 cosine similarity
- They occupy distinct but related regions in embedding space
- Direct similarity search from Q won't reliably find A

### 2. Arithmetic Expression Clustering
- Arithmetic expressions form abnormally dense clusters (avg similarity ≈0.945)
- This "hairball" effect poses challenges for mathematical reasoning
- Special handling needed for mathematical content

### 3. Weighted vs Uniform Integration
- Weighted integration consistently outperforms uniform integration
- Advantage most pronounced with high variance in item relevance
- Average improvement of +0.136 in extreme cases

## Directory Structure

```
pre-experiment/
├── src/
│   ├── test_vector_similarity.py      # Q-A space analysis
│   ├── test_uniform_weight_integration.py  # Integration comparison
│   ├── test_extreme_uniform_integration.py # Extreme case testing
│   ├── create_publication_figures.py   # Generate paper figures
│   └── figures/                        # Generated figures
├── results/
│   ├── figures/                        # Experimental visualizations
│   └── data/                          # Raw experimental data
├── PRELIMINARY_EXPERIMENTS_PAPER_SECTION.md  # Paper draft section
├── THEORETICAL_FRAMEWORK.md            # Mathematical foundation
├── EXPERIMENT_BRUSHUP_PLAN.md         # Plan for scaling to 100+ cases
└── README.md                          # This file
```

## Running the Experiments

### Prerequisites
```bash
pip install sentence-transformers numpy matplotlib seaborn scikit-learn
```

### Individual Experiments
```bash
# Question-Answer similarity analysis
python src/test_vector_similarity.py

# Weighted vs uniform integration
python src/test_uniform_weight_integration.py

# Extreme case testing
python src/test_extreme_uniform_integration.py

# Generate publication figures
python src/create_publication_figures.py
```

## Publication Materials

### Figures
- `qa_similarity_analysis.pdf/png` - Question-answer vector space relationships
- `integration_comparison.pdf/png` - Weighted vs uniform integration results
- `arithmetic_clustering.pdf/png` - Visualization of arithmetic expression clustering
- `summary_table.tex` - LaTeX table summarizing key findings

### Documentation
- `PRELIMINARY_EXPERIMENTS_PAPER_SECTION.md` - Ready-to-include paper section
- `THEORETICAL_FRAMEWORK.md` - Mathematical foundation and theory

## Next Steps

As outlined in `EXPERIMENT_BRUSHUP_PLAN.md`, the next phase involves:
1. Scaling to 100+ test cases for statistical rigor
2. Implementing automated evaluation pipeline
3. Testing across more diverse domains
4. Validating with different embedding models

## Citation

If you use these experiments in your research, please cite:
```
[Citation to be added upon publication]
```
EOF < /dev/null