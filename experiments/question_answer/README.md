# Question-Answer Experiment

## Overview

This experiment validates InsightSpike-AI's insight generation capabilities through a question-answering task designed based on research papers. Using 500 knowledge entries and 100 difficulty-graded questions, we analyze in detail how the system combines knowledge to generate new insights.

## Experiment Objectives

1. **Primary Objective**: Evaluate InsightSpike's insight generation capability in question answering
2. **Secondary Objectives**: 
   - Verify query storage functionality
   - Analyze the relationship between knowledge graph growth and insights
   - Investigate correlation between spike occurrence rate and answer quality

## Experiment Design

### Dataset

1. **Knowledge Base (500 entries)**:
   - Science & Technology (100 entries, 20%)
   - Mathematics & Logic (75 entries, 15%)
   - History & Culture (75 entries, 15%)
   - Daily Life (75 entries, 15%)
   - Arts & Literature (75 entries, 15%)
   - Philosophy & Psychology (50 entries, 10%)
   - Economics & Business (50 entries, 10%)

2. **Question Set (100 questions)**:
   - Easy (30 questions): Direct single knowledge reference
   - Medium (40 questions): Integration of 2-3 knowledge items
   - Hard (20 questions): Creative integration of multiple knowledge
   - Very Hard (10 questions): Revolutionary combination of cross-domain knowledge

### Evaluation Metrics

1. **Quantitative Metrics**:
   - Answer accuracy
   - Spike occurrence rate
   - Processing time
   - Memory usage

2. **Qualitative Metrics**:
   - Novelty of insights
   - Logical consistency of answers
   - Degree of knowledge integration

### Experimental Conditions

- **LLM Provider**: Claude 3.5 Sonnet (anthropic)
- **Knowledge Entries**: 500 (with genre tags)
- **Questions**: 100 (difficulty-graded)
- **Execution Method**: 
  - Knowledge addition: `add_knowledge` command
  - Question processing: Using `adaptive_loop`

## Directory Structure

```
question_answer/
├── src/
│   ├── run_experiment.py      # Main experiment script
│   ├── data_preparation.py    # Data preparation
│   ├── evaluation.py          # Evaluation metrics
│   └── analysis.py            # Result analysis
├── data/
│   ├── input/
│   │   ├── knowledge_base/    # Knowledge base files
│   │   └── questions/         # Question sets
│   └── processed/
│       ├── embeddings/        # Pre-computed embeddings
│       └── preprocessed/      # Preprocessed data
├── results/
│   ├── metrics/
│   │   ├── accuracy.json      # Accuracy metrics
│   │   ├── spike_analysis.json # Spike analysis
│   │   └── performance.json   # Performance metrics
│   ├── outputs/
│   │   ├── responses/         # Generated answers
│   │   └── insights/          # Detected insights
│   └── visualizations/
│       ├── spike_timeline.png # Spike occurrence timeline
│       ├── accuracy_heatmap.png # Accuracy heatmap
│       └── knowledge_graph.png # Knowledge graph visualization
├── data_snapshots/            # Experiment data snapshots
└── README.md                  # This document
```

## Experiment Procedure

### 1. Data Preparation
```bash
# Prepare knowledge base (500 entries) and questions (100)
poetry run python src/data_preparation.py --prepare-all
```

### 2. Knowledge Base Construction
```bash
# Add 500 knowledge entries using add_knowledge
poetry run python src/run_experiment.py --phase knowledge-loading
```

### 3. Question-Answer Experiment
```bash
# Process 100 questions using adaptive_loop
poetry run python src/run_experiment.py --phase question-answering
```

### 4. Result Analysis and Visualization
```bash
# Data analysis and visualization
poetry run python src/analysis.py --generate-all
```

## Data Collection Format

### Question-Answer Data
- **JSON Format**: Complete detailed data (including vector values)
- **CSV Format**: Aggregated data for analysis

### Collection Items
- Question ID, Question text
- Insight determination (has_insight)
- Branching determination (has_branching)
- Insight episode vector (384 dimensions)
- Branching episode vector (384 dimensions)
- LLM prompt
- LLM response
- LLM response vector (384 dimensions)
- Similarity to insight episode
- Processing time

## Pre-Experiment Checklist

- [ ] Verify that answers are not directly contained in the data
- [ ] Confirm baseline is properly configured
- [ ] Verify evaluation metrics are clearly defined
- [ ] Confirm seed is fixed for reproducibility
- [ ] Check if knowledge transfer from existing snapshots is needed

## Expected Outcomes

1. **Insight Patterns by Difficulty**:
   - Easy: Insight rate < 10%
   - Medium: Insight rate 20-40%
   - Hard: Insight rate 50-70%
   - Very Hard: Insight rate > 70%

2. **Cross-Genre Knowledge Combination**:
   - Identify genre combinations most likely to generate insights
   - Knowledge graph growth patterns

3. **Practical Insights**:
   - Characteristics of questions optimal for insight generation
   - Trade-off between processing time and insight quality

## Visualization Plan

- Insight occurrence rate heatmap (difficulty × genre)
- Processing time distribution (box plot by difficulty)
- Knowledge graph growth process
- t-SNE visualization of insight episodes
- Knowledge flow via Sankey diagram

## Important Notes

- Project root `data/` is read-only
- Experiment data must be backed up to `data_snapshots/`
- Use FileSystemDataStore (memory store prohibited)

## Related Documentation

- [CLAUDE.md](/CLAUDE.md) - Experiment execution guidelines
- [Query Storage Documentation](/docs/features/query_storage.md) - Query storage feature
- [MainAgent Behavior](/docs/architecture/mainagent_behavior.md) - Agent behavior

---

*Last updated: 2025-07-29*