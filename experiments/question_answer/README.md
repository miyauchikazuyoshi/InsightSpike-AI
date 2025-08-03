# Question-Answer Experiment (Minimal Solution Selection)

## Overview

This experiment compares geDIG's minimal knowledge selection capabilities with direct LLM answers. Using 500 knowledge entries and 100 difficulty-graded questions, we demonstrate how graph-based optimization can efficiently select the most relevant knowledge for answering questions.

## Experiment Objectives

1. **Primary Objective**: Compare geDIG minimal knowledge selection vs LLM direct answers
2. **Secondary Objectives**: 
   - Demonstrate efficiency of graph-based knowledge selection
   - Evaluate precision/recall of knowledge selection
   - Show practical application for RAG systems

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
   - Precision/Recall/F1 of knowledge selection
   - Number of selected knowledge items
   - Processing time
   - Answer quality comparison

2. **Qualitative Metrics**:
   - Relevance of selected knowledge
   - Efficiency vs completeness trade-off
   - Unexpected but valuable knowledge combinations

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
│   ├── run_minimal_solution_experiment.py  # InsightSpike integration
│   ├── minimal_solution_experiment.py      # Demo implementation
│   ├── test_minimal_solution.py           # Test script
│   └── visualize_results.py               # Result visualization
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

### 1. Quick Test
```bash
# Test with minimal dataset
poetry run python src/test_minimal_solution.py
```

### 2. Full Experiment
```bash
# Run full experiment with all questions
poetry run python src/run_minimal_solution_experiment.py \
    --config experiment_config_minimal.yaml
```

### 3. Custom Dataset
```bash
# Run with custom knowledge and questions
poetry run python src/run_minimal_solution_experiment.py \
    --knowledge data/input/knowledge_base/custom_knowledge.json \
    --questions data/input/questions/custom_questions.json
```

### 4. Easy Script
```bash
# Use the convenience script
./run_experiment.sh
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

1. **Knowledge Selection Efficiency**:
   - Easy: 1-2 knowledge items selected
   - Medium: 2-3 knowledge items selected
   - Hard: 3-4 knowledge items selected
   - Very Hard: 4-5 knowledge items selected

2. **Performance Comparison**:
   - geDIG+LLM outperforms LLM-only on Medium+ questions
   - F1 score > 0.7 for knowledge selection
   - Average selection time < 1 second

3. **Practical Applications**:
   - Efficient RAG system knowledge selection
   - Reduced context window usage
   - Better answer quality with minimal knowledge

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

*Last updated: 2025-08-03*
*Redesigned from spike detection to minimal solution selection*