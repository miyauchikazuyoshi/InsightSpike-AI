# Experiment Reorganization Summary

Date: 2025-07-03

## Overview

All result directories from `gedig_embedding_evaluation/code/` have been moved to properly structured experiment directories under `gedig_embedding_evaluation/`.

## Reorganized Experiments

### 1. Episodic Learning Experiment
- **Directory**: `episodic_learning_experiment/`
- **Python Script**: `gedig_episodic_learning.py`
- **Results**: 
  - `episodic_learning_analysis.png`
  - `episodic_learning_results.json`
  - `EPISODIC_LEARNING_SUMMARY.md`

### 2. Improved Embedding Experiment
- **Directory**: `improved_embedding_experiment/`
- **Python Script**: `gedig_embedding_improved.py`
- **Results**:
  - `improved_comparison.png`
  - `improved_results.json`
  - `performance_by_type.png`
  - `IMPROVED_RESULTS_SUMMARY.md`

### 3. InsightSpike Comparison Experiment
- **Directory**: `insightspike_comparison_experiment/`
- **Python Script**: `insightspike_vs_baselines_comparison.py`
- **Results**:
  - `comparison_results.json`
  - `insightspike_comparison.png`
  - `insightspike_detailed_analysis.png`
  - `COMPARISON_REPORT.md`

### 4. Intrinsic Threshold Experiment
- **Directory**: `intrinsic_threshold_experiment/`
- **Python Script**: `gedig_intrinsic_threshold_experiment.py`
- **Results**:
  - `intrinsic_threshold_analysis.png`
  - `intrinsic_threshold_results.json`
  - `THRESHOLD_ANALYSIS_SUMMARY.md`

### 5. Logical Operations Experiment
- **Directory**: `logical_operations_experiment/`
- **Python Script**: `gedig_logical_and_experiment.py`
- **Results**:
  - `logical_operations_analysis.png`
  - `logical_operations_results.json`
  - `LOGICAL_OPERATIONS_SUMMARY.md`

### 6. Real Graph Embedding Experiment
- **Directory**: `real_graph_embedding_experiment/`
- **Python Script**: `insightspike_with_real_graph_embedding.py`
- **Results**:
  - `real_graph_embedding_results.png`
  - `real_graph_results.json`
  - `GRAPH_EMBEDDING_REPORT.md`

### 7. RAG Comparison Experiment (Updated)
- **Directory**: `rag_comparison_experiment/` (existing)
- **Python Script**: `final_rag_comparison.py` (already present)
- **Results** (merged):
  - `comparison_summary.md`
  - `comprehensive_comparison_report.json`
  - `comprehensive_rag_comparison.png`

## Directory Structure

Each experiment now follows a consistent structure:
```
experiment_name/
├── code/
│   └── main_script.py
└── results/
    ├── result_files.json
    ├── visualizations.png
    └── summary.md
```

## Actions Taken

1. Created new experiment directories with proper `code/` and `results/` subdirectories
2. Moved all result files from `code/results_*/` to appropriate `experiment_name/results/`
3. Copied associated Python scripts to `experiment_name/code/`
4. Removed empty result directories from the `code/` folder
5. Merged final RAG comparison results into the existing `rag_comparison_experiment/`

All experiments are now properly organized with clear separation between code and results.