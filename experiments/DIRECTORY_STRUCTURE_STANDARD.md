# Experiment Directory Structure Standard

This document defines the standard directory structure for all experiments in InsightSpike-AI.

## 📁 Standard Structure

Every experiment MUST follow this structure:

```
experiment_name/
├── code/           # All implementation scripts
├── data/           # Input data or data generation scripts
├── results/        # All output files (JSON, PNG, CSV, etc.)
└── README.md       # Experiment documentation
```

## ✅ Good Examples

```
gedig_embedding_evaluation/
├── rag_comparison_experiment/
│   ├── code/
│   │   └── final_rag_comparison.py
│   ├── data_backup/
│   │   ├── episodes.json
│   │   └── index.faiss
│   ├── results/
│   │   ├── comprehensive_comparison_report.json
│   │   └── comprehensive_rag_comparison.png
│   └── README.md
```

## ❌ Bad Examples

**DON'T put results in code directory:**
```
experiment/
├── code/
│   ├── script.py
│   └── results_something/    # ❌ WRONG
│       └── output.json
```

**DON'T use non-standard result directory names:**
```
experiment/
├── results_improved/         # ❌ Should be results/improved/
├── comparison_results/       # ❌ Should be results/comparison/
└── results_correct_ged/      # ❌ Should be results/correct_ged/
```

## 📋 Rules

1. **All results go in `results/` directory**
   - Never in `code/`
   - Never in top-level repository directory
   - Use subdirectories within `results/` for organization

2. **Code and data are separate from results**
   - `code/`: Implementation only
   - `data/`: Input data only
   - `results/`: Output only

3. **Use descriptive subdirectories within standard directories**
   ```
   results/
   ├── baseline/
   ├── improved/
   └── final/
   ```

4. **Every experiment needs documentation**
   - README.md is mandatory
   - Explain purpose, methodology, and key findings
   - Link to related experiments

## 🔄 Migration Guide

When cleaning up old experiments:

1. **Move misplaced results:**
   ```bash
   # From code directory
   mv code/results_* results/
   
   # From non-standard names
   mv results_improved results/improved
   ```

2. **Create standard structure:**
   ```bash
   mkdir -p experiment_name/{code,data,results}
   ```

3. **Document the experiment:**
   - Create README.md
   - Update EXPERIMENT_REGISTRY.md

## 📊 Benefits

- **Consistency**: Easy to navigate any experiment
- **Clarity**: Clear separation of concerns
- **Reproducibility**: Input and output clearly defined
- **Maintainability**: Easy to clean up or archive

Last Updated: 2025-01-03