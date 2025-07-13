# Experiments Directory Reorganization Summary

## Date: 2025-01-14

## Reorganization Actions

### 1. **comparative_study/**
- ✅ Created standard directory structure
- ✅ Moved Python files → `src/`
- ✅ Moved `experiment_config.json` → `data/input/`
- ✅ Moved `logs/` → `results/logs/`
- ✅ Cleaned up `__pycache__`

### 2. **english_insight_experiment/**
- ✅ Created standard directory structure
- ✅ Moved `run_english_experiment.py`, `visualize_english_insights.py` → `src/`
- ✅ Moved data files (`*.json`, `*.csv`, `*.yaml`) → `data/input/`
- ✅ Moved visualizations (`*.png`) → `results/visualizations/`
- ✅ Moved reports (`*.md`) → `results/`
- ✅ Moved result files → `results/outputs/`

### 3. **gedig_validation_/**
- ✅ Created missing directories
- ✅ Kept Python files in `src/`
- ✅ Moved `graph_pyg.pt` → `data/input/`
- ✅ Moved visualizations → `results/visualizations/`
- ✅ Moved outputs → `results/outputs/`
- ✅ Moved logs and reports → `results/`

### 4. **query_transformation/** (New)
- ✅ Created new experiment structure
- ✅ Moved `query_transformation_prototype.py` → `src/`
- ✅ Created README.md

### 5. **quick_validation/**
- ✅ Created `src/` and `data/` directories
- ✅ Moved Python files → `src/`
- ✅ Moved documentation → `results/`

## Standard Structure Applied

All experiments now follow:
```
experiment_name/
├── src/                  # Experiment programs
├── data/                 # Experiment data
│   ├── input/           # Input data
│   └── processed/       # Processed data
├── results/             # Experiment results
│   ├── metrics/         # Evaluation metrics
│   ├── outputs/         # Output files
│   └── visualizations/  # Graphs and charts
├── data_snapshots/      # Data folder backups
└── README.md            # Experiment description
```

## Benefits
1. **Consistency**: All experiments follow the same structure
2. **Clarity**: Clear separation of code, data, and results
3. **Reproducibility**: Easy to understand and reproduce experiments
4. **Version Control**: Large data files properly separated from code
5. **Backup Ready**: data_snapshots/ ready for experiment data backups