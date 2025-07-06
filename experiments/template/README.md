# Experiment Template

This is a template for creating new InsightSpike-AI experiments.

## How to Use This Template

1. **Copy this directory** to create a new experiment:
   ```bash
   cp -r experiments/template experiments/YYYY-MM-DD_experiment_XX_your_description
   ```

2. **Update experiment details** in `experiment.py`:
   - Change the class name from `MyExperiment` to something descriptive
   - Update the experiment name and description
   - Fill in the objective, hypothesis, and method

3. **Configure your experiment** in `config.yaml` or `config.json`

4. **Run your experiment**:
   ```bash
   cd experiments/YYYY-MM-DD_experiment_XX_your_description
   python experiment.py
   ```

## Directory Structure

```
your_experiment/
├── README.md          # This file - update with your experiment details
├── config.yaml        # Experiment configuration
├── experiment.py      # Main experiment code
├── code/             # Additional code files
│   └── utils.py      # Helper functions
├── data/             # Experiment data (not main system data)
│   ├── input/        # Input data for this experiment
│   └── output/       # Generated data
├── results/          # Experiment results
│   ├── metrics.json  # Performance metrics
│   ├── logs/         # Experiment logs
│   └── plots/        # Visualizations
└── notebooks/        # Analysis notebooks
    └── analysis.ipynb
```

## Experiment Details

**Name**: [Your Experiment Name]

**Date**: YYYY-MM-DD

**Author**: [Your Name]

**Objective**: 
[What are you trying to achieve?]

**Hypothesis**:
[What do you expect to find?]

**Method**:
[How will you test your hypothesis?]

## Results

[Update this section after running the experiment]

### Key Findings
- Finding 1
- Finding 2

### Metrics
- Metric 1: value
- Metric 2: value

### Conclusions
[What did you learn?]

### Next Steps
[What experiments should follow?]