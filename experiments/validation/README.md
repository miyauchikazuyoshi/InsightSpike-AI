# Experimental Validation Framework

This directory contains the experimental validation framework for InsightSpike-AI, implementing rigorous statistical methodology and addressing methodological concerns.

## Files

- `run_experimental_validation.py` - Main validation script implementing comprehensive experimental methodology
- `run_comprehensive_experiment.py` - Comprehensive experiment execution with multiple configurations
- `run_ablation_experiment.py` - Ablation studies to understand component contributions

## Methodological Standards

This framework implements:

✅ **Data Leak Elimination**: No hardcoded test responses
✅ **Competitive Baselines**: State-of-the-art comparison systems
✅ **Large-Scale Evaluation**: 1000+ samples per experiment
✅ **Standard Datasets**: OpenAI Gym, SQuAD, ARC, Natural Questions
✅ **Statistical Rigor**: Cross-validation, significance testing
✅ **Reproducibility**: Fixed seeds, documented methodology

## Usage

```bash
# Run complete experimental validation
python experiments/validation/run_experimental_validation.py

# Run comprehensive experiments
python experiments/validation/run_comprehensive_experiment.py

# Run ablation studies
python experiments/validation/run_ablation_experiment.py
```

## Output

Results are saved to `experiments/results/` with:
- Statistical analysis summaries (JSON)
- Detailed experimental reports (Markdown)
- Raw experimental data
- Visualization outputs
