# English Insight Reproduction Experiment

## Overview
This experiment reproduces the successful English insight experiment using the current InsightSpike implementation with DistilGPT2 for local execution.

## Experiment Design

### Knowledge Base
- 50 English knowledge episodes across 5 phases
- Progressive complexity from basic concepts to emergent insights
- Topics: Energy, Information, Entropy, Quantum Mechanics, Consciousness

### Test Questions
- 6 questions designed to elicit insight spikes
- Each question requires integration across multiple knowledge phases

### Model
- **DistilGPT2**: Small, fast, locally executable language model
- No external API dependencies
- Fully reproducible results

## Running the Experiment

```bash
cd experiments/english_insight_reproduction
poetry run python src/run_english_insight_experiment.py
```

## Expected Results
Based on the original experiment:
- ~83% spike detection accuracy
- Significant graph complexity increases (~127%)
- Integration of knowledge from multiple phases

## Files
- `config_experiment.yaml` - Experiment configuration
- `data/input/english_knowledge_base.json` - Knowledge episodes
- `data/input/test_questions.json` - Test questions
- `src/run_english_insight_experiment.py` - Main experiment script
- `results/` - Output directory for results

## Success Criteria
1. Spike detection accuracy > 70%
2. Measurable graph complexity changes
3. Coherent responses integrating multiple knowledge phases
4. Reproducible results with fixed random seed