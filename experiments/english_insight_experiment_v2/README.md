# English Insight Experiment V2

## Overview
- **Created**: 2025-07-20
- **Author**: InsightSpike Team
- **Status**: 🚧 In Progress
- **Purpose**: Re-implementation of english_insight_experiment with new architecture

## Experiment Goals
1. Test InsightSpike's multi-phase knowledge integration with the new GED/IG implementation
2. Compare with baseline approaches (Direct LLM, Standard RAG)
3. Validate spike detection with improved metrics
4. Generate visualizations of knowledge graph evolution

## Configuration
Using the new Pydantic-based configuration system with experiment preset.

## Methods
- **Model**: DistilGPT2 (local)
- **Embedding**: all-MiniLM-L6-v2
- **Knowledge Base**: 50 episodes across 5 phases
- **Evaluation**: Quality scores, spike detection, graph metrics

## Running the Experiment
```bash
cd experiments/english_insight_experiment_v2
poetry run python src/run_experiment.py
```

## Expected Results
- Spike detection rate > 80%
- Multi-phase integration in complex questions
- Visual evidence of graph structure evolution