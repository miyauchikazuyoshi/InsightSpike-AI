# Comprehensive geDIG Evaluation v2

## Overview
This is the improved version of the comprehensive geDIG evaluation experiment, addressing reviewer feedback and implementing direct ΔGED/ΔIG calculations.

## Key Improvements from v1
1. **Direct Metric Calculation**: Implements actual ΔGED and ΔIG calculations instead of proxy metrics
2. **Expanded Test Set**: 100+ questions instead of 20
3. **Baseline Comparisons**: Includes RAG and other baseline methods
4. **Statistical Robustness**: Multiple random seeds and proper confidence intervals
5. **Parameter Transparency**: All parameters match paper specifications

## Experiment Design

### Test Set
- Total questions: 100
- Easy: 25 questions
- Medium: 50 questions  
- Hard: 25 questions
- Random seeds: 10 for statistical significance

### Metrics
- Direct ΔGED calculation using NetworkX
- Direct ΔIG calculation using entropy-based method
- Spike detection accuracy
- Processing time
- Memory usage

### Baselines
1. Standard RAG with dense retrieval
2. Graph-based QA baseline
3. Simple similarity-based retrieval

## Implementation Status
- [x] Experiment directory structure
- [ ] Direct ΔGED/ΔIG implementation
- [ ] Expanded question set generation
- [ ] Baseline implementations
- [ ] Statistical analysis framework

## Timeline
- Week 1: Implement direct metrics and expand test set
- Week 2: Implement baselines and run experiments
- Week 3: Analysis and visualization
- Week 4: Documentation and paper updates