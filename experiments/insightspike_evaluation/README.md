# InsightSpike Evaluation Experiments

## Overview
This directory contains three different evaluations of the InsightSpike-AI system, each testing different aspects and configurations of the RAG (Retrieval-Augmented Generation) implementation.

## Experiment Structure

### 1. Complete Analysis (`complete_analysis/`)
- **Purpose**: Baseline evaluation with proper state saving
- **Documents**: 50
- **Insights Generated**: 39 (78% ratio)
- **Key Feature**: Complete state persistence with backup functionality
- **Status**: ✅ Working, but retrieval needs improvement

### 2. Fixed Analysis (`fixed_analysis/`)
- **Purpose**: Improved version with bug fixes and optimizations
- **Documents**: 100
- **Insights Generated**: 79 (79% ratio)
- **Key Feature**: Performance optimizations and partial retrieval fix
- **Status**: ✅ Best performing version

### 3. Full Analysis (`full_analysis/`)
- **Purpose**: Full-scale system test
- **Documents**: 100
- **Insights Generated**: 0 (0% ratio)
- **Key Feature**: Attempted full-scale deployment
- **Status**: ❌ Has issues - no insights generated

## Key Metrics Comparison

| Metric | Complete | Fixed | Full |
|--------|----------|-------|------|
| Documents Processed | 50 | 100 | 100 |
| Insights Generated | 39 | 79 | 0 |
| Insight Ratio | 78% | 79% | 0% |
| Avg Processing Time | 0.38s | 0.29s | 0.31s |
| Retrieval Accuracy | 0% | 5% | 0% |
| Avg Insight Quality | 82.8% | 82.4% | 0% |

## Directory Structure
```
insightspike_evaluation/
├── complete_analysis/
│   ├── code/         # Experiment scripts
│   ├── data/         # Input data (if any)
│   └── results/      # Output results and analysis
├── fixed_analysis/
│   ├── code/
│   ├── data/
│   └── results/
└── full_analysis/
    ├── code/
    ├── data/
    └── results/
```

## Key Findings
1. **Insight Generation**: Both complete and fixed analyses show consistent ~78-79% insight generation rates
2. **Performance**: Fixed analysis improved processing speed by 23%
3. **Retrieval Issues**: All experiments struggle with retrieval accuracy (max 5%)
4. **Scaling**: The full analysis experiment failed completely, suggesting configuration issues

## Recommendations
- Use the **fixed_analysis** version for production experiments
- Focus on improving retrieval accuracy across all versions
- Debug the full_analysis configuration issues
- Consider implementing better retrieval indexing strategies

## Related Work
These experiments are part of the larger geDIG embedding evaluation project. See the parent directory for related experiments on:
- Dynamic RAG comparison
- Foundational intrinsic motivation
- Graph embedding strategies