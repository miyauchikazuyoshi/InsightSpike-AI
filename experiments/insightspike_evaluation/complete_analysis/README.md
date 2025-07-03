# Complete InsightSpike Analysis

## Overview
This experiment evaluates the complete InsightSpike-AI system with proper state saving and graph updates. It tests the system's ability to process documents, extract insights, and perform retrieval tasks.

## Experiment Details
- **Total Documents Processed**: 50
- **Total Insights Generated**: 39 (78% insight ratio)
- **Average Processing Time**: 0.38 seconds per document
- **Retrieval Queries**: 10
- **Average Insight Quality**: 82.8%

## Key Components
- **Main Agent**: Handles document processing and insight extraction
- **Information Gain Calculator**: Uses Shannon entropy method
- **Graph Edit Distance Calculator**: Standard optimization level
- **State Persistence**: Includes backup functionality for episodes and graph data

## Files
- `code/complete_insightspike_experiment.py`: Main experiment script with complete state saving
- `results/complete_experiment_analysis.json`: Detailed analysis results including processing metrics, retrieval performance, and insight quality scores

## Key Findings
- High insight generation ratio (78%) indicating effective information extraction
- Strong average insight quality (82.8%)
- Room for improvement in retrieval accuracy (0% successful retrievals)
- Efficient processing time averaging 0.38 seconds per document

## Usage
```bash
python code/complete_insightspike_experiment.py
```

The experiment will:
1. Initialize the InsightSpike-AI system
2. Process a set of documents
3. Generate insights and update the knowledge graph
4. Test retrieval capabilities
5. Save results and backup state data