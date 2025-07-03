# InsightSpike-AI Experiment Registry

This registry provides a comprehensive overview of all experiments conducted in the InsightSpike-AI project.

## 📊 Experiment Categories

### 1. Core System Evaluation

#### 🔸 InsightSpike Evaluation (`/experiments/insightspike_evaluation/`)
- **Purpose**: Evaluate core InsightSpike-AI functionality
- **Variants**:
  - `complete_analysis/`: Baseline with 50 documents
  - `fixed_analysis/`: Improved version with 100 documents
  - `full_analysis/`: Large-scale test (configuration issues)
- **Key Metrics**: Insight generation rate, processing speed, retrieval accuracy
- **Status**: ✅ Completed

### 2. RAG System Comparisons

#### 🔸 geDIG Embedding Evaluation (`/experiments/gedig_embedding_evaluation/`)
- **Purpose**: Comprehensive evaluation of embedding methods and RAG performance
- **Sub-experiments**:
  - Episodic learning evaluation
  - Improved episodic learning
  - RAG comparison with cleaned data
- **Key Finding**: 79.4% deduplication achieved with proper embeddings
- **Status**: ✅ Completed

#### 🔸 Dynamic RAG Comparison (`/experiments/dynamic_rag_comparison/`)
- **Purpose**: Compare dynamic weight strategies for RAG systems
- **Key Results**: Optimal weight configurations identified
- **Status**: ✅ Completed

#### 🔸 Standalone RAG Comparison (`/experiments/rag_comparison_standalone/`)
- **Purpose**: Earlier RAG comparison experiment
- **Note**: Superseded by geDIG evaluation
- **Status**: ✅ Archived

#### 🔸 Integrated RAG Memory (`/experiments/integrated_rag_memory/`)
- **Purpose**: Test RAG integration with InsightSpike memory system
- **Date**: June 2025
- **Status**: ✅ Completed

### 3. Intrinsic Motivation Studies

#### 🔸 Foundational Intrinsic Motivation (`/experiments/foundational_intrinsic_motivation/`)
- **Purpose**: Study intrinsic motivation in agent learning
- **Environment**: Grid-world mazes
- **Key Metrics**: Exploration efficiency, learning curves
- **Status**: ✅ Completed

### 4. Efficiency Analysis

#### 🔸 Compression Efficiency (`/experiments/compression_efficiency/`)
- **Purpose**: Analyze storage efficiency vs traditional RAG
- **Key Finding**: 19.4x compression ratio
- **Status**: ✅ Completed

### 5. Colab Integration

#### 🔸 Colab Experiments (`/experiments/colab_experiments/`)
- **Purpose**: Google Colab-compatible experiments
- **Contents**:
  - Dynamic RAG comparison notebooks
  - Foundational experiments
- **Status**: ✅ Active

### 6. Data Preservation

#### 🔸 Data Preservation (`/experiments/data_preservation/`)
- **Purpose**: Critical data backups and preservation
- **Contents**: Processed datasets, critical backups
- **Status**: 🔄 Ongoing

## 📈 Performance Summary

### Best Performing Configurations

1. **Insight Generation**: Fixed analysis - 79% success rate
2. **Compression**: 94.8% storage savings vs traditional RAG
3. **Deduplication**: 79.4% reduction in redundant data
4. **Speed**: Hybrid RAG - 26ms retrieval time

### Areas for Improvement

1. **Retrieval Accuracy**: Currently 0-5% in most tests
2. **Large Dataset Performance**: Issues with full-scale tests
3. **GPU Optimization**: Not yet implemented

## 🚀 Future Experiments

1. **GPU Acceleration Testing**
   - Expected 5-10x speedup for InsightSpike-AI
   - FAISS GPU implementation

2. **Large-Scale Evaluation**
   - 1000+ document datasets
   - Multi-language support

3. **Commercial RAG Comparison**
   - Benchmark against OpenAI, Anthropic RAG
   - Cost-benefit analysis

4. **Multi-Modal Integration**
   - Image + text RAG
   - Audio transcription integration

## 📝 Experiment Standards

All experiments should include:
- `code/`: Implementation scripts
- `data/`: Input data or data generation scripts
- `results/`: Output files and visualizations
- `README.md`: Detailed documentation
- Configuration files for reproducibility

## 🔄 Status Legend

- ✅ Completed: Experiment finished, results documented
- 🔄 Ongoing: Active development or continuous monitoring
- ⏸️ Paused: Temporarily suspended
- ❌ Deprecated: No longer maintained
- 📋 Planned: Future experiment

## 📊 Metrics Tracking

Key metrics tracked across experiments:
- **Performance**: Speed, accuracy, F1-score
- **Efficiency**: Memory usage, storage requirements
- **Quality**: Insight detection, relevance scores
- **Scalability**: Performance vs dataset size

## 🤝 Contributing

When adding new experiments:
1. Create proper directory structure
2. Include comprehensive README
3. Add entry to this registry
4. Tag with appropriate status
5. Link related experiments

Last Updated: 2025-01-03