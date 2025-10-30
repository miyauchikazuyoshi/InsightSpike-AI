# GeDIG Validation - Experiment Design

## Experiment Name and Purpose
**Name**: GeDIG (Graph-enhanced Deep Insight Generation) Validation  
**Purpose**: Validate the theoretical geDIG framework using lightweight models to prove that insight discovery can be separated from language generation, demonstrating that even small models can benefit from graph-based reasoning.

## Hypothesis
The geDIG formula 𝓕 = w₁ ΔGED - kT ΔIG can:
1. Predict insight quality independent of LLM size
2. Enable small models to generate better responses when guided by graph insights
3. Provide a theoretical foundation for practical performance improvements

## Methodology
### Core Formula
```
𝓕 = w₁ × ΔGED - kT × ΔIG
```
Where:
- ΔGED: Change in Graph Edit Distance (structure simplification)
- ΔIG: Change in Information Gain (entropy reduction)
- w₁ = 0.6 (structure weight)
- kT = 0.4 (information weight)

### Architecture
1. **GNN Component**:
   - 3-layer Graph Convolutional Network
   - Hidden dimension: 128
   - ReLU activation + 0.1 dropout
   - Message passing for knowledge integration

2. **Enhanced Prompt Builder**:
   - Converts GNN insights to natural language
   - Detects spike conditions (ΔGED > threshold)
   - Preserves graph structural information

3. **Language Model**:
   - DistilGPT2 (82M parameters)
   - Temperature: 0.7
   - Max tokens: 100

### Test Data
- 1000 synthetic knowledge episodes
- Pre-computed graph embeddings
- Heterogeneous knowledge graph with multiple edge types

## Key Metrics
1. **Confidence Scores**: Aggregate insight quality metric
2. **Spike Detection Rate**: Percentage of queries triggering insights
3. **Formula Correlation**: Correlation between 𝓕 values and quality
4. **Insights per Question**: Average number of discoveries
5. **Runtime Efficiency**: Total processing time on CPU

## Expected Outcomes
### Theoretical Validation
- Positive 𝓕 values correlate with insight quality (r > 0.7)
- Optimal w₁/kT ratio around 1.5
- Clear separation between insight and non-insight states
- Mathematical foundation for future optimization

### Performance Improvements
- 100%+ improvement over direct LLM
- 50%+ improvement over standard RAG
- 30%+ spike detection rate
- Sub-20 second runtime on CPU

### Model Independence
- Proof that GNN handles insight discovery
- LLM size doesn't determine insight capability
- Separation of reasoning from generation
- Path to efficient edge deployment

### Practical Implications
- Insight detection works with 82M parameter models
- Graph reasoning as computational leverage
- Theoretical framework guides implementation
- Foundation for hardware-optimized versions