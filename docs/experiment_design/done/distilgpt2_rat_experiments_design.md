# DistilGPT2 RAT Experiments - Experiment Design

## Experiment Name and Purpose
**Name**: DistilGPT2 RAT Experiments  
**Purpose**: Validate InsightSpike's ability to solve Remote Associates Test (RAT) problems using a lightweight model (DistilGPT2, 82M parameters), demonstrating that creative insight discovery works even with small models.

## Hypothesis
InsightSpike's active connection exploration and spike detection mechanisms can enable even small language models to discover creative associations that traditional approaches (including RAG) cannot find.

## Methodology
### Test Framework
- **RAT Problems**: 3 standard creativity test questions
  - COTTAGE, SWISS, CAKE → CHEESE
  - CREAM, SKATE, WATER → ICE
  - DUCK, FOLD, DOLLAR → BILL

### Three-Way Comparison
1. **Base DistilGPT2**: Direct model inference
2. **Traditional RAG**: Retrieval-augmented generation
3. **InsightSpike**: Graph-based insight detection

### Spike Detection Algorithm
```python
# Connection density-based detection
spike = connection_density > 0.3 and connections >= 2
```

### Implementation Details
- Model: DistilGPT2 (82M parameters)
- Word association knowledge base
- Connection density threshold: 0.3
- Minimum connections for spike: 2

## Key Metrics
1. **RAT Accuracy**: Percentage of correctly solved problems
2. **Processing Time**: Average time per problem
3. **Spike Detection Rate**: Percentage of problems triggering spikes
4. **Connection Density**: Average density of discovered connections
5. **Performance Ratio**: InsightSpike vs baseline improvement factor

## Expected Outcomes
### Performance Targets
- Base LLM: 0-10% accuracy (near random)
- Traditional RAG: 0-20% accuracy (minimal improvement)
- InsightSpike: 60-80% accuracy (significant breakthrough)
- 100% spike detection on solvable problems

### Creative Insight Validation
- Demonstration that RAG alone doesn't enable creative thinking
- Evidence that active exploration finds hidden connections
- Proof that small models can achieve insights with proper guidance
- Quantification of the "insight moment" through spike detection

### Practical Implications
- Lightweight implementation feasible for edge devices
- Creative problem-solving doesn't require massive models
- Graph-based reasoning as force multiplier for small LLMs
- Opening new applications in resource-constrained environments