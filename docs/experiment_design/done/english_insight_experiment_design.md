# English Insight Experiment - Experiment Design

## Experiment Name and Purpose
**Name**: English Insight Generation  
**Purpose**: Demonstrate InsightSpike's ability to generate insights by integrating knowledge from multiple conceptual phases using real LLM (DistilGPT2), validating multi-phase knowledge integration for emergent insight discovery.

## Hypothesis
InsightSpike can generate higher-quality insights than traditional RAG by:
1. Integrating knowledge from multiple conceptual levels (phases)
2. Detecting when sufficient integration creates "insight spikes"
3. Generating emergent properties not present in individual knowledge pieces

## Methodology
### Knowledge Base Structure
- 50 episodes organized in 5 phases:
  - Phase 1: Basic Concepts
  - Phase 2: Relationships
  - Phase 3: Deep Integration
  - Phase 4: Emergent Insights
  - Phase 5: Integration and Circulation

### Test Questions
6 fundamental questions about:
- Energy-information relationships
- Consciousness emergence
- Creativity and chaos
- Entropy
- Quantum entanglement
- Unifying principles

### Spike Detection Criteria
- Integration from ≥3 different phases
- Similarity threshold: 0.3
- Confidence score ≥ 60%

### Implementation
- LLM: DistilGPT2 (lightweight validation)
- Embedding: all-MiniLM-L6-v2
- Temperature: 0.7, Top-p: 0.95
- Max tokens: 100

## Key Metrics
1. **Response Quality Score**: Overall answer quality (0-1)
2. **Multi-phase Integration Rate**: Percentage using 3+ phases
3. **Spike Detection Success**: Percentage triggering insight spikes
4. **Graph Structural Complexity**: Change in nodes, edges, density
5. **Emergent Concept Generation**: New concepts not in source material

## Expected Outcomes
### Quantitative Goals
- 80%+ spike detection rate on fundamental questions
- 100%+ increase in graph structural complexity
- Consistent multi-phase integration (3-5 phases)
- Small but significant quality improvement over RAG

### Graph Evolution Evidence
- Node increase: 30-50% when insights emerge
- Edge increase: 150-200% showing new connections
- Density increase: Clear clustering around insights
- Visual before/after graphs showing structural changes

### Emergent Properties
- New conceptual bridges between domains
- Synthesis of abstract principles
- Discovery of hidden relationships
- Generation of unified explanations

### Language Agnostic Validation
- Proof that InsightSpike works beyond Japanese
- Demonstration with English knowledge base
- Foundation for multilingual applications