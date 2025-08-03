# Paper Restructuring Plan

## Overview
Based on the new understanding of wake-sleep cycles and the dual role of geDIG formula, the paper needs fundamental restructuring.

## Key Changes from Original Plan
1. **geDIG as Objective Function**: Clear mathematical formulation as f(G₁, G₂)
2. **Dual-purpose optimization**: Same function, minimize in wake, maximize in sleep
3. **Unified framework**: Language understanding and spatial navigation use same mechanism
4. **Obstacles as Queries**: Revolutionary insight for navigation tasks
5. **Biological grounding**: Wake-sleep cycles as fundamental organizing principle

## Proposed Structure

### 1. Introduction (Maintain current paper's brain-inspired narrative)
- Brain structure and synaptic pruning as inspiration
- Efficient network structures in biological neural systems
- How the brain achieves high cognitive function with minimal energy
- Limitations of current AI systems (accumulation without understanding)
- **Integration**: Wake-sleep cycles as the mechanism for pruning
- InsightSpike proposal: Graph optimization mimicking brain processes
- Research question: Can we create an AI system that forms insights like the human brain?

### 2. Related Work
- Knowledge graphs and reasoning
- Graph neural networks
- **NEW**: Neuroscience of sleep and memory consolidation
- **NEW**: Computational models of synaptic homeostasis

### 3. The geDIG Hypothesis and Objective Function
- Introducing the geDIG objective function: f(G₁, G₂) = λ·ΔIG - μ·ΔGED
- **NEW**: Dual-purpose optimization - same function, different goals
- Wake phase: Minimize f (cost optimization)
- Sleep phase: Maximize f (reward optimization)
- Mathematical foundation: Conservation principles and normalization

### 4. Wake-Sleep Cycle Framework (**NEW SECTION**)

#### 4.1 Wake Phase (Learning/Inference)
- External query-driven processing
- Query-centric sphere search for relevant knowledge
- Objective: minimize f(G₁, G₂) for cost-efficient solutions
- **NEW**: Obstacles as implicit queries in spatial tasks
- Analogous to human active problem-solving

#### 4.2 Sleep Phase (Consolidation/Maintenance)
- Internal consistency-driven processing
- Objective: maximize f(G₁, G₂) for structural quality
- Memory consolidation through graph optimization
- Contradiction resolution via minimal branching
- Discovery of latent patterns and shortcuts
- Analogous to REM sleep creative insights

#### 4.3 Unified geDIG Formulation
- Same components (GED, IG), different usage
- Phase-dependent optimization objectives
- Biological plausibility arguments

### 5. The InsightSpike Architecture

#### 5.1 System Overview
- Core components and their interactions
- **NEW**: Mode manager for wake-sleep transitions

#### 5.2 Wake Mode Implementation
```
Algorithm 1: Wake Mode Query Processing
1: Input: query q, knowledge graph G
2: q_vec ← encode(q)
3: N ← {n ∈ G | ||q_vec - n_vec|| < τ_dist}
4: K* ← argmin_{K⊂N, |K|=k} Cost(K)
5: intermediate ← message_passing(K*, q)
6: response ← LLM(q, K*, similarities)
7: Return response
```

#### 5.3 Sleep Mode Implementation
```
Algorithm 2: Sleep Mode Consolidation
1: Input: knowledge graph G
2: contradictions ← detect_inconsistencies(G)
3: For each contradiction c:
4:   resolution ← find_minimal_resolution(c)
5:   G ← apply_resolution(G, resolution)
6: redundant ← identify_redundant_edges(G)
7: G ← prune_edges(G, redundant)
8: Update parameters to maximize R
```

### 6. Experiments

#### 6.1 Experimental Setup
- Dataset description
- Evaluation metrics
- Baseline comparisons

#### 6.2 Experiment 1: Wake Mode Performance
- Query response accuracy
- Cost efficiency of knowledge access
- Comparison with traditional retrieval

#### 6.3 Experiment 2: Sleep Mode Effectiveness
- Structure optimization metrics
- Contradiction resolution rate
- Long-term stability

#### 6.4 Experiment 3: Full Cycle Benefits
- Performance over extended periods
- Wake-sleep cycle frequency optimization
- Emergent behaviors

### 7. Results

#### 7.1 Quantitative Results
- Performance metrics for each mode
- Ablation studies

#### 7.2 Qualitative Analysis
- Case studies of insight formation
- Visualization of structural changes
- Examples of creative combinations

### 8. Discussion

#### 8.1 Theoretical Implications
- Unified framework for learning and consolidation
- Connections to cognitive science

#### 8.2 Practical Applications
- Lifelong learning systems
- Creative AI assistants
- Self-maintaining knowledge bases

#### 8.3 Limitations and Future Work
- Computational complexity
- Optimal cycle timing
- Extension to other domains

### 9. Conclusion
- Summary of contributions
- Broader impact on AI research

## Mathematical Notation Clarification

### The geDIG Objective Function
```
f(G₁, G₂) = λ·ΔIG(G₁, G₂) - μ·ΔGED(G₁, G₂)
```

### Phase-Specific Optimization
- **Wake Phase**: minimize f (cost-efficient solutions)
- **Sleep Phase**: maximize f (quality improvement)

### Key Innovation
Same objective function, different optimization directions based on operational context.

### Key Definitions
- Accretion: Cost of adding new nodes/edges
- Churn: Cost of rewiring existing connections
- ΔIG: Change in information gain (distance-based, not entropy)

## Writing Strategy

1. **Start with biological motivation** - makes the approach intuitive
2. **Clearly separate the two phases** - avoid confusion about objectives
3. **Use consistent terminology** - wake/sleep, not expansion/maintenance
4. **Provide concrete examples** - show actual graph transformations
5. **Emphasize the innovation** - same formula, different roles

## Timeline
- [ ] Revise introduction with wake-sleep framing
- [ ] Write new Section 4 (Wake-Sleep Framework)
- [ ] Update experiments to clearly test each phase
- [ ] Add biological plausibility arguments throughout
- [ ] Create clear visualizations of mode transitions

---
*This restructuring positions InsightSpike as a biologically-inspired approach to knowledge management, making it more compelling and easier to understand.*