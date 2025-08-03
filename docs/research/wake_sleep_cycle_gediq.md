# Wake-Sleep Cycle and geDIG Reward Function Usage

## Key Discovery
During dynamic learning experiments, we discovered that we should have been **minimizing cost** during the expansion phase, not maximizing reward. This fundamental insight led to a clearer separation of concerns.

## Wake-Sleep Cycle Framework

### Wake Phase (Expansion/Inference)
- **Operations**: Node addition, Edge addition
- **Objective**: **Cost minimization**
  ```
  min_P [Accretion(P) + Churn(P)]
  ```
- **Purpose**: Find the most economical way to add knowledge or resolve queries
- **Human analogy**: Active learning, exploration, hypothesis formation

### Sleep Phase (Maintenance/Consolidation)
- **Operations**: 
  - Primary: Edge deletion, Edge rewiring
  - Secondary: Branching & intermediate node formation (for contradiction resolution)
- **Objective**: **Reward maximization**
  ```
  R = λ * ΔIG - μ * (Churn + Accretion)
  ```
- **Purpose**: 
  - Consolidate knowledge, remove redundancy
  - Resolve internal contradictions through minimal branching
  - Discover latent connections between distant concepts
- **Human analogy**: Memory consolidation during REM sleep, creative insights in dreams

## Why This Separation Works

1. **Eliminates conflicting objectives**: Previously, using the same reward function for both adding and pruning created confusion
2. **Natural role division**: 
   - Wake = "Add only what's necessary" (cost consciousness)
   - Sleep = "Keep only what improves information gain" (quality consciousness)
3. **Matches biological processes**: Mirrors how human brains handle learning vs consolidation

## Implementation Implications

### For Current Experiments
The dynamic learning experiments should switch from reward maximization to cost minimization when adding new nodes/edges.

### For Query Processing
- Query arrives → Wake mode → Find minimal cost intermediate nodes
- Background processing → Sleep mode → Prune inefficient connections

### For Contradiction Resolution
- External contradiction (from query) → Wake mode → Add minimal bridging nodes
- Internal contradiction (discovered during maintenance) → Sleep mode → Add branching/intermediate nodes while maximizing overall reward
- Both cases use minimal node addition, but with different optimization objectives

## GED Components Clarification

### Churn vs Accretion
- **Accretion**: Natural growth (adding nodes/edges)
- **Churn**: Reorganization (rewiring existing connections)
- Most learning is Accretion-heavy, but Churn spikes indicate deep insights

### IG Redefinition
- Move from entropy-based to **distance-based** IG
- Better suited for SentenceTransformer embeddings
- Measures semantic consolidation rather than probability distributions

## Structural Gain Concept
The overarching principle: Seek "structural gain" (構造利得) - achieving the same information capacity with better organization. This drives both wake (efficient addition) and sleep (effective pruning) phases.

## Key Insight on Sleep Phase
While sleep phase primarily focuses on pruning and rewiring, it can also perform **internal-driven node creation**:
- Discovers latent contradictions that weren't apparent during wake phase
- Creates branching or intermediate nodes to resolve these contradictions
- This mirrors human REM sleep where creative connections and problem-solving occur

The difference is in the trigger:
- **Wake**: External query/stimulus drives node creation (cost minimization)
- **Sleep**: Internal consistency drives node creation (reward maximization within consolidation)

## Future Directions
1. Implement mode switching based on ΔIG activity
2. Design deletion logic for sleep phase (frequency-based, value-based, or time-decay)
3. Add contradiction detection during sleep phase maintenance
4. Test wake-sleep cycles on long-running experiments to prevent stagnation
5. Measure the ratio of wake-driven vs sleep-driven node creation

---
*Note: This framework resolves the confusion around when to use geDIG as a reward vs cost function, providing a cleaner theoretical foundation for the InsightSpike system.*