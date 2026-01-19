# 独立論文アウトライン: GED編集操作 = 閃きの計算的モデル

## 論文タイトル案

**Primary**:
> "Graph Edit Distance as a Computational Model of Scientific Insight"

**Alternative**:
- "From Analogy to Isomorphism: A Unified Theory of Creative Discovery"
- "Structural Unification: How AI Can Discover New Theories"
- "GED-Insight: Toward AI That Discovers, Not Just Retrieves"

---

## Abstract (200 words)

Scientific breakthroughs often arise from recognizing structural similarities across domains—Bohr saw the atom in the solar system; Einstein unified electromagnetism and mechanics. We propose that such "insights" can be computationally modeled as **Graph Edit Distance (GED) operations that transform one knowledge structure into another**.

We formalize three levels of structural discovery: (1) pattern matching, (2) analogy detection via structural similarity, and (3) isomorphism discovery via minimal edit operations. Our framework connects to molecular design AI, where scaffold hopping finds functionally equivalent but structurally different molecules using the same mathematical foundation.

Experiments on cross-domain QA tasks show F1 improvement of +60% when structural similarity is enabled, and we successfully reproduce three historical scientific discoveries (Bohr's atomic model, Kekulé's benzene ring, Darwin's natural selection) as structural analogy detection problems.

This work suggests a path from "AI that retrieves" to "AI that discovers"—a qualitative leap from search engine to scientist.

---

## 1. Introduction (1.5 pages)

### 1.1 The Problem: AI Cannot Discover

```
Current AI capabilities:
- Pattern recognition in data ✓
- Knowledge retrieval and combination ✓
- Statistical correlation detection ✓

What AI cannot do:
- Recognize deep structural analogies ✗
- Unify contradictory theories ✗
- Generate genuinely new concepts ✗
```

### 1.2 Key Insight

Scientific insights are not random—they have **computational structure**:

> The edit operations that transform one knowledge graph into another
> (or make two graphs isomorphic) constitute the "content" of an insight.

**Example**: Einstein's special relativity
- Input: Two contradictory structures (electromagnetism, classical mechanics)
- Output: Lorentz transformation (the minimal edit that unifies them)
- The transformation IS the insight

### 1.3 Connection to Molecular Design AI

Drug discovery AI already uses this principle:
- Scaffold hopping: Find molecules with same function, different structure
- Method: Molecular graph edit distance
- Success: Multiple drugs discovered via computational scaffold hopping

**Our claim**: The same mathematical framework applies to knowledge graphs.

### 1.4 Contributions

1. **Theoretical**: GED edit operations as computational model of insight
2. **Hierarchical framework**: Three levels (pattern → analogy → isomorphism)
3. **Experimental**: F1 +60% on cross-domain QA; 3/3 science history reproductions
4. **Bridge to drug discovery**: Explicit correspondence with scaffold hopping

---

## 2. Background (1 page)

### 2.1 Graph Edit Distance in Molecular Design

- Definition of molecular GED
- Scaffold hopping algorithms (RECAP, BRICS, etc.)
- Success cases in drug discovery

### 2.2 Analogical Reasoning in AI

- Structure Mapping Theory (Gentner, 1983)
- Computational approaches to analogy
- Limitations of current methods

### 2.3 geDIG Framework (Brief)

- F = ΔEPC - λΔIG (single gauge)
- AG/DG gates for "when" decisions
- Phase 1 results (maze, RAG)

---

## 3. Theory: Levels of Structural Discovery (2 pages)

### 3.1 Three-Level Hierarchy

```
Level 3: Isomorphism Discovery
   ┌─────────────────┐
   │ T(A) ≡ T(B)     │ ← Find transformation T
   │ (Einstein)      │
   └────────┬────────┘
            │
Level 2: Analogy Detection
   ┌────────┴────────┐
   │ A ≈ B            │ ← Structural similarity
   │ (Bohr)           │
   └────────┬────────┘
            │
Level 1: Pattern Matching
   ┌────────┴────────┐
   │ similarity(a,b) │ ← Element-wise similarity
   └─────────────────┘
```

### 3.2 Formal Definitions

**Level 1: Pattern Matching**
```
sim(a, b) = cosine(embed(a), embed(b))
```
- Element-level similarity
- Standard embedding approaches

**Level 2: Analogy Detection**
```
SS(G₁, G₂) = similarity(signature(G₁), signature(G₂))
```
- Graph structural similarity
- Motif-based, spectral, or WL-kernel methods
- Implemented in geDIG

**Level 3: Isomorphism Discovery**
```
T* = argmin_T GED(T(G₁), G₂)
```
- Find transformation that minimizes edit distance
- The transformation T* IS the insight

### 3.3 Connection to FEP/MDL

| Concept | FEP | MDL | GED-Insight |
|---------|-----|-----|-------------|
| Goal | Minimize surprise | Minimize description length | Minimize structural distance |
| Update | Prediction error | Compression gain | Edit operation |
| Insight | - | - | Transformation discovery |

### 3.4 Why GED Edit Operations = Insight

Three arguments:
1. **Compression**: Insight reduces description length of combined knowledge
2. **Prediction**: Insight reduces surprise when encountering new domains
3. **Unification**: Insight reveals hidden common structure

---

## 4. Experiments (2 pages)

### 4.1 Experiment 1: Cross-Domain Analogy QA

**Setup**:
- 5 domain pairs × 3-5 questions = 18 questions
- Structure types: hub_spoke, hierarchy, branching, chain, network

**Results**:

| Metric | SS Disabled | SS Enabled | Improvement |
|--------|-------------|------------|-------------|
| F1 Mean | 0.062 | **0.660** | **+60%** |
| Exact Match | 0.0% | 16.7% | +16.7% |
| Analogy Detection | 0.0% | 100% | +100% |

**Key finding**: Hub-spoke structures transfer best (F1=0.831 for solar-atom analogy)

### 4.2 Experiment 2: Science History Simulation

**Scenarios**:
1. Bohr's atomic model (1913): Solar system → Atom
2. Kekulé's benzene ring (1865): Ouroboros → Benzene
3. Darwin's natural selection (1859): Malthus economics → Evolution

**Results**:

| Discovery | Structural Similarity | Analogy Detected | geDIG Improvement |
|-----------|----------------------|------------------|-------------------|
| Bohr | 0.995 | ✓ | Δ = 0.599 |
| Kekulé | 0.967 | ✓ | Δ = 0.590 |
| Darwin | 0.985 | ✓ | Δ = 0.596 |

**3/3 scenarios successfully reproduced**

### 4.3 Experiment 3: Negative Result (HotPotQA Bridge)

- SS did not improve HotPotQA bridge problems (-4.0%)
- Lesson: SS works with **explicit structural patterns**, not implicit NL structures
- This confirms: geDIG's strength is in structured domains

### 4.4 Ablation: Structure Type Analysis

| Structure Type | F1 (SS Enabled) | Example |
|---------------|-----------------|---------|
| hub_spoke | **0.831** | Solar system ≈ Atom |
| hierarchy | 0.695 | Company ≈ Military |
| network | 0.566 | SNS ≈ Epidemic |
| branching | 0.539 | Blood vessels ≈ River |
| chain | 0.468 | Supply chain ≈ Neural |

---

## 5. Discussion (1.5 pages)

### 5.1 What This Means for AI

```
Current AI paradigm:
  Query → Retrieve similar documents → Combine → Generate

GED-Insight paradigm:
  Structures → Detect structural similarity → Find unifying transformation → Generate NEW structure
```

### 5.2 Connection to Molecular Design AI

| Molecular Design AI | GED-Insight |
|---------------------|-------------|
| Molecular graph | Knowledge graph |
| Molecular edit distance | GED |
| Same efficacy, different structure | Same explanatory power, different theory |
| Scaffold hopping | Theory unification |

**Implication**: Algorithms from drug discovery can be adapted for knowledge discovery.

### 5.3 Levels of Insight in Scientific History

- **Level 1 insights**: Recognizing similar patterns (common, incremental)
- **Level 2 insights**: Cross-domain analogies (Bohr, Kekulé)
- **Level 3 insights**: Paradigm shifts (Einstein, Darwin)

### 5.4 Limitations and Future Work

1. **Level 3 not yet implemented**: We detect analogies (Level 2), not discover transformations (Level 3)
2. **Requires explicit graph structure**: Natural language needs graph extraction first
3. **Computational cost**: GED is NP-hard; approximations needed for scale
4. **Validation**: More diverse domains needed

---

## 6. Related Work (1 page)

### 6.1 Analogical Reasoning
- Structure Mapping Theory (Gentner, 1983)
- ANALOGY (Evans, 1968)
- SME (Structure Mapping Engine)

### 6.2 Graph Neural Networks for Knowledge
- Knowledge graph embedding
- Graph-to-sequence models
- Relational reasoning

### 6.3 Molecular Design AI
- RECAP, BRICS for scaffold decomposition
- Molecular transformers
- Graph-based drug discovery

### 6.4 Creativity in AI
- Computational creativity
- Concept blending
- Scientific discovery systems (BACON, AM, etc.)

---

## 7. Conclusion (0.5 page)

We proposed that **GED edit operations constitute the computational essence of scientific insight**. Our three-level framework (pattern → analogy → isomorphism) provides a principled hierarchy for understanding creative discovery.

Experiments demonstrate:
- F1 +60% improvement on cross-domain QA with structural similarity
- 3/3 successful reproductions of historical scientific discoveries
- Clear connection to molecular design AI (scaffold hopping)

This work suggests a path from "AI that retrieves" to "AI that discovers"—a fundamental shift in what artificial intelligence can achieve. The algorithms that discover new drugs may, with adaptation, discover new theories.

---

## Appendix

### A. Implementation Details
- geDIG parameters
- Structural similarity algorithms (signature, spectral, motif, WL-kernel)
- Graph construction from text

### B. Full Experimental Results
- All 18 cross-domain QA examples
- Science history simulation details
- HotPotQA bridge analysis

### C. Theoretical Proofs
- GED ≈ MDL under assumptions
- Structural similarity bounds

---

## References

1. Friston, K. (2010). The free-energy principle.
2. Grünwald, P. D. (2007). Minimum description length principle.
3. Gentner, D. (1983). Structure-mapping.
4. Lewell, X. Q., et al. (1998). RECAP.
5. Degen, J., et al. (2008). BRICS.
6. [geDIG Phase 1 paper - self-citation]

---

## Key Messages

### One sentence:
> "Like drug discovery AI searches for molecular isomorphs, geDIG searches for theoretical isomorphs—the edit operations that unify them are the essence of insight."

### Three-line syllogism:
```
1. Molecular design AI discovers new drugs via molecular graph edits
2. Knowledge is also a graph structure
3. Therefore, knowledge graph edits can discover new theories
```

### Impact statement:
```
Current AI: "Search and combine existing knowledge"
Proposed AI: "Discover new knowledge structures"

= Qualitative leap from search engine to scientist
```

---

## Target Venues

1. **NeurIPS / ICML**: Main ML venues (if experimental results are strong enough)
2. **AAAI**: AI focus, accepts more theoretical work
3. **IJCAI**: Broader AI, good for novel frameworks
4. **Computational Creativity Conference**: Specialized but high-impact for this topic
5. **Nature Machine Intelligence**: If results are paradigm-shifting

## Timeline

- **Month 1-2**: Implement Level 3 (isomorphism discovery) prototype
- **Month 3-4**: Run comprehensive experiments
- **Month 5**: Write and refine paper
- **Month 6**: Submit to target venue

---

## Action Items

1. [ ] Implement Level 3 isomorphism discovery algorithm
2. [ ] Expand cross-domain QA dataset (more domains, more questions)
3. [ ] Add more science history scenarios
4. [ ] Benchmark against existing analogy detection methods
5. [ ] Connect with molecular design AI researchers for collaboration
6. [ ] Prepare visualizations (Figure 1: levels, Figure 2: molecular correspondence)
