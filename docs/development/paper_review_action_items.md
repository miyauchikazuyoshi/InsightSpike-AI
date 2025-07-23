# Paper Review Action Items

## geDIG論文の査読想定フィードバックと改善計画

*Last Updated: 2025-07-23*

### Reviewer Perspective Analysis

#### Strengths Identified:
1. **Difficulty Reversal Phenomenon** - Novel finding that harder questions achieve higher accuracy
2. **Brain-Science Consistency** - 4-layer architecture isn't just metaphor, it actually functions
3. **Math Concept Evolution** - Observing episodic memory splits is genuinely new

#### Weaknesses to Address:

##### 1. Statistical Reliability (HIGH PRIORITY)
- **Issue**: N=20 total, N=5 for hard questions → 95% CI [47.8%, 100%] too wide
- **Action**: 
  - Expand test set to at least 100 questions (20+ per difficulty)
  - Run multiple random seeds
  - Report confidence intervals properly

##### 2. Baseline Comparisons (HIGH PRIORITY)
- **Issue**: No quantitative comparison with existing methods
- **Action**: Implement baselines
  - Standard RAG (Retrieval + GPT-4)
  - Dense retrieval (FAISS + reranking)
  - Graph-based QA (ConceptNet + reasoning)
  - Report accuracy, latency, and insight detection rates

##### 3. Knowledge Base Objectivity (MEDIUM PRIORITY)
- **Issue**: 100-item hierarchical KB might be cherry-picked
- **Action**:
  - Test on standard benchmarks:
    - ConceptNet for general knowledge
    - WikiData for structured facts
    - Scientific paper abstracts for technical domains
  - Create KB construction guidelines with objective criteria

##### 4. Reproducibility (MEDIUM PRIORITY)
- **Issue**: Can other researchers reproduce the results?
- **Action**:
  - Publish complete experiment data
  - Create seed datasets with annotations
  - Provide KB construction tools/scripts

##### 5. Decoding Limitation (ACKNOWLEDGED)
- Already honestly addressed in paper
- Future work: dedicated decoder development

### Experiment TODO List:

1. **Baseline Experiment Suite**
   ```python
   baselines = {
       'rag_gpt4': RAGBaseline(retriever='dense', llm='gpt-4'),
       'rag_claude': RAGBaseline(retriever='dense', llm='claude-3'),
       'conceptnet': ConceptNetBaseline(),
       'graph_qa': GraphQABaseline()
   }
   ```

2. **Standard Benchmark Evaluation**
   - CommonsenseQA dataset
   - ConceptNet QA tasks
   - SciQ (scientific questions)
   - ARC (AI2 Reasoning Challenge)

3. **Statistical Robustness**
   - Bootstrap confidence intervals
   - Cross-validation across question sets
   - Ablation studies on components

### Expected Timeline:
- Week 1-2: Implement baselines
- Week 3-4: Run benchmark evaluations
- Week 5: Statistical analysis and paper revision

### Success Criteria:
- Show geDIG maintains >70% relative improvement over best baseline
- Demonstrate generalization to at least 2 standard benchmarks
- Achieve p < 0.01 statistical significance with proper sample sizes