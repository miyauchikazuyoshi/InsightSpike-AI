# Preliminary Experiments: Vector Space Analysis and Integration Methods

## Abstract

We present preliminary experiments investigating the vector space properties of sentence embeddings in the context of insight discovery systems. Our experiments reveal critical insights about (1) the distribution of questions and answers in embedding space, (2) the challenges of arithmetic expression representation, and (3) the effectiveness of different vector integration strategies. These findings inform the design of InsightSpike's message passing mechanisms and provide empirical justification for our architectural decisions.

## 1. Introduction

Modern NLP systems rely heavily on dense vector representations produced by sentence transformers. While these embeddings excel at capturing semantic similarity, their behavior in specialized contexts—particularly for insight discovery and knowledge integration—remains underexplored. We conducted three preliminary experiments to characterize these properties and inform our system design.

## 2. Experimental Setup

### 2.1 Embedding Model
All experiments use `sentence-transformers/all-MiniLM-L6-v2`, producing 384-dimensional embeddings. This model was chosen for its balance of performance and computational efficiency.

### 2.2 Similarity Metric
We employ cosine similarity throughout, defined as:
$$\text{sim}(u, v) = \frac{u \cdot v}{||u|| \cdot ||v||}$$

## 3. Experiment 1: Question-Answer Vector Space Relationships

### 3.1 Motivation
A critical assumption in many QA systems is that questions and their answers occupy similar regions in vector space. We test this hypothesis empirically.

### 3.2 Method
We analyzed three question-answer pairs across different domains:
1. Scientific concepts (black holes)
2. Cognitive science (creativity)
3. Computer science (recursion)

For each pair, we computed:
- Direct Q↔A similarity
- Similarities between Q/A and related documents
- Statistical analysis of similarity distributions

### 3.3 Results

| Question Domain | Q↔A Similarity | Avg Q↔Docs | Avg A↔Docs | Std Dev |
|----------------|----------------|-------------|-------------|---------|
| Black Holes    | 0.791          | 0.712       | 0.780       | 0.114   |
| Creativity     | 0.805          | 0.765       | 0.814       | 0.077   |
| Recursion      | 0.830          | 0.726       | 0.775       | 0.090   |

**Key Finding**: Questions and answers maintain high similarity (≈0.8) but occupy distinct regions in vector space, suggesting that direct similarity search from Q may not find optimal answers.

## 4. Experiment 2: Arithmetic Expression Representation

### 4.1 The "Hairball" Phenomenon
We discovered that arithmetic expressions create a dense cluster in embedding space, with abnormally high inter-expression similarities.

### 4.2 Analysis
Testing 36 arithmetic expressions (e.g., "2+3", "15÷3", "18-11"):

```
Average similarity: 0.945 ± 0.024
Min similarity: 0.866
Max similarity: 0.992
```

### 4.3 Implications
This clustering effect poses challenges for:
- Mathematical reasoning tasks
- Distinguishing between different arithmetic operations
- Knowledge graph construction in mathematical domains

## 5. Experiment 3: Weighted vs. Uniform Integration

### 5.1 Integration Strategies
Given a question Q and knowledge items {K₁, K₂, ..., Kₙ}, we compare two strategies for creating an integrated representation X:

**Weighted Integration**:
$$X_w = \frac{Q + \sum_{i} \text{sim}(Q, K_i) \cdot K_i}{1 + \sum_{i} \text{sim}(Q, K_i)}$$

**Uniform Integration**:
$$X_u = \frac{1}{n+1}(Q + \sum_{i} K_i)$$

### 5.2 Results

#### Standard Test Cases
| Test Case | Method | X↔Q | X↔D | X↔Items |
|-----------|---------|------|------|---------|
| Scientific Discovery | Weighted | 0.928 | 0.755 | 0.868 |
| Scientific Discovery | Uniform | 0.824 | 0.734 | 0.860 |
| Problem Solving | Weighted | 0.913 | 0.810 | 0.865 |
| Problem Solving | Uniform | 0.810 | 0.799 | 0.859 |

#### Extreme Cases (High Variance in Relevance)
| Test Case | Method | X↔D Similarity |
|-----------|---------|----------------|
| High Variance (mixed relevance) | Weighted | 0.833 |
| High Variance (mixed relevance) | Uniform (Q included) | 0.634 |
| High Variance (mixed relevance) | Uniform (Q excluded) | 0.450 |
| All High Relevance | Weighted | 0.758 |
| All High Relevance | Uniform (Q included) | 0.668 |

### 5.3 Analysis

1. **Weighted integration consistently outperforms uniform integration** when targeting specific answers (D)
2. **The advantage is most pronounced with high variance in item relevance**, where weighted integration effectively filters noise
3. **Uniform integration degrades significantly** when irrelevant items are included

## 6. Discussion

### 6.1 Vector Space Insights
Our experiments reveal that sentence transformer embeddings exhibit:
- **Semantic coherence**: Related concepts cluster appropriately
- **Question-answer separation**: Q and A occupy distinct but related regions
- **Domain-specific anomalies**: Arithmetic expressions form abnormally dense clusters

### 6.2 Design Implications for InsightSpike

1. **Message Passing Architecture**: The Q-A separation justifies our message passing approach, which bridges the gap between question and answer spaces through iterative refinement

2. **Weighted Integration**: Our results strongly support using relevance-weighted integration in the insight vector creation process

3. **Question-Aware Processing**: The consistent 0.8 similarity between Q and A suggests that maintaining question context during processing is crucial

## 7. Limitations and Future Work

1. **Scale**: Current experiments use 3-10 test cases; production validation requires 100+ examples
2. **Domain Coverage**: Extended testing across more diverse domains needed
3. **Model Variations**: Results may vary with different embedding models

## 8. Conclusion

These preliminary experiments provide empirical justification for key architectural decisions in InsightSpike:
- Message passing to bridge Q-A vector space gaps
- Weighted integration for noise filtering
- Question-aware processing throughout the pipeline

The findings also identify challenges (arithmetic clustering) that inform our ongoing development of specialized handling for mathematical content.

## References

[To be added based on paper requirements]

## Appendix: Experimental Code

All experimental code is available at:
```
experiments/pre-experiment/src/
├── test_vector_similarity.py
├── test_uniform_weight_integration.py
└── test_extreme_uniform_integration.py
```