# Theoretical Framework for InsightSpike

## 1. Core Hypothesis: The Insight Discovery Problem

### 1.1 Problem Formulation

Given:
- A question Q in natural language
- A knowledge base K = {k₁, k₂, ..., kₙ} of facts/documents
- An embedding function f: Text → ℝᵈ

The insight discovery problem seeks to find or generate an answer A such that:
1. A semantically addresses Q
2. A is derivable from K through reasoning
3. A may not exist explicitly in K

### 1.2 The Vector Space Gap

Our experiments reveal a fundamental challenge:
- Questions and their ideal answers maintain similarity ≈ 0.8
- This indicates they occupy related but distinct regions in vector space
- Direct similarity search from Q will not find A if A ∉ K

## 2. Graph-Based Knowledge Representation

### 2.1 Knowledge Graph Construction

We model knowledge as a directed graph G = (V, E) where:
- V = {vᵢ} represents knowledge items (episodes)
- E = {(vᵢ, vⱼ, wᵢⱼ)} represents semantic relationships
- wᵢⱼ = similarity(f(vᵢ), f(vⱼ))

### 2.2 Episode Definition

An episode e is a tuple:
```
e = (text, embedding, c_value, metadata)
```
where:
- text: raw knowledge content
- embedding: f(text) ∈ ℝᵈ
- c_value: confidence/quality score
- metadata: temporal and contextual information

## 3. Message Passing for Insight Discovery

### 3.1 The Message Passing Framework

We adapt Graph Neural Network principles to knowledge integration:

**Node Update Rule**:
```
h_i^(t+1) = σ(W_self · h_i^(t) + Σⱼ∈N(i) α_ij · W_msg · h_j^(t))
```

where:
- h_i^(t): node i's representation at iteration t
- N(i): neighbors of node i
- α_ij: attention weight between nodes i and j
- W_self, W_msg: learnable parameters (or fixed in our case)

### 3.2 Question-Aware Attention

The key innovation is making attention weights question-aware:
```
α_ij = softmax(sim(h_i, h_j) · (1 + β · sim(h_j, Q)))
```

This biases information flow toward question-relevant paths.

## 4. Insight Vector Construction

### 4.1 Weighted Integration

After T iterations of message passing, we construct the insight vector:
```
X = Σᵢ w_i · h_i^(T) / Σᵢ w_i
```

where weights w_i incorporate:
- Question relevance: sim(h_i^(T), Q)
- Node confidence: c_value_i
- Structural importance: degree centrality

### 4.2 Why Weighted Beats Uniform

Our experiments demonstrate that weighted integration:
- Filters noise from irrelevant knowledge
- Preserves question-answer alignment
- Improves X↔D similarity by average +0.136 in extreme cases

## 5. Spike Detection Theory

### 5.1 Information Gain Through Integration

We measure the "spike" (insight emergence) as:
```
IG(X, K) = H(X) - E[H(X|K)]
```

where:
- H(X): entropy of the insight vector
- E[H(X|K)]: expected entropy given knowledge base

High IG indicates novel insight synthesis beyond simple retrieval.

### 5.2 Graph Edit Distance (GED)

We use GED to measure structural novelty:
```
GED(G_before, G_after) = min_P Σ cost(op) for op ∈ P
```

where P is an edit path transforming the graph before to after insight integration.

## 6. Theoretical Justification

### 6.1 Convergence Properties

The message passing process exhibits:
- **Monotonic improvement**: Each iteration moves X closer to answer space
- **Bounded updates**: Vector norms remain stable
- **Question alignment**: X maintains connection to Q throughout

### 6.2 Expressiveness

The framework can theoretically:
- Capture multi-hop reasoning (through message passing depth)
- Integrate disparate knowledge (through weighted aggregation)
- Generate novel combinations (through vector arithmetic in embedding space)

## 7. Connection to Existing Theory

### 7.1 Relation to GNNs

Our approach is a specialized GNN where:
- Node features are text embeddings
- Edge weights are semantic similarities
- The task is generative rather than classificatory

### 7.2 Relation to Retrieval-Augmented Generation

Unlike RAG systems that retrieve then generate, we:
- Integrate at the vector level before generation
- Use graph structure to guide integration
- Measure novelty explicitly through IG and GED

## 8. Empirical Validation

Our theoretical framework is supported by:
1. **Vector space analysis**: Confirming Q-A separation
2. **Integration experiments**: Validating weighted approach
3. **Spike detection**: Successful identification of insights in practice

## 9. Future Theoretical Directions

### 9.1 Optimal Message Passing Depth

Determining T^* that maximizes:
```
T^* = argmax_T [IG(X^(T)) - λ · computational_cost(T)]
```

### 9.2 Learnable Integration

Moving from fixed weights to learned parameters:
```
w_i = MLP(h_i^(T), Q, context)
```

### 9.3 Theoretical Bounds

Establishing guarantees on:
- Convergence rates
- Approximation quality
- Insight novelty measures

## 10. Conclusion

This theoretical framework provides:
- Mathematical foundation for insight discovery
- Justification for architectural choices
- Clear metrics for evaluation
- Directions for future research

The combination of graph-based knowledge representation, question-aware message passing, and weighted integration creates a principled approach to the challenging problem of automated insight discovery.