# Layer 4 Refactoring Plan - Evolution Patterns

## Overview
Layer 4 is evolving beyond simple prompt building to become a true **Insight Decoding Layer** that can transform vector-based discoveries into natural language. This document outlines multiple evolution patterns based on different optimization goals.

### Current State Analysis
- **Problem**: Insights are "buried" - system detects spikes but can't explain what was discovered
- **Root Cause**: One-way encoding (text→vector) without proper decoding (vector→text)
- **Symptoms**: 
  - Generic "INSIGHT SPIKE DETECTED" without specifics
  - Mechanical text synthesis ("Additionally...", "Furthermore...")
  - Loss of causal relationships and semantic meaning

### Vision
Transform Layer 4 from a template-based prompt builder into an intelligent insight decoder that can:
1. Decode vector insights into natural language
2. Explain graph transformations as coherent narratives  
3. Eventually operate without LLM dependency through distillation

## Evolution Pattern 1: Enhanced Template System (Immediate Impact)

### Goal
Improve insight explanations without architectural changes.

### Implementation

#### 1.1 Causal Relationship Templates
```python
def _build_insight_explanation(self, graph_info: Dict[str, Any]) -> str:
    """Generate natural explanations for discovered insights"""
    insights = []
    
    # Explain structural simplification
    if graph_info.get("metrics", {}).get("delta_ged", 0) < -0.3:
        insights.append(
            "The knowledge structure has been significantly simplified. "
            "Multiple concepts have merged into a unified understanding, "
            "revealing a simpler underlying pattern."
        )
    
    # Explain new connections with reasoning
    if "connected_nodes" in graph_info:
        nodes = graph_info["connected_nodes"]
        if len(nodes) >= 2:
            # Find common intermediate concepts
            common = self._find_common_concepts(nodes)
            if common:
                insights.append(
                    f"{nodes[0]} and {nodes[1]} are connected through {common[0]}, "
                    f"suggesting they share fundamental properties."
                )
            else:
                insights.append(
                    f"A direct relationship between {nodes[0]} and {nodes[1]} "
                    f"has been discovered, opening new understanding."
                )
    
    return " ".join(insights)
```

#### 1.2 Narrative Flow Templates
```python
# Reasoning path storytelling
def _build_reasoning_narrative(self, reasoning_chain: List[str]) -> str:
    """Convert reasoning steps into a coherent story"""
    narrative = ["To answer your question, I followed this reasoning path:"]
    
    for i, step in enumerate(reasoning_chain):
        if i == 0:
            narrative.append(f"First, I {step.lower()}.")
        elif i == len(reasoning_chain) - 1:
            narrative.append(f"This led me to {step.lower()}.")
        else:
            narrative.append(f"Then, {step.lower()}.")
    
    narrative.append(
        "Through this process, a new insight emerged that connects "
        "previously separate concepts."
    )
    return " ".join(narrative)
```

#### 1.3 Dynamic Template Selection
```python
TEMPLATE_VARIANTS = {
    "discovery": [
        "The relationship between {X} and {Y} has been revealed",
        "{X} and {Y} are more closely connected than previously understood",
        "A hidden link between {X} and {Y} has emerged"
    ],
    "transformation": [
        "Significant conceptual transformation detected",
        "The understanding has evolved dramatically",
        "A paradigm shift in comprehension occurred"
    ]
}
```

### Timeline: 1-2 days
### Risk: Low
### Impact: Immediate improvement in readability

---

## Evolution Pattern 2: kNN-Based Insight Decoding (Quick Win)

### Goal
Leverage existing vector search to decode insights without training.

### Implementation

```python
def decode_insight_vector(self, vec: np.ndarray, k: int = 8) -> str:
    """Decode vector insights using nearest neighbor search"""
    # 1. Find similar episodes
    neighbors, distances = self.faiss_index.search(vec[None, :], k)
    
    # 2. Extract key concepts from neighbors
    concepts = self._extract_key_concepts([self.episodes[i] for i in neighbors[0]])
    
    # 3. Generate insight based on common themes
    if len(concepts) >= 3:
        return (
            f"This insight connects {concepts[0]} and {concepts[1]} "
            f"through their shared relationship with {concepts[2]}. "
            f"This suggests a deeper pattern where {self._infer_pattern(concepts)}."
        )
    
    # 4. Fallback to similarity-based explanation
    return f"This relates to: {', '.join(concepts[:3])}"

def _extract_key_concepts(self, episodes: List[Episode]) -> List[str]:
    """Extract common concepts using TF-IDF or similar"""
    # Implementation: TextRank, RAKE, or simple frequency analysis
    pass
```

### Advanced: kNN + LLM Rewrite
```python
def decode_with_rewrite(self, vec: np.ndarray) -> str:
    """kNN retrieval followed by LLM synthesis"""
    neighbors = self.get_k_nearest(vec, k=5)
    
    prompt = (
        "Synthesize these related insights into a coherent explanation:\n"
        + "\n".join([f"- {n.text}" for n in neighbors])
        + "\n\nUnified insight:"
    )
    
    return self.llm.generate(prompt, max_tokens=150)
```

### Timeline: 2-3 days
### Risk: Low
### Impact: Natural explanations without training

---

## Evolution Pattern 3: Graph-to-Text Generation (Structural Clarity)

### Goal
Directly translate graph transformations into explanatory text.

### Implementation

```python
class GraphNarrativeGenerator:
    """Convert graph changes to natural language stories"""
    
    def explain_transformation(self, before: Graph, after: Graph) -> str:
        """Generate narrative explaining graph transformation"""
        story = []
        
        # 1. New edges (relationships discovered)
        new_edges = after.edges - before.edges
        if new_edges:
            story.append(self._explain_new_connections(new_edges))
        
        # 2. Merged nodes (concept unification)
        merged = self._find_merged_nodes(before, after)
        if merged:
            story.append(self._explain_concept_merger(merged))
        
        # 3. Structural patterns
        if self._is_hub_formation(after) and not self._is_hub_formation(before):
            hub = self._find_hub_node(after)
            story.append(
                f"{hub} has emerged as a central concept, connecting "
                f"multiple previously separate ideas."
            )
        
        return " ".join(story)
    
    def _explain_new_connections(self, edges: Set[Edge]) -> str:
        """Explain why new connections matter"""
        if len(edges) == 1:
            e = list(edges)[0]
            return (
                f"A connection between {e.source} and {e.target} was discovered. "
                f"This {self._infer_relationship_type(e)} relationship suggests "
                f"they {self._infer_implication(e)}."
            )
        else:
            return (
                f"Multiple new connections were discovered, creating a "
                f"richer understanding of how these concepts interrelate."
            )
```

### Advanced: Triple-Based Generation
```python
# Convert to RDF-like triples
triples = [
    ("Energy", "related_to", "Information"),
    ("Information", "measured_by", "Entropy"),
    ("Entropy", "increases_with", "Energy")
]

# Use Graph2Seq model (T5-based)
narrative = graph2text_model.generate(triples)
# Output: "Energy and information are fundamentally related through entropy..."
```

### Timeline: 1 week
### Risk: Medium
### Impact: Clear structural explanations

---

## Evolution Pattern 4: LLM-Free Decoder Training (Long-term Independence)

### Goal
Train a dedicated decoder to eliminate LLM dependency.

### Implementation Roadmap

#### 4.1 Data Collection Phase
```python
class InsightLogger:
    """Collect high-quality LLM outputs for training"""
    
    def log_if_valuable(self, vec: np.ndarray, text: str, metrics: Dict):
        """Log only high-value insights"""
        insight_score = (
            0.4 * metrics.get('delta_ig', 0) +
            0.4 * abs(metrics.get('delta_ged', 0)) +
            0.2 * metrics.get('diversity', 0)
        )
        
        if insight_score > 0.7 and metrics.get('spike_detected'):
            self.training_corpus.append({
                'vector': vec,
                'text': text,
                'score': insight_score,
                'timestamp': time.time()
            })
```

#### 4.2 Progressive Decoder Training
```python
class InsightDecoder(nn.Module):
    """Lightweight decoder for insight vectors"""
    
    def __init__(self, vec_dim=768, hidden_dim=512):
        super().__init__()
        self.projection = nn.Linear(vec_dim, hidden_dim)
        self.decoder = GPT2Model.from_pretrained('distilgpt2')
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, insight_vec):
        # Project vector to decoder space
        hidden = self.projection(insight_vec)
        # Generate text autoregressively
        output = self.decoder(inputs_embeds=hidden)
        return self.lm_head(output)
```

#### 4.3 Distillation Strategy
```python
# Phase 1: Collect 10k high-quality examples
teacher_outputs = collect_llm_outputs(n=10000, quality_threshold=0.8)

# Phase 2: Train decoder with contrastive learning
for vec, good_text, bad_texts in training_batch:
    loss = info_nce_loss(decoder(vec), good_text, bad_texts)
    
# Phase 3: Self-improvement loop
while teacher_dependency_rate > 0.1:
    student_output = decoder(vec)
    if score(student_output) < threshold:
        teacher_output = llm(vec)
        fine_tune(decoder, vec, teacher_output)
    teacher_dependency_rate = update_metrics()
```

### Progressive Milestones

| Stage | Model Size | Training Time | LLM Usage |
|-------|------------|---------------|------------|
| v1 | 100M params | 6 hours | 100% |
| v2 | 500M params | 3 days | 50% |
| v3 | 1B params | 1 week | 20% |
| v4 | 2B params | 2 weeks | <5% |

### Timeline: 1-2 months (full cycle)
### Risk: High
### Impact: Complete LLM independence

---

## Evolution Pattern 5: Hybrid Adaptive System (Recommended Path)

### Goal
Combine all approaches adaptively based on context.

### Implementation

```python
class AdaptiveInsightDecoder:
    """Intelligently choose decoding strategy"""
    
    def decode(self, vec: np.ndarray, context: Dict) -> str:
        confidence = context.get('reasoning_quality', 0)
        spike = context.get('spike_detected', False)
        
        # High confidence + spike: Use trained decoder
        if confidence > 0.8 and spike and self.decoder_available:
            return self.decoder.generate(vec)
        
        # Medium confidence: kNN + templates
        elif confidence > 0.5:
            neighbors = self.knn_search(vec)
            return self.template_synthesis(neighbors)
        
        # Low confidence but spike: LLM exploration
        elif spike:
            return self.llm_explore_insight(vec)
        
        # Default: Enhanced templates
        else:
            return self.template_generation(context)
```

### Adaptive Pipeline
```
[Vector] → [Confidence Check] → [Strategy Selection]
                                         ↓
                    ┌─────────────────────┼─────────────────────┐
                    ↓                     ↓                     ↓
            [Trained Decoder]      [kNN+Template]         [LLM Polish]
                    ↓                     ↓                     ↓
                    └─────────────────────┼─────────────────────┘
                                         ↓
                                 [Quality Check] → [Output/Retrain]
```

### Timeline: Implement incrementally
### Risk: Low (graceful degradation)
### Impact: Best of all approaches

## Implementation Strategy

### Recommended Progression
1. **Week 1**: Pattern 1 (Enhanced Templates) - Immediate improvement
2. **Week 2-3**: Pattern 2 (kNN Decoding) - Leverage existing infrastructure
3. **Month 2**: Pattern 3 (Graph-to-Text) - Structural clarity
4. **Month 3+**: Pattern 4 (Decoder Training) with Pattern 5 (Hybrid) framework

### Success Metrics
- **Insight Specificity**: % of responses that explain what was discovered
- **Causal Clarity**: % of responses that explain why connections matter  
- **LLM Reduction**: Decrease in LLM API calls over time
- **User Satisfaction**: Qualitative feedback on insight explanations

## Configuration Evolution

```yaml
# Phase 1: Template Enhancement
layer4:
  mode: "template_enhanced"
  insight_explanation: true
  causal_templates: true

# Phase 2: kNN Integration  
layer4:
  mode: "knn_decode"
  k_neighbors: 8
  synthesis_method: "weighted"
  
# Phase 3: Graph Narrative
layer4:
  mode: "graph_to_text"
  include_structure: true
  narrative_style: "scientific"
  
# Phase 4: Decoder Training
layer4:
  mode: "adaptive"
  decoder_model: "insight-decoder-v1"
  fallback_cascade: ["decoder", "knn", "template", "llm"]
  confidence_thresholds:
    decoder: 0.8
    knn: 0.6
    template: 0.4
```

## Key Insights from Research

1. **Vector Decoding Challenge**: One-way encoding is the core limitation
2. **Distillation Opportunity**: LLM outputs can train specialized decoders
3. **InsightSpike Advantage**: ΔGED/ΔIG metrics provide automatic quality scoring
4. **Graph Structure**: The graph transformation itself contains the insight story

## Next Steps

1. Implement Pattern 1 templates in `layer4_prompt_builder.py`
2. Add insight logging for future decoder training
3. Prototype kNN-based decoding using existing FAISS index
4. Design experiments to compare pattern effectiveness

## References

- Current implementation: `src/insightspike/implementations/layers/layer4_prompt_builder.py`
- Graph metrics: `src/insightspike/features/graph_reasoning/`
- Vector search: `src/insightspike/implementations/layers/layer2_memory_manager.py`