# Generative Grammar and Graph Shape Matching in InsightSpike

## Core Insight: Syntactic Trees as Subgraphs

The syntactic tree structures from generative grammar can be treated as subgraphs within the brain's semantic graph, enabling shape-based pattern matching that transcends surface variations.

## The Triangle Recognition Analogy

Just as humans can recognize a triangle regardless of:
- Rotation
- Scale
- Distortion
- Line style (straight, curved, broken)

The brain might recognize syntactic patterns regardless of:
- Lexical variation
- Language differences
- Metaphorical usage
- Contextual embedding

## Generative Grammar as Graph Templates

### Traditional View
```
S → NP VP
NP → Det N
VP → V NP

Tree structure:
    S
   / \
  NP  VP
 / \  / \
Det N V  NP
```

### Graph Shape View
```python
class SyntacticShape:
    """Syntactic pattern as graph shape"""
    def __init__(self, pattern_type):
        self.shape = nx.Graph()
        
        if pattern_type == "transitive_action":
            # Agent → Action → Patient
            self.shape.add_edges_from([
                ("agent", "action"),
                ("action", "patient")
            ])
        
        elif pattern_type == "causal_chain":
            # Cause → Effect → Consequence
            self.shape.add_edges_from([
                ("cause", "effect"),
                ("effect", "consequence")
            ])
```

## Shape-Based Memory Search

### Traditional Approach
```python
# String matching or tree matching
def find_similar_sentences(query):
    return [s for s in corpus if tree_distance(parse(query), parse(s)) < threshold]
```

### Shape Matching Approach
```python
def find_shape_similar_memories(query_shape, memory_graph):
    """
    Find subgraphs in memory that match the query shape,
    regardless of specific content
    """
    matches = []
    
    # Extract query shape (topology only)
    query_pattern = extract_graph_shape(query_shape)
    
    # Search for isomorphic subgraphs
    for subgraph in memory_graph.get_subgraphs():
        if is_shape_similar(query_pattern, subgraph):
            matches.append({
                'subgraph': subgraph,
                'shape_similarity': compute_shape_similarity(query_pattern, subgraph),
                'content': extract_content(subgraph)
            })
    
    return matches
```

## Graph Isomorphism in Cognition

### Exact Isomorphism (Too Rigid)
```python
# Requires perfect structural match
nx.is_isomorphic(G1, G2)  # True/False only
```

### Cognitive Isomorphism (Flexible)
```python
def cognitive_isomorphism(shape1, shape2, tolerance=0.2):
    """
    Allows for minor variations in structure
    Like recognizing a slightly distorted triangle
    """
    # Core structure preservation
    core_similarity = compare_core_topology(shape1, shape2)
    
    # Allow some variation
    if core_similarity > (1 - tolerance):
        return True, core_similarity
    
    return False, core_similarity
```

## Implications for InsightSpike

### 1. Memory Storage
Instead of storing exact sentences, store syntactic shapes:
```python
memory = {
    "content": "The cat chased the mouse",
    "shape": TransitiveActionShape(),
    "roles": {
        "agent": "cat",
        "action": "chase",
        "patient": "mouse"
    }
}
```

### 2. Pattern Recognition
Find similar situations through shape matching:
```python
# Query: "The dog pursued the rabbit"
# Finds: All transitive action patterns, regardless of specific actors
similar_patterns = find_shape_matches(
    query_shape=TransitiveActionShape(),
    role_constraints={"action": "movement_verbs"}
)
```

### 3. Cross-Domain Transfer
```python
# Physical domain: "push → move → stop"
# Matches with:
# Social domain: "influence → change → stabilize"
# Economic domain: "invest → grow → plateau"
# All share the same causal chain shape
```

## Implementation Strategy

### Phase 1: Shape Extraction
```python
class ShapeExtractor:
    def extract_from_text(self, text):
        # 1. Parse to syntactic tree
        tree = parse(text)
        
        # 2. Abstract to shape
        shape = self.tree_to_shape(tree)
        
        # 3. Normalize (rotation-invariant representation)
        normalized = self.normalize_shape(shape)
        
        return normalized
```

### Phase 2: Shape-Based geDIG
```python
class ShapeAwareGeDIG:
    def calculate_shape_similarity(self, shape1, shape2):
        # Structural similarity at shape level
        ged_shape = self.shape_edit_distance(shape1, shape2)
        
        # Information gain from shape matching
        ig_shape = self.shape_information_gain(shape1, shape2)
        
        return self.gediq_objective(ged_shape, ig_shape)
```

### Phase 3: Memory Chunking
```python
class GenerativeMemoryChunker:
    def chunk_experience(self, experience):
        # Extract syntactic shapes
        shapes = self.extract_all_shapes(experience)
        
        # Group by shape similarity
        shape_clusters = self.cluster_by_shape(shapes)
        
        # Create memory nodes for each cluster
        memories = []
        for cluster in shape_clusters:
            memories.append({
                'shape': cluster.representative_shape,
                'instances': cluster.members,
                'abstraction': self.abstract_pattern(cluster)
            })
        
        return memories
```

## Connection to Wake-Sleep Cycles

### Wake Phase: Shape Detection
- Identify syntactic shapes in input
- Match against known shape patterns
- Create new shape patterns for novel structures

### Sleep Phase: Shape Consolidation
- Merge similar shapes
- Extract shape invariants
- Discover shape transformations

## Biological Plausibility

### Visual Cortex Analogy
- V1: Edge detection → Syntax: Word boundaries
- V2: Shape detection → Syntax: Phrase structure
- V4: Object recognition → Syntax: Complete patterns

### Invariance Properties
Just as visual system achieves:
- Translation invariance
- Rotation invariance  
- Scale invariance

Linguistic system achieves:
- Lexical invariance
- Syntactic invariance
- Semantic invariance

## Research Directions

1. **Shape Grammar Library**: Build repertoire of common syntactic shapes
2. **Cross-Linguistic Shapes**: Find universal patterns across languages
3. **Shape Evolution**: How shapes change during learning
4. **Efficient Shape Matching**: Algorithms for large-scale shape search

## Key Insight

**"The brain doesn't memorize sentences, it memorizes shapes of meaning."**

Just as we recognize triangles regardless of their specific instantiation, we recognize linguistic patterns regardless of their surface realization. This explains:
- Why we understand novel metaphors
- How children generalize from limited examples
- Why translation preserves meaning across languages
- How insights transfer across domains

---
*This shape-based view of memory unifies generative grammar, graph theory, and cognitive pattern recognition into a coherent framework for InsightSpike.*