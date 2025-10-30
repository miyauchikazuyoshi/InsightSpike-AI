---
status: active
category: gedig
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# Message Passing in geDIG Decoder Design

*Created: 2025-07-24*

## Core Insight
Message passing is the fundamental mechanism that enables dynamic, context-aware decoding from vector representations to natural language. It connects subgraph structures, syntax generation, and semantic coherence in a unified framework.

## 1. Subgraph Activation via Message Passing

### Basic Activation Propagation
```python
class SubgraphMessagePassing:
    """
    Propagate activation through knowledge subgraphs
    during decoding process
    """
    def propagate_activation(self, initial_vector, knowledge_graph):
        # Initialize activations based on vector similarity
        active_nodes = {
            node: cosine_similarity(node.vector, initial_vector)
            for node in knowledge_graph.nodes
        }
        
        # Iterative message passing
        for iteration in range(self.max_iterations):
            new_activations = {}
            
            for node in knowledge_graph.nodes:
                # Collect messages from neighbors
                messages = [
                    neighbor.activation * edge.weight
                    for neighbor, edge in node.neighbors
                ]
                
                # Update activation
                new_activations[node] = self.aggregate(
                    node.activation,
                    messages
                )
            
            active_nodes = new_activations
            
            # Check convergence
            if self.converged(active_nodes):
                break
        
        return active_nodes
```

### Hierarchical Message Passing
```python
class HierarchicalMessagePassing:
    """
    Efficient propagation using graph hierarchy
    """
    def hierarchical_decode(self, concept_vector, hierarchical_graph):
        decoded_levels = []
        
        for level in range(hierarchical_graph.depth):
            # Get nodes at current level
            level_nodes = hierarchical_graph.get_level(level)
            
            # Within-level message passing
            level_activations = self.level_message_passing(
                level_nodes,
                concept_vector
            )
            
            # Inter-level messages
            if level > 0:
                parent_messages = self.compute_parent_messages(
                    level_activations,
                    decoded_levels[level-1]
                )
                level_activations = self.integrate_parent_messages(
                    level_activations,
                    parent_messages
                )
            
            # Generate text for this level
            level_text = self.generate_level_description(
                level_activations,
                level
            )
            decoded_levels.append(level_text)
        
        return self.integrate_levels(decoded_levels)
```

## 2. Real-time Generation Guidance

### Token-by-Token Message Guidance
```python
class MessageGuidedDecoding:
    """
    Use message passing to guide next token generation
    """
    def generate_next_token(self, partial_sentence, active_subgraph):
        # Encode current context
        context_vector = encode(partial_sentence)
        
        # Collect messages from active subgraph
        messages = self.collect_messages(
            context_vector,
            active_subgraph
        )
        
        # Predict next concept from messages
        next_concept = self.predict_from_messages(messages)
        
        # Convert concept to token
        if next_concept.activation < self.clarity_threshold:
            # Low activation → pronoun
            return self.generate_pronoun(next_concept)
        else:
            # High activation → specific word
            return next_concept.surface_form
```

### Dynamic Context Adjustment
```python
def adjust_context_via_messages(self, generation_state):
    """
    Dynamically adjust which parts of the graph
    are relevant during generation
    """
    # Current focus
    focus_nodes = generation_state.current_focus
    
    # Spread activation to find related concepts
    expanded_context = self.spread_activation(
        focus_nodes,
        max_hops=3,
        decay_factor=0.7
    )
    
    # Prune irrelevant nodes
    relevant_context = self.prune_by_relevance(
        expanded_context,
        generation_state.query_vector
    )
    
    return relevant_context
```

## 3. Bidirectional Optimization

### Syntax-Graph Co-optimization
```python
class BidirectionalMessagePassing:
    """
    Optimize both syntax tree and concept graph
    simultaneously via message passing
    """
    def optimize_generation(self, syntax_tree, concept_graph):
        for iteration in range(max_iterations):
            # Forward: Graph → Syntax
            syntax_messages = self.graph_to_syntax_messages(
                concept_graph,
                syntax_tree
            )
            syntax_tree = self.update_syntax(syntax_tree, syntax_messages)
            
            # Backward: Syntax → Graph
            graph_messages = self.syntax_to_graph_messages(
                syntax_tree,
                concept_graph
            )
            concept_graph = self.update_graph(concept_graph, graph_messages)
            
            # Check bidirectional convergence
            if self.bidirectional_convergence(syntax_tree, concept_graph):
                break
        
        return syntax_tree
```

### Constraint Propagation
```python
def propagate_constraints(self, syntax_node, concept_node):
    """
    Propagate grammatical and semantic constraints
    between syntax tree and concept graph
    """
    constraints = []
    
    # Grammatical constraints from syntax
    if syntax_node.requires_object():
        constraints.append({
            'type': 'grammatical',
            'requirement': 'object_concept',
            'target': concept_node
        })
    
    # Semantic constraints from concepts
    if concept_node.is_abstract():
        constraints.append({
            'type': 'semantic',
            'requirement': 'abstract_syntax',
            'target': syntax_node
        })
    
    # Propagate via messages
    self.send_constraint_messages(constraints)
```

## 4. Attention-Enhanced Message Passing

### Query-Driven Attention
```python
class AttentionMessagePassing:
    """
    Focus message passing on query-relevant regions
    """
    def compute_attention_weights(self, query_vector, subgraph):
        # Initial attention based on query similarity
        attention = self.initial_attention(query_vector, subgraph)
        
        # Propagate attention through messages
        for layer in range(self.num_layers):
            attention_messages = {}
            
            for node in subgraph.nodes:
                # Compute attention message
                message = self.compute_attention_message(
                    node,
                    attention[node],
                    query_vector
                )
                
                # Send to neighbors
                for neighbor in node.neighbors:
                    attention_messages[neighbor] = self.aggregate(
                        attention_messages.get(neighbor, 0),
                        message
                    )
            
            # Update attention weights
            attention = self.update_attention(attention, attention_messages)
        
        return attention
```

### Multi-Head Message Attention
```python
class MultiHeadMessageAttention:
    """
    Different attention heads for different aspects
    """
    def __init__(self, num_heads=8):
        self.heads = {
            'semantic': SemanticAttentionHead(),
            'syntactic': SyntacticAttentionHead(),
            'hierarchical': HierarchicalAttentionHead(),
            'temporal': TemporalAttentionHead(),
            # ... more heads
        }
    
    def compute_multi_head_messages(self, node, context):
        head_messages = {}
        
        for head_name, head in self.heads.items():
            head_messages[head_name] = head.compute_message(
                node, context
            )
        
        # Combine messages from all heads
        combined = self.combine_head_messages(head_messages)
        return combined
```

## 5. Error Correction via Messages

### Semantic Coherence Checking
```python
class ErrorCorrectionMessages:
    """
    Detect and correct semantic inconsistencies
    through message passing
    """
    def verify_and_correct(self, generated_text, source_graph):
        # Re-encode generated text
        generated_vector = encode(generated_text)
        generated_graph = text_to_graph(generated_text)
        
        # Detect discrepancies
        discrepancies = self.detect_discrepancies(
            source_graph,
            generated_graph
        )
        
        # Propagate correction messages
        correction_messages = {}
        for disc in discrepancies:
            message = self.create_correction_message(disc)
            affected_nodes = self.propagate_correction(
                message,
                generated_graph
            )
            correction_messages.update(affected_nodes)
        
        # Apply corrections
        corrected_text = self.apply_corrections(
            generated_text,
            correction_messages
        )
        
        return corrected_text
```

### Consistency Enforcement
```python
def enforce_consistency_via_messages(self, partial_generation):
    """
    Ensure generated text remains consistent
    with source knowledge
    """
    # Track generated concepts
    generated_concepts = extract_concepts(partial_generation)
    
    # Check against source graph
    for concept in generated_concepts:
        if not self.is_consistent(concept, self.source_graph):
            # Send inhibitory messages
            self.send_inhibition(concept)
            
            # Send excitatory messages to alternatives
            alternatives = self.find_consistent_alternatives(concept)
            for alt in alternatives:
                self.send_excitation(alt)
```

## 6. Efficiency Optimizations

### Sparse Message Passing
```python
class SparseMessagePassing:
    """
    Only propagate messages where needed
    """
    def selective_propagation(self, active_set, threshold=0.1):
        # Only propagate from significantly active nodes
        propagating_nodes = {
            node for node in active_set
            if node.activation > threshold
        }
        
        # Limited hop propagation
        messages = {}
        for node in propagating_nodes:
            local_messages = self.local_propagation(
                node,
                max_hops=2
            )
            messages.update(local_messages)
        
        return messages
```

### Message Caching
```python
class MessageCache:
    """
    Cache frequently used message patterns
    """
    def __init__(self):
        self.cache = {}
        self.access_counts = defaultdict(int)
    
    def get_or_compute_message(self, source, target, context):
        key = (source.id, target.id, hash(context))
        
        if key in self.cache:
            self.access_counts[key] += 1
            return self.cache[key]
        
        # Compute and cache
        message = self.compute_message(source, target, context)
        self.cache[key] = message
        
        # Evict least used if cache too large
        if len(self.cache) > self.max_size:
            self.evict_least_used()
        
        return message
```

## 7. Integration with geDIG Principles

### Message-Driven ΔGED Optimization
```python
def optimize_structure_via_messages(self, current_graph):
    """
    Use messages to guide structural simplification
    """
    # Identify redundant connections
    redundancy_messages = self.detect_redundancies(current_graph)
    
    # Propagate simplification suggestions
    for source, targets in redundancy_messages.items():
        simplification = self.compute_simplification(source, targets)
        self.propagate_simplification(simplification)
    
    # Apply structural changes
    simplified = self.apply_simplifications(current_graph)
    
    # Verify information preservation (ΔIG constraint)
    if self.information_preserved(current_graph, simplified):
        return simplified
    else:
        return self.restore_critical_edges(simplified)
```

### Message-Driven ΔIG Optimization
```python
def optimize_information_via_messages(self, concept_clusters):
    """
    Use messages to improve information organization
    """
    # Compute cluster cohesion via messages
    cohesion_messages = self.compute_cohesion_messages(concept_clusters)
    
    # Reorganize based on message patterns
    new_clusters = self.reorganize_clusters(
        concept_clusters,
        cohesion_messages
    )
    
    # Verify improved organization
    if self.entropy_reduced(new_clusters, concept_clusters):
        return new_clusters
    else:
        return concept_clusters
```

## 8. Future Directions

### Neural Message Passing
- Learn message functions from data
- Differentiable message aggregation
- End-to-end training

### Quantum-Inspired Messages
- Superposition of message states
- Entangled concept representations
- Quantum interference patterns

### Biological Realism
- Spike-timing dependent messages
- Synaptic plasticity simulation
- Neurotransmitter-like message types

## Conclusion

Message passing is not just an implementation detail but the core mechanism that makes the geDIG decoder work. It enables:
- Dynamic context-aware generation
- Bidirectional optimization
- Error correction
- Efficient computation
- Theoretical elegance

This approach unifies graph neural networks, natural language generation, and cognitive modeling in a single coherent framework.