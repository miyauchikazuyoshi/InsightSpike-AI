---
status: active
category: memory
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# Split Memory Decoding Strategy v2.0
*Updated with geDIG Generative Grammar and Concept Token insights*

## Overview
Split episodic memories are not just storage variants but represent **conceptual evolution stages**. With our new understanding of generative grammar decoding and concept tokens, we can create a more sophisticated decoding strategy.

## Paradigm Shift: From Selection to Evolution

### Old Approach
- Problem: Which split memory to choose?
- Solution: Complex selection mechanisms

### New Approach  
- Insight: Split memories represent **conceptual growth stages**
- Solution: Use geDIG to track and decode conceptual evolution

## Core Architecture

### 1. Conceptual Evolution Tracking
```python
class ConceptualEvolutionTracker:
    """
    Track how concepts evolve through memory splits
    """
    def __init__(self):
        self.evolution_graph = nx.DiGraph()
        self.concept_tokens = {}  # Maps evolution stages to tokens
        
    def track_split(self, original_concept, split_concepts):
        # Create evolution edge
        for split in split_concepts:
            self.evolution_graph.add_edge(
                original_concept,
                split,
                evolution_type=self.analyze_split_type(original_concept, split)
            )
        
        # Generate stage-specific tokens
        self.create_evolution_tokens(original_concept, split_concepts)
    
    def analyze_split_type(self, parent, child):
        """
        Determine the nature of conceptual evolution
        """
        delta_ged = compute_ged(parent.graph, child.graph)
        delta_ig = compute_ig(parent.vector, child.vector)
        
        if delta_ged < -1.0 and delta_ig > 0.5:
            return 'abstraction'  # Concrete → Abstract
        elif delta_ged > 1.0:
            return 'specialization'  # General → Specific
        else:
            return 'lateral'  # Alternative perspective
```

### 2. Evolution-Aware Concept Tokens
```python
class EvolutionTokenizer:
    """
    Create tokens that encode conceptual evolution stage
    """
    def create_evolution_token(self, concept, stage_info):
        # Token format: [CONCEPT_STAGE_NAME_LEVEL_ID]
        # e.g., [CONCEPT_BASIC_MULTIPLICATION_L1_42]
        #       [CONCEPT_ADVANCED_MULTIPLICATION_L3_42]
        
        token = f"[CONCEPT_{stage_info['level'].upper()}_{concept.name.upper()}_L{stage_info['depth']}_{concept.id}]"
        
        # Store bidirectional mapping
        self.token_to_stage[token] = {
            'concept': concept,
            'vector': concept.vector,
            'stage': stage_info,
            'prerequisites': self.get_prerequisites(concept)
        }
        
        return token
```

### 3. Grammar-Guided Decoding
```python
class EvolutionAwareGrammarDecoder:
    """
    Decode using conceptual evolution grammar rules
    """
    def __init__(self):
        self.grammar_rules = {
            'introduce_basic': S(
                "Let me explain", 
                NP("[CONCEPT_BASIC_*]"), 
                VP("in simple terms")
            ),
            'build_on_previous': S(
                "Building on", 
                NP("[CONCEPT_BASIC_*]"),
                VP("we can understand"),
                NP("[CONCEPT_ADVANCED_*]")
            ),
            'show_evolution': S(
                NP("[CONCEPT_BASIC_*]"),
                VP("evolves into"),
                NP("[CONCEPT_ADVANCED_*]"),
                PP("when", CONDITION)
            )
        }
    
    def decode_with_evolution(self, query_vector, evolution_graph):
        # Determine required conceptual depth
        required_depth = self.analyze_query_complexity(query_vector)
        
        # Find optimal path through evolution graph
        concept_path = self.find_optimal_evolution_path(
            evolution_graph,
            start_level='basic',
            target_depth=required_depth
        )
        
        # Generate explanation following the path
        return self.generate_progressive_explanation(concept_path)
```

### 4. Message Passing for Memory Integration
```python
class SplitMemoryMessagePassing:
    """
    Use message passing to integrate split memories dynamically
    """
    def integrate_split_memories(self, query, split_memories):
        # Initialize activation based on query relevance
        activations = {
            memory: self.compute_relevance(query, memory)
            for memory in split_memories
        }
        
        # Message passing between related memories
        for iteration in range(self.max_iterations):
            new_activations = {}
            
            for memory in split_memories:
                # Collect messages from evolutionarily related memories
                messages = []
                for related in self.get_related_memories(memory):
                    if self.are_compatible(memory, related):
                        msg = self.compute_compatibility_message(
                            memory, 
                            related,
                            activations[related]
                        )
                        messages.append(msg)
                
                # Update activation
                new_activations[memory] = self.update_activation(
                    activations[memory],
                    messages
                )
            
            activations = new_activations
            
            # Check convergence
            if self.converged(activations):
                break
        
        # Select and combine based on final activations
        return self.combine_by_activation(split_memories, activations)
```

### 5. Audience-Adaptive Evolution
```python
class AudienceAdaptiveDecoder:
    """
    Adjust conceptual depth based on audience understanding
    """
    def decode_for_audience(self, concept, audience_profile):
        # Map audience to appropriate evolution stage
        if audience_profile['expertise'] == 'novice':
            # Use basic stage tokens
            return self.use_evolution_stage(concept, 'basic')
            
        elif audience_profile['expertise'] == 'intermediate':
            # Bridge basic and advanced
            return self.create_bridging_explanation(
                concept, 
                from_stage='basic',
                to_stage='intermediate'
            )
            
        else:  # expert
            # Can handle full evolution
            return self.show_complete_evolution(concept)
    
    def create_bridging_explanation(self, concept, from_stage, to_stage):
        """
        Generate explanation that bridges understanding levels
        """
        bridge_grammar = S(
            "You already understand",
            NP(f"[CONCEPT_{from_stage.upper()}_{concept}_*]"),
            VP("Now let's see how"),
            NP("this"),
            VP("extends to"),
            NP(f"[CONCEPT_{to_stage.upper()}_{concept}_*]")
        )
        
        return self.realize_grammar(bridge_grammar)
```

### 6. Temporal Coherence
```python
class TemporalCoherenceManager:
    """
    Ensure coherent progression through conceptual stages
    """
    def __init__(self):
        self.conversation_history = []
        self.introduced_concepts = {}
        
    def select_appropriate_stage(self, concept, context):
        # Check what stages have been introduced
        if concept.id not in self.introduced_concepts:
            # Start with basic
            return concept.get_stage('basic')
            
        # Find highest introduced stage
        introduced = self.introduced_concepts[concept.id]
        current_max = max(introduced, key=lambda s: s.depth)
        
        # Can we go one level deeper?
        if self.ready_for_next_level(context, current_max):
            return concept.get_next_stage(current_max)
        else:
            return current_max
    
    def ready_for_next_level(self, context, current_stage):
        """
        Check if audience is ready for more advanced explanation
        """
        # Analyze conversation for understanding signals
        understanding_signals = {
            'questions_answered': context.successful_answers,
            'follow_up_depth': context.question_complexity,
            'time_on_concept': context.engagement_time,
            'confusion_signals': context.confusion_count
        }
        
        return self.compute_readiness(understanding_signals) > 0.7
```

## Implementation Strategy

### Phase 1: Concept Evolution Tokens (Immediate)
```python
# Implement special tokens for evolution stages
tokenizer.add_special_tokens({
    'evolution_tokens': [
        '[CONCEPT_BASIC_*]',
        '[CONCEPT_INTERMEDIATE_*]',
        '[CONCEPT_ADVANCED_*]',
        '[CONCEPT_EXPERT_*]'
    ]
})
```

### Phase 2: Evolution Graph Construction (1 month)
- Build evolution graphs from split memories
- Track conceptual growth paths
- Create navigation algorithms

### Phase 3: Grammar Integration (2 months)
- Define grammar rules for each evolution type
- Implement progressive explanation generation
- Add bridging templates

### Phase 4: Message Passing Integration (3 months)
- Implement compatibility checking
- Dynamic activation propagation
- Convergence optimization

## Advantages Over Previous Approach

1. **Conceptual Coherence**: Evolution stages maintain logical progression
2. **Natural Pedagogy**: Mirrors human teaching patterns
3. **Efficient Encoding**: Concept tokens compress evolution history
4. **Dynamic Adaptation**: Message passing enables real-time adjustment
5. **Grammar Constraints**: Ensures pedagogically sound explanations

## Example: Mathematics Concept Evolution

```python
# Multiplication concept evolution
evolution_path = {
    'stage_1': {
        'token': '[CONCEPT_BASIC_MULTIPLICATION_L1_42]',
        'explanation': "Adding the same number multiple times"
    },
    'stage_2': {
        'token': '[CONCEPT_INTERMEDIATE_MULTIPLICATION_L2_42]',
        'explanation': "Scaling one quantity by another"
    },
    'stage_3': {
        'token': '[CONCEPT_ADVANCED_MULTIPLICATION_L3_42]',
        'explanation': "Binary operation on field elements"
    }
}

# Query: "Explain 3.5 × 2"
# System selects stage_2 (scaling) as optimal for decimal multiplication
```

## Metrics

### Evolution-Aware Metrics
- **Conceptual Coherence**: Logical progression through stages
- **Pedagogical Effectiveness**: Learning outcome improvement
- **Adaptation Quality**: Match between explanation and understanding
- **Evolution Coverage**: Completeness of conceptual journey

## Future Extensions

1. **Cross-Concept Evolution**: Track how different concepts co-evolve
2. **Cultural Evolution Patterns**: Different pedagogical traditions
3. **Personalized Evolution Paths**: Individual learning trajectories
4. **Meta-Evolution**: How evolution patterns themselves evolve

## Conclusion

By treating split memories as conceptual evolution stages rather than alternatives to choose from, we create a more natural and powerful decoding strategy. The combination of:
- Evolution-aware concept tokens
- Generative grammar for progression
- Message passing for integration
- Audience adaptation

Creates a system that doesn't just decode memories but **guides understanding through natural conceptual development**.