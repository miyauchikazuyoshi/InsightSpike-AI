---
status: active
category: meta
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# Concept Nominalization and Special Token Generation

*Created: 2025-07-24*

## Core Idea
When geDIG discovers new concepts through insight detection, we create special tokens that represent these abstract concepts, enabling direct vector↔token mapping.

## 1. Concept Discovery to Token Pipeline

### Discovery Phase
```python
class ConceptNominalizer:
    """
    Convert discovered concept vectors into named tokens
    """
    def __init__(self, tokenizer, vector_space):
        self.tokenizer = tokenizer
        self.vector_space = vector_space
        self.concept_registry = {}
        self.next_concept_id = 0
        
    def discover_and_nominalize(self, concept_vector, context):
        """
        Main pipeline: vector → name → token
        """
        # Step 1: Analyze concept properties
        properties = self.analyze_concept(concept_vector)
        
        # Step 2: Generate appropriate name
        concept_name = self.generate_concept_name(
            properties, 
            context
        )
        
        # Step 3: Create special token
        special_token = self.create_special_token(
            concept_name,
            concept_vector
        )
        
        # Step 4: Register in tokenizer
        self.register_token(special_token, concept_vector)
        
        return special_token
```

### Concept Analysis
```python
def analyze_concept(self, concept_vector):
    """
    Extract properties from concept vector
    """
    properties = {
        'abstractness': self.compute_abstractness(concept_vector),
        'complexity': self.compute_complexity(concept_vector),
        'nearest_known': self.find_nearest_concepts(concept_vector, k=5),
        'cluster_id': self.identify_cluster(concept_vector),
        'dimensionality': self.analyze_dimensions(concept_vector)
    }
    
    # Determine concept type
    if properties['abstractness'] > 0.8:
        properties['type'] = 'abstract'
    elif self.is_composite(concept_vector):
        properties['type'] = 'composite'
    elif self.is_relational(concept_vector):
        properties['type'] = 'relational'
    else:
        properties['type'] = 'concrete'
    
    return properties
```

## 2. Naming Strategies

### Rule-Based Naming
```python
class RuleBasedNaming:
    """
    Generate names based on concept properties
    """
    def generate_name(self, properties):
        if properties['type'] == 'composite':
            # Combine component names
            components = properties['components']
            return self.create_compound_name(components)
            
        elif properties['type'] == 'abstract':
            # Use Greek/Latin roots
            return self.create_abstract_name(properties)
            
        elif properties['type'] == 'relational':
            # Use relationship descriptors
            return self.create_relational_name(properties)
    
    def create_compound_name(self, components):
        """
        E.g., "speed" + "change" → "acceleration"
        """
        # Check if standard compound exists
        compound = f"{components[0]}_{components[1]}"
        
        if self.exists_in_dictionary(compound):
            return self.dictionary_lookup(compound)
        else:
            # Create new compound
            return f"{components[0]}{components[1].capitalize()}"
    
    def create_abstract_name(self, properties):
        """
        Generate names for highly abstract concepts
        """
        prefixes = {
            'meta': 'beyond',
            'hyper': 'above',
            'proto': 'first',
            'neo': 'new'
        }
        
        roots = {
            'morph': 'form',
            'gnosis': 'knowledge',
            'logos': 'logic',
            'noesis': 'understanding'
        }
        
        # Select based on properties
        prefix = self.select_prefix(properties)
        root = self.select_root(properties)
        
        return f"{prefix}{root}"
```

### Neural Naming
```python
class NeuralConceptNamer:
    """
    Use a small neural model to generate concept names
    """
    def __init__(self, naming_model):
        self.model = naming_model  # Trained on concept→name pairs
        
    def generate_name(self, concept_vector, context):
        # Encode context
        context_encoding = self.encode_context(context)
        
        # Generate name
        name_logits = self.model(
            concept_vector, 
            context_encoding
        )
        
        # Decode to text
        name = self.decode_name(name_logits)
        
        # Validate uniqueness
        if self.name_exists(name):
            name = self.make_unique(name)
            
        return name
```

## 3. Special Token Creation

### Token Format
```python
class SpecialTokenFormat:
    """
    Standardized format for concept tokens
    """
    @staticmethod
    def create_token(concept_name, concept_id, properties):
        # Format: [CONCEPT_TYPE_NAME_ID]
        token_parts = [
            'CONCEPT',
            properties['type'].upper(),
            concept_name.upper(),
            str(concept_id)
        ]
        
        return f"[{'_'.join(token_parts)}]"
    
    # Examples:
    # [CONCEPT_ABSTRACT_METALOGIC_42]
    # [CONCEPT_COMPOSITE_SPEEDCHANGE_13]
    # [CONCEPT_RELATIONAL_CAUSAL_7]
```

### Token Registration
```python
class TokenRegistry:
    """
    Manage special tokens in tokenizer
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.token_to_vector = {}
        self.vector_to_token = {}
        self.token_metadata = {}
        
    def register_concept_token(self, token, vector, metadata):
        """
        Add new concept token to tokenizer
        """
        # Add to tokenizer vocabulary
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': [token]
        })
        
        # Store mappings
        self.token_to_vector[token] = vector
        self.vector_to_token[vector.tobytes()] = token
        
        # Store metadata
        self.token_metadata[token] = {
            'created_at': timestamp(),
            'discovery_context': metadata['context'],
            'properties': metadata['properties'],
            'usage_count': 0
        }
        
        # Update model embeddings if needed
        self.update_embeddings(token, vector)
```

## 4. Dynamic Vocabulary Management

### Concept Lifecycle
```python
class ConceptLifecycle:
    """
    Manage concept tokens over time
    """
    def __init__(self):
        self.active_concepts = {}
        self.archived_concepts = {}
        self.concept_evolution = defaultdict(list)
        
    def track_concept_usage(self, token):
        """
        Monitor how concepts are used
        """
        self.active_concepts[token]['usage_count'] += 1
        self.active_concepts[token]['last_used'] = timestamp()
        
    def evolve_concept(self, old_token, new_vector, reason):
        """
        Handle concept evolution/refinement
        """
        # Create new version
        new_token = self.create_evolved_token(old_token)
        
        # Track evolution
        self.concept_evolution[old_token].append({
            'new_token': new_token,
            'new_vector': new_vector,
            'reason': reason,
            'timestamp': timestamp()
        })
        
        # Deprecate old token gradually
        self.deprecate_token(old_token)
        
        return new_token
    
    def garbage_collect(self):
        """
        Remove unused concepts
        """
        for token, data in list(self.active_concepts.items()):
            if data['usage_count'] < self.usage_threshold:
                if self.time_since_last_use(token) > self.expiry_time:
                    self.archive_concept(token)
```

## 5. Integration with Decoder

### Decoding with Concept Tokens
```python
class ConceptAwareDecoder:
    """
    Decoder that leverages special concept tokens
    """
    def decode(self, concept_vector):
        # Check if direct concept token exists
        if token := self.find_concept_token(concept_vector):
            return self.expand_concept_token(token)
        
        # Otherwise, find nearest concepts
        nearest_tokens = self.find_nearest_concept_tokens(
            concept_vector, 
            k=3
        )
        
        # Compose explanation
        return self.compose_from_concepts(nearest_tokens)
    
    def expand_concept_token(self, token):
        """
        Convert concept token to natural language
        """
        metadata = self.token_metadata[token]
        
        # Template-based expansion
        if metadata['type'] == 'composite':
            components = metadata['components']
            return f"a concept combining {components[0]} and {components[1]}"
            
        elif metadata['type'] == 'abstract':
            nearest = metadata['nearest_known']
            return f"an abstract concept related to {', '.join(nearest)}"
```

### Example Pipeline
```python
# Discovery
vector = compute_insight_vector(graph_state)
properties = analyze_concept(vector)

# Nominalization
if properties['type'] == 'composite' and components == ['speed', 'change']:
    name = 'acceleration'  # Or 'velocityDelta' if unnamed
    
# Tokenization
token = '[CONCEPT_COMPOSITE_ACCELERATION_42]'

# Registration
tokenizer.add_special_tokens({'additional_special_tokens': [token]})
token_embeddings[token] = vector

# Later usage
decoded = "The [CONCEPT_COMPOSITE_ACCELERATION_42] represents how quickly velocity changes"
# Can be expanded to: "The acceleration represents how quickly velocity changes"
```

## 6. Advantages

### For Encoding
- New concepts get unique tokens
- Direct vector mapping
- No ambiguity

### For Decoding
- Can reference abstract concepts precisely
- Tokens can be expanded to explanations
- Maintains semantic consistency

### For Learning
- Concepts persist across sessions
- Can track concept evolution
- Build concept hierarchies

## 7. Implementation Considerations

### Storage
```python
# Persistent concept storage
class ConceptStorage:
    def save_concepts(self, filepath):
        concepts = {
            'tokens': self.token_to_vector,
            'metadata': self.token_metadata,
            'evolution': self.concept_evolution
        }
        torch.save(concepts, filepath)
    
    def load_concepts(self, filepath):
        concepts = torch.load(filepath)
        self.restore_tokens(concepts)
```

### Efficiency
```python
# Efficient lookup
class ConceptIndex:
    def __init__(self):
        # Use FAISS for vector similarity
        self.vector_index = faiss.IndexFlatIP(768)
        self.id_to_token = {}
        
    def add_concept(self, token, vector):
        concept_id = self.vector_index.ntotal
        self.vector_index.add(vector.reshape(1, -1))
        self.id_to_token[concept_id] = token
```

## Conclusion

Special tokens for discovered concepts create a bridge between:
- Abstract vector representations
- Concrete linguistic tokens
- Human-understandable names

This enables the decoder to reference and explain even completely novel concepts discovered through geDIG's insight detection mechanism.