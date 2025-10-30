---
status: active
category: llm
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# Bidirectional Conceptual LLM Specification

## Overview
A specialized language model designed for bidirectional transformation between concept names, descriptions, and vector representations, inspired by human conceptual understanding and self-attention mechanisms.

## Core Concept: Triangular Bidirectionality

```
     Name (名詞)
       /    \
      /      \
     /        \
Description ← → Vector
(説明)        (ベクトル)
```

Each vertex can be derived from the other two, creating a self-reinforcing understanding system.

## Architecture Design

### 1. Three-Modal Transformer Architecture

```python
class BidirectionalConceptualLLM(nn.Module):
    """
    A transformer-based model handling three modalities:
    - Name: Single token or short phrase
    - Description: Natural language explanation
    - Vector: Dense semantic representation
    """
    
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model  # 768
        self.vocab_size = config.vocab_size  # 30000 (domain-specific)
        self.vector_dim = config.vector_dim  # 768
        
        # Encoders for each modality
        self.name_encoder = NameEncoder(self.vocab_size, self.d_model)
        self.desc_encoder = DescriptionEncoder(self.vocab_size, self.d_model)
        self.vector_encoder = VectorEncoder(self.vector_dim, self.d_model)
        
        # Shared transformer backbone
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=12,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=3072,
            dropout=0.1
        )
        
        # Output heads
        self.name_head = NameGenerationHead(self.d_model, self.vocab_size)
        self.desc_head = DescriptionGenerationHead(self.d_model, self.vocab_size)
        self.vector_head = VectorProjectionHead(self.d_model, self.vector_dim)
```

### 2. Bidirectional Self-Attention Mechanism

```python
class ConceptualSelfAttention(nn.Module):
    """
    Implements cross-modal attention where each representation
    attends to the others to create a unified understanding.
    """
    
    def forward(self, name_emb, desc_emb, vector_emb):
        # Each modality queries the others
        name_attended = self.cross_attention(
            query=name_emb,
            key=torch.cat([desc_emb, vector_emb], dim=1),
            value=torch.cat([desc_emb, vector_emb], dim=1)
        )
        
        desc_attended = self.cross_attention(
            query=desc_emb,
            key=torch.cat([name_emb, vector_emb], dim=1),
            value=torch.cat([name_emb, vector_emb], dim=1)
        )
        
        vector_attended = self.cross_attention(
            query=vector_emb,
            key=torch.cat([name_emb, desc_emb], dim=1),
            value=torch.cat([name_emb, desc_emb], dim=1)
        )
        
        # Fusion layer
        return self.fuse_representations(
            name_attended, desc_attended, vector_attended
        )
```

### 3. Training Objectives

```python
class ConceptualLoss(nn.Module):
    """
    Multi-task loss for bidirectional concept learning
    """
    
    def forward(self, predictions, targets):
        # Direct prediction losses
        loss_name = self.name_prediction_loss(
            predictions.name, targets.name
        )
        loss_desc = self.description_generation_loss(
            predictions.description, targets.description
        )
        loss_vector = self.vector_reconstruction_loss(
            predictions.vector, targets.vector
        )
        
        # Cycle consistency losses
        loss_name_cycle = self.cycle_consistency_loss(
            targets.name,
            encode_then_decode(targets.name, "name→vector→name")
        )
        loss_desc_cycle = self.cycle_consistency_loss(
            targets.description,
            encode_then_decode(targets.description, "desc→vector→desc")
        )
        
        # Semantic consistency loss
        loss_semantic = self.semantic_consistency_loss(
            encode(targets.name),
            encode(targets.description),
            targets.vector
        )
        
        return (
            loss_name + loss_desc + loss_vector +
            0.5 * (loss_name_cycle + loss_desc_cycle) +
            0.3 * loss_semantic
        )
```

## Implementation Specification

### 1. Generative Grammar Integration

```python
class GrammarGuidedDecoder:
    """
    Uses generative grammar rules to structure output
    """
    
    def __init__(self):
        self.grammar_rules = {
            "definition": S(NP("concept"), VP("is", NP("description"))),
            "relation": S(NP("A"), VP("relates to", NP("B"), PP("through", NP("C")))),
            "property": S(NP("concept"), VP("has property", NP("attribute"))),
            "composition": S(NP("whole"), VP("consists of", NP_LIST("parts")))
        }
    
    def decode_with_grammar(self, vector, grammar_type):
        # Select appropriate grammar template
        template = self.grammar_rules[grammar_type]
        
        # Fill slots based on vector properties
        filled_template = self.fill_template(template, vector)
        
        # Generate natural language
        return self.realize(filled_template)
```

### 2. Concept Nominalization Module

```python
class ConceptNominalizer:
    """
    Converts vector representations to appropriate nouns/noun phrases
    """
    
    def nominalize(self, vector, context=None):
        # Analyze vector properties
        properties = self.analyze_vector(vector)
        
        if properties.is_atomic:
            # Simple concept → single word
            return self.generate_simple_noun(vector)
        elif properties.is_composite:
            # Composite concept → compound noun
            components = self.decompose_vector(vector)
            return self.generate_compound_noun(components)
        elif properties.is_abstract:
            # Abstract concept → derived noun
            base = self.find_conceptual_base(vector)
            return self.generate_abstract_noun(base, properties)
```

### 3. Semantic Consistency Validator

```python
class SemanticConsistencyValidator:
    """
    Ensures bidirectional transformations preserve meaning
    """
    
    def validate_consistency(self, original, transformed, modality_path):
        # Measure semantic similarity
        similarity = self.compute_similarity(original, transformed)
        
        if similarity < self.threshold:
            # Attempt refinement
            refined = self.refine_transformation(
                original, transformed, modality_path
            )
            return self.validate_consistency(original, refined, modality_path)
        
        return transformed
```

## Training Data Specification

### 1. Data Sources
- **Academic**: Definitions from textbooks, encyclopedias
- **Technical**: API documentation, technical glossaries
- **Conceptual**: Philosophy texts, scientific papers
- **Linguistic**: WordNet, ConceptNet, FrameNet

### 2. Data Format
```json
{
    "concept_id": "derivative_001",
    "name": "derivative",
    "descriptions": [
        "The instantaneous rate of change of a function",
        "The slope of the tangent line at a point",
        "A measure of how a function changes as its input changes"
    ],
    "vector": [0.234, -0.567, ...],  // 768-dimensional
    "contexts": {
        "mathematics": "fundamental concept in calculus",
        "physics": "velocity is the derivative of position",
        "economics": "marginal cost is a derivative"
    },
    "related_concepts": ["integral", "limit", "rate_of_change"],
    "hierarchical_relations": {
        "hypernym": "mathematical_operation",
        "hyponyms": ["partial_derivative", "directional_derivative"]
    }
}
```

### 3. Synthetic Data Generation
```python
class ConceptualDataAugmentation:
    """
    Generate training data through systematic transformations
    """
    
    def augment_concept(self, concept):
        augmented = []
        
        # Paraphrase descriptions
        for desc in concept.descriptions:
            paraphrases = self.generate_paraphrases(desc)
            augmented.extend(paraphrases)
        
        # Create contextual variations
        for context in concept.contexts:
            contextual_desc = self.contextualize(
                concept.description, context
            )
            augmented.append(contextual_desc)
        
        # Generate compositional examples
        for related in concept.related_concepts:
            composite = self.create_composite_concept(
                concept, related
            )
            augmented.append(composite)
        
        return augmented
```

## Experiment Plan

### Phase 1: Proof of Concept (1-2 months)
1. **Small-scale prototype**
   - 1000 concepts from mathematics/physics
   - Basic transformer architecture
   - Focus on name↔vector bidirectionality

2. **Metrics**
   - Reconstruction accuracy
   - Semantic similarity preservation
   - Cycle consistency

### Phase 2: Grammar Integration (2-3 months)
1. **Implement generative grammar decoder**
   - Rule-based templates
   - Grammar-guided generation
   - Syntactic diversity

2. **Evaluation**
   - Grammatical correctness
   - Semantic accuracy
   - Human evaluation of naturalness

### Phase 3: Full System (3-4 months)
1. **Complete three-way bidirectionality**
   - Name↔Description↔Vector
   - All transformation paths
   - Consistency enforcement

2. **Scale-up**
   - 10,000+ concepts
   - Multiple domains
   - Cross-domain generalization

### Phase 4: Integration with InsightSpike (1-2 months)
1. **Replace current decoder**
   - Integrate with existing pipeline
   - Performance optimization
   - A/B testing

2. **Enhancement**
   - Fine-tune on InsightSpike discoveries
   - Adapt to specific use cases
   - Continuous learning

## Success Criteria

### Technical Metrics
- **Reconstruction F1**: >0.85 for all modality pairs
- **Cycle consistency**: >0.90 similarity after round-trip
- **Generation fluency**: >4.0/5.0 human rating
- **Latency**: <50ms per transformation

### Qualitative Goals
- Natural, grammatical descriptions
- Accurate concept naming
- Preservation of semantic nuances
- Handling of novel concepts

## Model Size and Resources

### Target Specifications
- **Parameters**: ~125M (smaller than GPT-2 medium)
- **Memory**: <1GB model size
- **Training time**: ~2 weeks on 4x V100
- **Inference**: Real-time on single GPU

### Optimization Strategies
- Knowledge distillation from larger models
- Efficient attention mechanisms
- Quantization for deployment
- Caching for common concepts

## Risk Mitigation

### Technical Risks
1. **Semantic drift**: Regular consistency validation
2. **Grammar rigidity**: Hybrid rule-neural approach
3. **Novel concepts**: Compositional generation fallback
4. **Training data bias**: Diverse data sources

### Implementation Risks
1. **Complexity**: Phased implementation
2. **Integration**: Modular design
3. **Performance**: Early optimization
4. **Maintenance**: Comprehensive testing

## Conclusion

This bidirectional conceptual LLM represents a fundamental advancement in how AI systems understand and express concepts. By modeling the human-like triangular relationship between names, descriptions, and semantic vectors, we can create a system that truly "understands" concepts rather than merely manipulating symbols.

The feasibility is high due to:
1. Mature transformer technology
2. Clear, focused objective
3. Available training data
4. Reasonable computational requirements
5. Direct applicability to InsightSpike

This is not just a decoder—it's a complete conceptual understanding system.