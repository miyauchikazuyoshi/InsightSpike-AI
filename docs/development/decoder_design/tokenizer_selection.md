---
status: active
category: meta
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# Tokenizer Selection for Vector-Text Alignment

*Created: 2025-07-24*

## Core Requirement
Need tokenizer that aligns well with SentenceTransformer's internal representation for seamless vector↔text conversion.

## SentenceTransformer's Tokenizer

### Current Setup (all-MiniLM-L6-v2)
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
# Internally uses:
# - Tokenizer: BertTokenizerFast
# - Model: BERT-based
# - Vocab size: 30,522 tokens
```

### Key Properties
- WordPiece tokenization
- Handles subwords well
- [CLS] and [SEP] tokens
- Max sequence length: 256 tokens

## Tokenizer Options

### 1. Direct SentenceTransformer Tokenizer (RECOMMENDED)
```python
from sentence_transformers import SentenceTransformer

class DirectSTTokenizer:
    """
    Use SentenceTransformer's internal tokenizer directly
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.tokenizer = self.model.tokenizer  # Direct access
        
    def tokenize(self, text):
        return self.tokenizer(text, return_tensors='pt')
    
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)
    
    def get_vocab_embeddings(self):
        # Access word embeddings directly
        return self.model[0].auto_model.embeddings.word_embeddings
```

**Pros:**
- Perfect alignment with vector space
- No conversion needed
- Access to subword embeddings

**Cons:**
- Tied to specific model
- Limited customization

### 2. Transformers AutoTokenizer
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    'sentence-transformers/all-MiniLM-L6-v2'
)

# Exact same tokenizer as SentenceTransformer uses
```

**Pros:**
- More flexible
- Can swap models easily
- Full transformers ecosystem

**Cons:**
- Need to ensure alignment

### 3. SentencePiece (Alternative)
```python
import sentencepiece as spm

# Train custom tokenizer on your domain
spm.SentencePieceTrainer.train(
    input='domain_corpus.txt',
    model_prefix='gedig',
    vocab_size=32000,
    model_type='unigram'  # or 'bpe'
)
```

**Pros:**
- Language agnostic
- Can optimize for specific domain
- Handles rare words better

**Cons:**
- Need alignment layer to SentenceTransformer
- Additional training required

### 4. Hybrid Approach (BEST FOR PRODUCTION)
```python
class HybridTokenizer:
    """
    Combines SentenceTransformer compatibility with
    custom vocabulary for domain terms
    """
    def __init__(self, base_model='all-MiniLM-L6-v2'):
        # Base tokenizer from ST
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            f'sentence-transformers/{base_model}'
        )
        
        # Custom vocabulary for domain terms
        self.custom_vocab = {}
        self.load_domain_vocabulary()
        
    def tokenize_with_alignment(self, text):
        # First check custom vocabulary
        custom_tokens = self.extract_custom_tokens(text)
        
        # Replace with placeholders
        text_processed = self.replace_custom_tokens(text, custom_tokens)
        
        # Tokenize with base
        base_tokens = self.base_tokenizer(text_processed)
        
        # Align and merge
        return self.align_tokens(base_tokens, custom_tokens)
    
    def extract_custom_tokens(self, text):
        """
        Extract domain-specific terms that should be 
        treated as single tokens
        """
        custom_tokens = []
        for term in self.custom_vocab:
            if term in text:
                custom_tokens.append({
                    'term': term,
                    'vector': self.custom_vocab[term],
                    'position': text.index(term)
                })
        return custom_tokens
```

## Vector-Token Alignment Strategy

### 1. Token-Level Embeddings
```python
class TokenVectorAlignment:
    """
    Maintain alignment between tokens and vectors
    """
    def __init__(self, sentence_transformer):
        self.st_model = sentence_transformer
        self.token_embeddings = self.extract_token_embeddings()
        
    def extract_token_embeddings(self):
        """
        Get individual token embeddings from ST model
        """
        # Access BERT's word embeddings
        word_embeddings = self.st_model[0].auto_model.embeddings.word_embeddings
        return word_embeddings.weight.data  # [vocab_size, hidden_dim]
    
    def token_to_vector(self, token_id):
        """
        Convert token ID to its embedding vector
        """
        return self.token_embeddings[token_id]
    
    def nearest_token(self, vector):
        """
        Find nearest token to a given vector
        """
        similarities = torch.cosine_similarity(
            vector.unsqueeze(0),
            self.token_embeddings
        )
        return torch.argmax(similarities)
```

### 2. Subword Handling
```python
def handle_subwords(self, word):
    """
    Handle WordPiece tokenization for OOV words
    """
    # Tokenize into subwords
    tokens = self.tokenizer.tokenize(word)
    
    if len(tokens) == 1:
        # Single token - direct mapping
        return tokens[0]
    else:
        # Multiple subwords - need aggregation
        subword_vectors = [
            self.token_to_vector(self.tokenizer.convert_tokens_to_ids(t))
            for t in tokens
        ]
        
        # Aggregate strategy (mean, first, last, etc.)
        return self.aggregate_subword_vectors(subword_vectors)
```

## Implementation Recommendations

### 1. For Research/Prototype
```python
# Use direct SentenceTransformer tokenizer
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = model.tokenizer

# Direct access to everything needed
```

### 2. For Production
```python
# Hybrid approach with caching
class ProductionTokenizer:
    def __init__(self):
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        self.vector_cache = {}
        self.custom_terms = self.load_domain_terms()
        
    @lru_cache(maxsize=10000)
    def tokenize_and_vectorize(self, text):
        tokens = self.tokenize_with_custom_handling(text)
        vectors = self.tokens_to_vectors(tokens)
        return tokens, vectors
```

### 3. For Multilingual
```python
# Use multilingual SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
tokenizer = model.tokenizer  # Supports 50+ languages
```

## Special Considerations for geDIG

### 1. Concept Tokens
```python
class ConceptTokenizer:
    """
    Special handling for abstract concepts
    """
    def __init__(self):
        self.concept_registry = {}
        
    def register_concept(self, concept_name, vector):
        """
        Register new concepts discovered by geDIG
        """
        # Create special token
        special_token = f"[CONCEPT_{concept_name.upper()}]"
        
        # Store mapping
        self.concept_registry[special_token] = {
            'vector': vector,
            'name': concept_name,
            'discovered_at': timestamp()
        }
        
        # Add to tokenizer vocabulary
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': [special_token]
        })
```

### 2. Hierarchical Tokenization
```python
def hierarchical_tokenize(self, text, level='word'):
    """
    Tokenize at different granularities
    """
    if level == 'char':
        return list(text)
    elif level == 'subword':
        return self.tokenizer.tokenize(text)
    elif level == 'word':
        # Preserve whole words when possible
        words = text.split()
        return self.preserve_word_tokenize(words)
    elif level == 'phrase':
        # Use grammatical phrases
        return self.phrase_tokenize(text)
```

## Integration Example

```python
class GeDIGDecoder:
    def __init__(self):
        # Use same model as encoder
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = self.st_model.tokenizer
        
        # Alignment layer
        self.token_vectors = self.build_token_vector_mapping()
        
    def vector_to_tokens(self, vector):
        """
        Convert concept vector to most likely tokens
        """
        # Method 1: Nearest neighbor in token space
        token_similarities = torch.cosine_similarity(
            vector.unsqueeze(0),
            self.token_vectors
        )
        
        # Get top-k tokens
        top_k = 10
        values, indices = torch.topk(token_similarities, top_k)
        
        # Convert to tokens
        candidate_tokens = [
            self.tokenizer.convert_ids_to_tokens(idx.item())
            for idx in indices
        ]
        
        return candidate_tokens, values
```

## Recommendation

**For geDIG decoder, use the hybrid approach:**

1. **Base**: SentenceTransformer's tokenizer for compatibility
2. **Extensions**: Custom vocabulary for discovered concepts
3. **Alignment**: Direct access to token embeddings
4. **Flexibility**: Can swap ST models without changing tokenizer interface

This ensures perfect alignment with your existing SentenceTransformer setup while allowing flexibility for the decoder's unique needs.