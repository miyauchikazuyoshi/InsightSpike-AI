---
status: active
category: meta
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# CLI Command Naming Rationale

## Why `query` and `embed`?

### `query` (instead of `ask`)

**Industry Standard Terms:**
- **Database Systems**: SQL uses `QUERY` to retrieve data
- **Search Engines**: Elasticsearch, Solr use "query" for searches
- **Vector Databases**: Pinecone, Weaviate, Qdrant all use "query" 
- **RAG Systems**: LangChain, LlamaIndex use "query" for retrieval

**Technical Accuracy:**
- `query`: Formal request to a data system
- `ask`: Informal, conversational
- InsightSpike is a knowledge retrieval system, not a chatbot

### `embed` (instead of `learn`)

**Industry Standard Terms:**
- **ML Frameworks**: TensorFlow, PyTorch use "embedding" for vectorization
- **NLP Libraries**: Sentence-Transformers, OpenAI all use "embed"
- **Vector DBs**: All vector databases use "embed" or "embedding"

**Technical Accuracy:**
- `embed`: Convert text to vector representation
- `learn`: Ambiguous - could mean training, fine-tuning, or memorizing
- InsightSpike converts documents to embeddings, not learning in ML sense

## Backward Compatibility

To ensure smooth transition:
- `spike ask` → redirects to `spike query`
- `spike learn` → redirects to `spike embed`
- All existing scripts continue to work

## Aliases

- `q` → `query` (natural abbreviation)
- `e` → `embed` (first letter)
- `l` → `embed` (legacy support from `learn`)
- `c` → `chat` (unchanged)

This naming aligns InsightSpike with industry standards while maintaining user convenience.