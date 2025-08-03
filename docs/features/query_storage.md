# Query Storage Feature

## Overview

The Query Storage feature enables InsightSpike-AI to persist all user queries along with their processing metadata, responses, and relationships to generated insights. This allows for:

- Historical analysis of query patterns
- Learning from past interactions
- Tracking spike generation success rates
- Analyzing system performance over time

## Architecture

### Storage Layer
Queries are stored in the DataStore layer with the following structure:

```python
{
    "id": "query_<timestamp>_<uuid>",
    "text": "The original query text",
    "vec": np.ndarray,  # Query embedding vector
    "has_spike": bool,  # Whether the query generated an insight spike
    "spike_episode_id": str,  # ID of generated episode if spike occurred
    "response": str,  # The response given to the user
    "timestamp": float,
    "metadata": {
        "processing_time": float,
        "llm_provider": str,
        "total_cycles": int,
        "reasoning_quality": float,
        "retrieved_doc_count": int,
        # Additional metadata depending on processor
    }
}
```

### Graph Integration
When using CachedMemoryManager, queries are also added as nodes to the knowledge graph with edges:

- **query_spike**: Connects query to the episode it generated (weight=1.0)
- **query_retrieval**: Connects query to episodes retrieved for context (weight=similarity)

## Usage

### Automatic Query Saving

Queries are automatically saved when using MainAgent or AdaptiveProcessor with a DataStore:

```python
from insightspike.implementations.agents.main_agent import MainAgent
from insightspike.implementations.datastore.filesystem_store import FileSystemDataStore

# Create agent with datastore
datastore = FileSystemDataStore("data")
agent = MainAgent(config=config, datastore=datastore)

# Process question - query is automatically saved
result = agent.process_question("What is insight?")
print(f"Query saved with ID: {result.query_id}")
```

### Query Retrieval

```python
# Get recent queries
from insightspike.implementations.layers.cached_memory_manager import CachedMemoryManager

manager = CachedMemoryManager(datastore)

# Get all recent queries
recent_queries = manager.get_recent_queries(limit=100)

# Get only queries that generated spikes
spike_queries = manager.get_recent_queries(has_spike=True)

# Get queries that didn't generate spikes
no_spike_queries = manager.get_recent_queries(has_spike=False)
```

### Query Analysis

```python
# Get query statistics
stats = manager.get_query_statistics()

print(f"Total queries: {stats['total_queries']}")
print(f"Spike rate: {stats['spike_rate'] * 100:.1f}%")
print(f"Average processing time: {stats['avg_processing_time']:.2f}s")
print(f"LLM providers used: {stats['llm_providers']}")
```

### Direct DataStore Access

```python
# Load queries directly from datastore
queries = datastore.load_queries(
    namespace="queries",
    has_spike=True,  # Optional filter
    limit=50  # Optional limit
)

# Save custom queries
custom_query = {
    "id": f"query_{int(time.time() * 1000)}",
    "text": "Custom query",
    "vec": embedding_vector,
    "has_spike": False,
    "response": "Response",
    "metadata": {"custom": "data"}
}
datastore.save_queries([custom_query])
```

## Analysis Examples

### Success Rate Analysis
```python
# Calculate spike success rate over time
all_queries = datastore.load_queries()
spike_count = sum(1 for q in all_queries if q["has_spike"])
success_rate = spike_count / len(all_queries) if all_queries else 0
```

### Performance Tracking
```python
# Track average processing time by provider
provider_times = {}
for query in all_queries:
    provider = query["metadata"].get("llm_provider", "unknown")
    time = query["metadata"].get("processing_time", 0)
    
    if provider not in provider_times:
        provider_times[provider] = []
    provider_times[provider].append(time)

# Calculate averages
for provider, times in provider_times.items():
    avg_time = sum(times) / len(times)
    print(f"{provider}: {avg_time:.2f}s average")
```

### Query Complexity Analysis
```python
# Analyze query patterns
simple_queries = [q for q in all_queries if len(q["text"].split()) <= 3]
complex_queries = [q for q in all_queries if len(q["text"].split()) > 5]

simple_spike_rate = sum(1 for q in simple_queries if q["has_spike"]) / len(simple_queries)
complex_spike_rate = sum(1 for q in complex_queries if q["has_spike"]) / len(complex_queries)
```

## Storage Backends

### FileSystemDataStore
- Stores queries in `<base_path>/queries/queries.json`
- Appends new queries to existing file
- Human-readable JSON format

### SQLiteDataStore
- Stores in `queries` table with indices
- Supports efficient filtering and pagination
- Better for large-scale deployments

### InMemoryDataStore
- Stores queries in memory only
- Good for testing and temporary analysis
- Data lost on restart

## Best Practices

1. **Regular Analysis**: Periodically analyze query patterns to understand user behavior
2. **Archive Old Queries**: Move old queries to archive storage to maintain performance
3. **Privacy Considerations**: Be mindful of sensitive information in queries
4. **Monitor Storage**: Track disk usage as queries accumulate over time

## Future Enhancements

- Query similarity clustering
- Automatic pattern learning from successful queries  
- Query recommendation based on history
- Time-series analysis of spike generation rates
- Export to analytics platforms