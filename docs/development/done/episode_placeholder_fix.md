# Episode Placeholder Fix

## Problem
In `layer2_working_memory.py`, when searching episodes from DataStore, the code creates placeholder text like `"Episode {idx}"` instead of fetching actual episode content. This severely impacts answer quality.

## Solution

### Option 1: Implement get_episode_by_id in DataStore

Add method to SQLiteDataStore:

```python
async def get_episode_by_id(self, episode_id: str, namespace: str = "default") -> Optional[Dict[str, Any]]:
    """Fetch a single episode by ID"""
    async with aiosqlite.connect(self.db_path) as db:
        cursor = await db.execute(
            """
            SELECT id, text, vector, c_value, metadata, created_at
            FROM episodes
            WHERE namespace = ? AND id = ?
            """,
            (namespace, episode_id)
        )
        
        row = await cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "text": row[1],
                "vector": np.frombuffer(row[2], dtype=np.float32),
                "c_value": row[3],
                "metadata": json.loads(row[4]) if row[4] else {},
                "created_at": row[5]
            }
    return None
```

### Option 2: Batch fetch episodes after search

Modify `search_episodes` in layer2_working_memory.py:

```python
def search_episodes(self, query: str, k: int = 5, filter_fn: Optional[callable] = None) -> List[Dict[str, Any]]:
    # ... existing search code ...
    
    # After getting indices from vector search
    if indices:
        # Batch fetch actual episodes
        episode_ids = [f"episode_{idx}" for idx in indices[:k]]
        
        # Use DataStore's batch get (if available) or individual fetches
        actual_episodes = []
        for ep_id in episode_ids:
            # This needs implementation in DataStore
            episode_data = self.datastore.get_episode(ep_id, namespace=self.namespace)
            if episode_data:
                actual_episodes.append({
                    "text": episode_data["text"],
                    "similarity": similarities[i],
                    "c": episode_data.get("c_value", 0.5),
                    "metadata": episode_data.get("metadata", {})
                })
        
        return actual_episodes
```

## Impact
- Without this fix, the LLM receives placeholder text instead of real context
- This degrades answer quality significantly
- The fix ensures proper memory retrieval for context-aware responses