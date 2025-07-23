# Complete SQLiteDataStore implementation
# This shows the remaining methods to be added to sqlite_store.py

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import aiosqlite

logger = logging.getLogger(__name__)


def save_metadata(
    self, metadata: Dict[str, Any], key: str, namespace: str = "metadata"
) -> bool:
    """Save arbitrary metadata"""
    try:
        asyncio.run(self._save_metadata(metadata, key, namespace))
        return True
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
        return False


async def _save_metadata(self, metadata: Dict[str, Any], key: str, namespace: str):
    """Internal async metadata save"""
    async with aiosqlite.connect(self.db_path) as db:
        await db.execute(
            """
            INSERT OR REPLACE INTO metadata (key, namespace, value)
            VALUES (?, ?, ?)
        """,
            (key, namespace, json.dumps(metadata)),
        )
        await db.commit()


def load_metadata(
    self, key: str, namespace: str = "metadata"
) -> Optional[Dict[str, Any]]:
    """Load metadata"""
    return asyncio.run(self._load_metadata(key, namespace))


async def _load_metadata(self, key: str, namespace: str) -> Optional[Dict[str, Any]]:
    """Internal async metadata load"""
    async with aiosqlite.connect(self.db_path) as db:
        cursor = await db.execute(
            """
            SELECT value FROM metadata
            WHERE key = ? AND namespace = ?
        """,
            (key, namespace),
        )

        row = await cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None


def delete(self, key: str, namespace: str) -> bool:
    """Delete data by key"""
    try:
        asyncio.run(self._delete(key, namespace))
        return True
    except Exception as e:
        logger.error(f"Failed to delete: {e}")
        return False


async def _delete(self, key: str, namespace: str):
    """Internal async delete"""
    async with aiosqlite.connect(self.db_path) as db:
        # Delete from appropriate table based on namespace
        if namespace == "metadata":
            await db.execute(
                "DELETE FROM metadata WHERE key = ? AND namespace = ?", (key, namespace)
            )
        elif namespace == "graphs":
            # Delete graph nodes and edges
            await db.execute(
                "DELETE FROM graph_edges WHERE namespace = ? AND (source_id = ? OR target_id = ?)",
                (namespace, key, key),
            )
            await db.execute(
                "DELETE FROM graph_nodes WHERE id = ? AND namespace = ?",
                (key, namespace),
            )
        else:
            # Assume episodes
            await db.execute(
                "DELETE FROM episodes WHERE id = ? AND namespace = ?", (key, namespace)
            )

        await db.commit()


def list_keys(self, namespace: str, pattern: Optional[str] = None) -> List[str]:
    """List keys in namespace"""
    return asyncio.run(self._list_keys(namespace, pattern))


async def _list_keys(self, namespace: str, pattern: Optional[str]) -> List[str]:
    """Internal async list keys"""
    keys = []

    async with aiosqlite.connect(self.db_path) as db:
        if namespace == "metadata":
            query = "SELECT key FROM metadata WHERE namespace = ?"
            params = [namespace]
        elif namespace == "graphs":
            query = "SELECT DISTINCT id FROM graph_nodes WHERE namespace = ?"
            params = [namespace]
        else:
            query = "SELECT id FROM episodes WHERE namespace = ?"
            params = [namespace]

        # Add pattern matching if provided
        if pattern:
            query += " AND key LIKE ?"
            params.append(f"%{pattern}%")

        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()

        keys = [row[0] for row in rows]

    return keys


def clear_namespace(self, namespace: str) -> bool:
    """Clear all data in a namespace"""
    try:
        asyncio.run(self._clear_namespace(namespace))
        return True
    except Exception as e:
        logger.error(f"Failed to clear namespace: {e}")
        return False


async def _clear_namespace(self, namespace: str):
    """Internal async clear namespace"""
    async with aiosqlite.connect(self.db_path) as db:
        await db.execute("DELETE FROM episodes WHERE namespace = ?", (namespace,))
        await db.execute("DELETE FROM graph_edges WHERE namespace = ?", (namespace,))
        await db.execute("DELETE FROM graph_nodes WHERE namespace = ?", (namespace,))
        await db.execute("DELETE FROM metadata WHERE namespace = ?", (namespace,))
        await db.commit()

    # Clear vector index
    if namespace in self.vector_indices:
        del self.vector_indices[namespace]
        # Remove index file
        index_path = f"{self.db_path}.{namespace}.faiss"
        if os.path.exists(index_path):
            os.remove(index_path)
            os.remove(f"{index_path}.mapping")


def get_stats(self) -> Dict[str, Any]:
    """Get storage statistics"""
    return asyncio.run(self._get_stats())


async def _get_stats(self) -> Dict[str, Any]:
    """Internal async get stats"""
    stats = {"episodes": {}, "graphs": {}, "metadata": {}, "indices": {}}

    async with aiosqlite.connect(self.db_path) as db:
        # Episode stats
        cursor = await db.execute(
            """
            SELECT namespace, COUNT(*) as count
            FROM episodes
            GROUP BY namespace
        """
        )
        rows = await cursor.fetchall()
        for row in rows:
            stats["episodes"][row[0]] = row[1]

        # Graph stats
        cursor = await db.execute(
            """
            SELECT namespace, 
                   COUNT(DISTINCT id) as node_count
            FROM graph_nodes
            GROUP BY namespace
        """
        )
        rows = await cursor.fetchall()
        for row in rows:
            stats["graphs"][row[0]] = {"nodes": row[1]}

        # Edge counts
        cursor = await db.execute(
            """
            SELECT namespace, COUNT(*) as edge_count
            FROM graph_edges
            GROUP BY namespace
        """
        )
        rows = await cursor.fetchall()
        for row in rows:
            if row[0] in stats["graphs"]:
                stats["graphs"][row[0]]["edges"] = row[1]

        # Metadata stats
        cursor = await db.execute(
            """
            SELECT namespace, COUNT(*) as count
            FROM metadata
            GROUP BY namespace
        """
        )
        rows = await cursor.fetchall()
        for row in rows:
            stats["metadata"][row[0]] = row[1]

    # Vector index stats
    for namespace, index in self.vector_indices.items():
        stats["indices"][namespace] = {
            "vectors": index.get_size(),
            "dimension": index.dimension,
            "type": index.index_type,
        }

    # Database file size
    if os.path.exists(self.db_path):
        stats["db_size_bytes"] = os.path.getsize(self.db_path)

    return stats


# Vector search batch implementation
async def search_vectors_batch(
    self, query_vectors: np.ndarray, k: int = 10, namespace: str = "vectors"
) -> List[Tuple[List[int], List[float]]]:
    """Batch vector similarity search"""
    index = await self._get_vector_index(namespace)
    if index is None or index.get_size() == 0:
        return [([], []) for _ in range(len(query_vectors))]

    # Search all at once
    distances, indices = index.search(query_vectors, min(k, index.get_size()))

    # Format results
    results = []
    for i in range(len(query_vectors)):
        valid_indices = []
        valid_distances = []

        for j in range(k):
            if indices[i][j] >= 0:
                valid_indices.append(indices[i][j])
                valid_distances.append(distances[i][j])

        results.append((valid_indices, valid_distances))

    return results


# Save all vector indices to disk
def save_indices(self):
    """Save all vector indices to disk"""
    for namespace, index in self.vector_indices.items():
        index_path = f"{self.db_path}.{namespace}.faiss"
        index.save_index(index_path)
        logger.info(f"Saved index for namespace {namespace} to {index_path}")


# Close and cleanup
def close(self):
    """Close datastore and save indices"""
    self.save_indices()
    logger.info("SQLiteDataStore closed")
