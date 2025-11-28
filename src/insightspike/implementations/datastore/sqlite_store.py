"""
SQLite Data Store Implementation
================================

SQLite-based implementation of AsyncDataStore for scalable local storage.
Uses aiosqlite for async operations and integrates with FAISS for vector search.
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import aiosqlite
import numpy as np

from ...core.base.async_datastore import AsyncDataStore
from ...core.base.datastore import VectorIndex
from ...vector_index import VectorIndexFactory

logger = logging.getLogger(__name__)


class ConfigurableVectorIndex(VectorIndex):
    """Vector index implementation using VectorIndexFactory"""

    def __init__(self, dimension: int, index_type: str = "Flat"):
        self.dimension = dimension
        self.index_type = index_type

        # Use VectorIndexFactory to create backend-agnostic index
        self.index = VectorIndexFactory.create_index(
            dimension=dimension,
            index_type="auto"  # Let factory decide based on availability
        )

        self.id_map = {}  # Map from internal ID to our IDs
        self.reverse_id_map = {}  # Map from our IDs to internal IDs

    def add_vectors(self, vectors: np.ndarray, ids: Optional[List[int]] = None) -> bool:
        """Add vectors to FAISS index"""
        try:
            if ids is None:
                ids = list(range(len(vectors)))

            # Add to FAISS
            start_idx = self.index.ntotal
            self.index.add(vectors.astype(np.float32))

            # Update ID mappings
            for i, vec_id in enumerate(ids):
                faiss_id = start_idx + i
                self.id_map[faiss_id] = vec_id
                self.reverse_id_map[vec_id] = faiss_id

            return True
        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            return False

    def search(
        self, query_vectors: np.ndarray, k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors"""
        distances, indices = self.index.search(query_vectors.astype(np.float32), k)

        # Map FAISS indices to our IDs
        mapped_indices = []
        for i in range(indices.shape[0]):
            row_indices = []
            for j in range(indices.shape[1]):
                faiss_id = indices[i, j]
                if faiss_id >= 0 and faiss_id in self.id_map:
                    row_indices.append(self.id_map[faiss_id])
                else:
                    row_indices.append(-1)  # Invalid
            mapped_indices.append(row_indices)

        mapped_indices = np.array(mapped_indices)

        return distances, mapped_indices

    def remove_vectors(self, ids: List[int]) -> bool:
        """Remove vectors - not supported in basic FAISS"""
        logger.warning("Vector removal not supported in FAISS Flat index")
        return False

    def save_index(self, path: str) -> bool:
        """Save FAISS index"""
        try:
            faiss.write_index(self.index, path)
            # Save ID mappings
            mapping_path = path + ".mapping"
            with open(mapping_path, "w") as f:
                json.dump(
                    {
                        "id_map": {str(k): v for k, v in self.id_map.items()},
                        "reverse_id_map": {
                            str(k): v for k, v in self.reverse_id_map.items()
                        },
                    },
                    f,
                )
            return True
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False

    def load_index(self, path: str) -> bool:
        """Load FAISS index"""
        try:
            self.index = faiss.read_index(path)
            # Load ID mappings
            mapping_path = path + ".mapping"
            if os.path.exists(mapping_path):
                with open(mapping_path, "r") as f:
                    mappings = json.load(f)
                    self.id_map = {int(k): v for k, v in mappings["id_map"].items()}
                    self.reverse_id_map = {
                        int(k): v for k, v in mappings["reverse_id_map"].items()
                    }
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def get_size(self) -> int:
        """Get number of vectors"""
        return self.index.ntotal


# Backward compatibility alias for legacy tests
class FAISSVectorIndex(ConfigurableVectorIndex):
    """Alias to maintain compatibility with older FAISSVectorIndex references."""
    pass


class SQLiteDataStore(AsyncDataStore):
    """SQLite implementation of AsyncDataStore"""

    def __init__(self, db_path: str, vector_dim: int = 384):
        self.db_path = db_path
        self.vector_dim = vector_dim
        self._ensure_db_exists()

        # Initialize vector indices
        self.vector_indices = {}

        # Create tables
        self._init_tables_sync()

    def _ensure_db_exists(self):
        """Ensure database directory exists"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    def _init_tables_sync(self):
        """Initialize database tables (sync version)"""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Episodes table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                text TEXT NOT NULL,
                vector BLOB NOT NULL,
                c_value REAL DEFAULT 0.5,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Indices for episodes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_namespace ON episodes(namespace)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_created_at ON episodes(created_at)"
        )

        # Graph nodes table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                node_type TEXT,
                attributes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Graph edges table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT NOT NULL,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT,
                attributes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES graph_nodes(id),
                FOREIGN KEY (target_id) REFERENCES graph_nodes(id)
            )
        """
        )

        # Indices for graph
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_namespace ON graph_edges(namespace)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target_id)"
        )

        # Metadata table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    async def _init_tables(self):
        """Initialize database tables"""
        async with aiosqlite.connect(self.db_path) as db:
            # Episodes table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    text TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    c_value REAL DEFAULT 0.5,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Indices for episodes
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_namespace ON episodes(namespace)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_created_at ON episodes(created_at)"
            )

            # Graph nodes table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    node_type TEXT,
                    attributes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Graph edges table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    edge_type TEXT,
                    attributes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES graph_nodes(id),
                    FOREIGN KEY (target_id) REFERENCES graph_nodes(id)
                )
            """
            )

            # Indices for graph
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_namespace ON graph_edges(namespace)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target_id)"
            )

            # Metadata table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            await db.commit()

    # ========== Episode Operations ==========

    async def add_episode(
        self, episode: Dict[str, Any], namespace: str = "default"
    ) -> str:
        """Add a single episode"""
        episode_id = episode.get("id", str(uuid4()))

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO episodes (id, namespace, text, vector, c_value, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    episode_id,
                    namespace,
                    episode["text"],
                    episode["vec"].tobytes()
                    if isinstance(episode["vec"], np.ndarray)
                    else episode["vec"],
                    episode.get("c", 0.5),
                    json.dumps(episode.get("metadata", {})),
                ),
            )
            await db.commit()

        # Update vector index
        await self._update_vector_index(namespace, episode_id, episode["vec"])

        return episode_id

    async def batch_add_episodes(
        self, episodes: List[Dict[str, Any]], namespace: str = "default"
    ) -> List[str]:
        """Add multiple episodes in batch"""
        episode_ids = []

        async with aiosqlite.connect(self.db_path) as db:
            for episode in episodes:
                episode_id = episode.get("id", str(uuid4()))
                episode_ids.append(episode_id)

                await db.execute(
                    """
                    INSERT OR REPLACE INTO episodes (id, namespace, text, vector, c_value, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        episode_id,
                        namespace,
                        episode["text"],
                        episode["vec"].tobytes()
                        if isinstance(episode["vec"], np.ndarray)
                        else episode["vec"],
                        episode.get("c", 0.5),
                        json.dumps(episode.get("metadata", {})),
                    ),
                )

            await db.commit()

        # Update vector index in batch
        vectors = np.array([ep["vec"] for ep in episodes])
        await self._batch_update_vector_index(namespace, episode_ids, vectors)

        return episode_ids

    async def search_episodes_by_vector(
        self,
        query_vector: np.ndarray,
        k: int = 20,
        threshold: float = 0.7,
        namespace: str = "default",
    ) -> List[Dict[str, Any]]:
        """Search for episodes by vector similarity"""
        # Get vector index for namespace
        index = await self._get_vector_index(namespace)
        if index is None or index.get_size() == 0:
            return []

        # Search in FAISS
        distances, indices = index.search(
            query_vector.reshape(1, -1), min(k, index.get_size())
        )

        # Filter by threshold (convert L2 distance to similarity)
        valid_ids = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if isinstance(idx, (list, np.ndarray)):
                # If idx is already mapped
                actual_id = idx[0] if isinstance(idx, list) else idx.item()
            else:
                # If idx is a number
                actual_id = idx

            if actual_id != -1:  # Valid index
                # Convert L2 distance to cosine similarity approximation
                similarity = 1 - (dist / (2 * np.sqrt(2)))  # Rough approximation
                if similarity >= threshold:
                    valid_ids.append(actual_id)

        if not valid_ids:
            return []

        # Fetch episodes from database
        episodes = []
        async with aiosqlite.connect(self.db_path) as db:
            placeholders = ",".join(["?" for _ in valid_ids])
            # Detect legacy column naming (content vs text)
            text_col = "text"
            try:
                colcur = await db.execute("PRAGMA table_info(episodes)")
                cols = [row[1] for row in await colcur.fetchall()]
                if "text" not in cols and "content" in cols:
                    text_col = "content"
            except Exception:
                text_col = "text"
            query = f"""
                SELECT id, {text_col} as text, vector, c_value, metadata
                FROM episodes
                WHERE namespace = ? AND id IN ({placeholders})
            """

            cursor = await db.execute(query, [namespace] + valid_ids)
            rows = await cursor.fetchall()

            for row in rows:
                episodes.append(
                    {
                        "id": row[0],
                        "text": row[1],
                        "vec": np.frombuffer(row[2], dtype=np.float32),
                        "c": row[3],
                        "metadata": json.loads(row[4]),
                    }
                )

        return episodes

    async def get_episodes_by_ids(
        self, ids: List[str], namespace: str = "default"
    ) -> List[Dict[str, Any]]:
        """Get multiple episodes by their IDs"""
        episodes = []

        async with aiosqlite.connect(self.db_path) as db:
            # Detect legacy column naming (content vs text)
            text_col = "text"
            try:
                colcur = await db.execute("PRAGMA table_info(episodes)")
                cols = [row[1] for row in await colcur.fetchall()]
                if "text" not in cols and "content" in cols:
                    text_col = "content"
            except Exception:
                text_col = "text"
            placeholders = ",".join(["?" for _ in ids])
            query = f"""
                SELECT id, {text_col} as text, vector, c_value, metadata
                FROM episodes
                WHERE namespace = ? AND id IN ({placeholders})
            """
            cursor = await db.execute(query, [namespace] + ids)
            rows = await cursor.fetchall()

            for row in rows:
                episodes.append(
                    {
                        "id": row[0],
                        "text": row[1],
                        "vec": np.frombuffer(row[2], dtype=np.float32),
                        "c": row[3],
                        "metadata": json.loads(row[4]),
                    }
                )

        return episodes

    async def update_episode(
        self, episode_id: str, updates: Dict[str, Any], namespace: str = "default"
    ) -> bool:
        """Update an existing episode"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Build update query
                update_fields = []
                values = []

                if "text" in updates:
                    update_fields.append("text = ?")
                    values.append(updates["text"])

                if "vec" in updates:
                    update_fields.append("vector = ?")
                    vec = updates["vec"]
                    values.append(vec.tobytes() if isinstance(vec, np.ndarray) else vec)

                if "c" in updates:
                    update_fields.append("c_value = ?")
                    values.append(updates["c"])

                if "metadata" in updates:
                    update_fields.append("metadata = ?")
                    values.append(json.dumps(updates["metadata"]))

                update_fields.append("updated_at = CURRENT_TIMESTAMP")

                query = f"""
                    UPDATE episodes
                    SET {', '.join(update_fields)}
                    WHERE id = ? AND namespace = ?
                """

                values.extend([episode_id, namespace])
                await db.execute(query, values)
                await db.commit()

                # Update vector index if vector changed
                if "vec" in updates:
                    await self._update_vector_index(
                        namespace, episode_id, updates["vec"]
                    )

                return True

        except Exception as e:
            logger.error(f"Failed to update episode: {e}")
            return False

    # ========== Vector Index Management ==========

    async def _get_vector_index(self, namespace: str) -> Optional[ConfigurableVectorIndex]:
        """Get or create vector index for namespace"""
        if namespace not in self.vector_indices:
            # Try to load from disk
            index_path = f"{self.db_path}.{namespace}.faiss"
            if os.path.exists(index_path):
                index = ConfigurableVectorIndex(self.vector_dim)
                if index.load_index(index_path):
                    self.vector_indices[namespace] = index
                else:
                    # Create new index
                    self.vector_indices[namespace] = ConfigurableVectorIndex(self.vector_dim)
            else:
                # Create new index
                self.vector_indices[namespace] = ConfigurableVectorIndex(self.vector_dim)

        return self.vector_indices.get(namespace)

    async def _update_vector_index(
        self, namespace: str, episode_id: str, vector: np.ndarray
    ):
        """Update vector index with single vector"""
        index = await self._get_vector_index(namespace)
        if index:
            index.add_vectors(vector.reshape(1, -1), [episode_id])

    async def _batch_update_vector_index(
        self, namespace: str, episode_ids: List[str], vectors: np.ndarray
    ):
        """Update vector index with batch of vectors"""
        index = await self._get_vector_index(namespace)
        if index:
            index.add_vectors(vectors, episode_ids)

    # ========== Sync Methods (from base class) ==========

    def save_episodes(
        self, episodes: List[Dict[str, Any]], namespace: str = "default"
    ) -> bool:
        """Sync version of save_episodes"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.batch_add_episodes(episodes, namespace))
                return True
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Failed to save episodes: {e}")
            return False

    def load_episodes(self, namespace: str = "default") -> List[Dict[str, Any]]:
        """Sync version of load_episodes"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._load_all_episodes(namespace))
        finally:
            loop.close()

    async def _load_all_episodes(self, namespace: str) -> List[Dict[str, Any]]:
        """Load all episodes from namespace"""
        episodes = []

        async with aiosqlite.connect(self.db_path) as db:
            # Detect legacy column naming (content vs text)
            text_col = "text"
            try:
                colcur = await db.execute("PRAGMA table_info(episodes)")
                cols = [row[1] for row in await colcur.fetchall()]
                if "text" not in cols and "content" in cols:
                    text_col = "content"
            except Exception:
                text_col = "text"
            query = f"""
                SELECT id, {text_col} as text, vector, c_value, metadata
                FROM episodes
                WHERE namespace = ?
                ORDER BY created_at
            """
            cursor = await db.execute(query, (namespace,))

            rows = await cursor.fetchall()

            for row in rows:
                episodes.append(
                    {
                        "id": row[0],
                        "text": row[1],
                        "vec": np.frombuffer(row[2], dtype=np.float32),
                        "c": row[3],
                        "metadata": json.loads(row[4]),
                    }
                )

        return episodes

    # ========== Graph Operations (from Mixin) ==========

    async def get_graph_neighbors(
        self, node_id: str, hop: int = 1, namespace: str = "graphs"
    ) -> Dict[str, List[str]]:
        """Get neighboring nodes without loading entire graph"""
        neighbors = {}

        async with aiosqlite.connect(self.db_path) as db:
            # Start with the given node
            current_nodes = [node_id]

            for h in range(1, hop + 1):
                if not current_nodes:
                    break

                neighbors[h] = []

                # Find all neighbors of current nodes
                placeholders = ",".join(["?" for _ in current_nodes])
                query = f"""
                    SELECT DISTINCT target_id
                    FROM graph_edges
                    WHERE namespace = ? AND source_id IN ({placeholders})
                    UNION
                    SELECT DISTINCT source_id
                    FROM graph_edges
                    WHERE namespace = ? AND target_id IN ({placeholders})
                """

                params = [namespace] + current_nodes + [namespace] + current_nodes
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()

                # Collect neighbors, excluding already visited nodes
                visited = set([node_id])
                for i in range(1, h):
                    if i in neighbors:
                        visited.update(neighbors[i])

                for row in rows:
                    neighbor_id = row[0]
                    if neighbor_id not in visited and neighbor_id != node_id:
                        neighbors[h].append(neighbor_id)

                # Next hop starts from current neighbors
                current_nodes = neighbors[h]

        return neighbors

    async def update_graph_edges(
        self,
        edges_to_add: List[Tuple[str, str, Dict[str, Any]]],
        edges_to_remove: List[Tuple[str, str]],
        namespace: str = "graphs",
    ) -> bool:
        """Update graph edges incrementally"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Remove edges
                for source, target in edges_to_remove:
                    await db.execute(
                        """
                        DELETE FROM graph_edges
                        WHERE namespace = ? AND source_id = ? AND target_id = ?
                    """,
                        (namespace, source, target),
                    )

                # Add edges
                for source, target, attributes in edges_to_add:
                    # Ensure nodes exist
                    await db.execute(
                        """
                        INSERT OR IGNORE INTO graph_nodes (id, namespace, node_type)
                        VALUES (?, ?, 'auto')
                    """,
                        (source, namespace),
                    )

                    await db.execute(
                        """
                        INSERT OR IGNORE INTO graph_nodes (id, namespace, node_type)
                        VALUES (?, ?, 'auto')
                    """,
                        (target, namespace),
                    )

                    # Add edge
                    await db.execute(
                        """
                        INSERT INTO graph_edges (namespace, source_id, target_id, attributes)
                        VALUES (?, ?, ?, ?)
                    """,
                        (namespace, source, target, json.dumps(attributes)),
                    )

                await db.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to update graph edges: {e}")
            return False

    async def get_subgraph(
        self, node_ids: List[str], namespace: str = "graphs"
    ) -> Dict[str, Any]:
        """Get subgraph containing specified nodes"""
        subgraph = {"nodes": {}, "edges": []}

        async with aiosqlite.connect(self.db_path) as db:
            # Get nodes
            placeholders = ",".join(["?" for _ in node_ids])
            query = f"""
                SELECT id, node_type, attributes
                FROM graph_nodes
                WHERE namespace = ? AND id IN ({placeholders})
            """

            cursor = await db.execute(query, [namespace] + node_ids)
            rows = await cursor.fetchall()

            for row in rows:
                subgraph["nodes"][row[0]] = {
                    "type": row[1],
                    "attributes": json.loads(row[2]) if row[2] else {},
                }

            # Get edges between these nodes
            query = f"""
                SELECT source_id, target_id, edge_type, attributes
                FROM graph_edges
                WHERE namespace = ? 
                AND source_id IN ({placeholders})
                AND target_id IN ({placeholders})
            """

            cursor = await db.execute(query, [namespace] + node_ids + node_ids)
            rows = await cursor.fetchall()

            for row in rows:
                subgraph["edges"].append(
                    {
                        "source": row[0],
                        "target": row[1],
                        "type": row[2],
                        "attributes": json.loads(row[3]) if row[3] else {},
                    }
                )

        return subgraph

    # ========== Streaming Operations ==========

    async def stream_episodes(self, batch_size: int = 100, namespace: str = "default"):
        """Stream episodes in batches"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT id, text, vector, c_value, metadata
                FROM episodes
                WHERE namespace = ?
                ORDER BY created_at
            """,
                (namespace,),
            )

            while True:
                rows = await cursor.fetchmany(batch_size)
                if not rows:
                    break

                batch = []
                for row in rows:
                    batch.append(
                        {
                            "id": row[0],
                            "text": row[1],
                            "vec": np.frombuffer(row[2], dtype=np.float32),
                            "c": row[3],
                            "metadata": json.loads(row[4]),
                        }
                    )

                yield batch

    async def stream_search_results(
        self,
        query_vector: np.ndarray,
        max_results: int = 1000,
        batch_size: int = 100,
        namespace: str = "default",
    ):
        """Stream search results in batches"""
        # Get vector index
        index = await self._get_vector_index(namespace)
        if index is None or index.get_size() == 0:
            return

        # Search for more results than needed
        distances, indices = index.search(
            query_vector.reshape(1, -1), min(max_results, index.get_size())
        )

        # Process in batches
        valid_indices = [idx for idx in indices[0] if idx >= 0]

        for i in range(0, len(valid_indices), batch_size):
            batch_indices = valid_indices[i : i + batch_size]
            if not batch_indices:
                break

            # Fetch batch from database
            episodes = await self.get_episodes_by_ids(batch_indices, namespace)
            yield episodes

    # ========== Transaction Support ==========

    async def begin_transaction(self) -> str:
        """Begin a new transaction"""
        # SQLite handles transactions automatically with async context
        # Return a pseudo transaction ID
        from datetime import datetime

        return f"txn_{datetime.now().timestamp()}"

    async def commit_transaction(self, transaction_id: str) -> bool:
        """Commit a transaction"""
        # SQLite auto-commits with async context
        return True

    async def rollback_transaction(self, transaction_id: str) -> bool:
        """Rollback a transaction"""
        # Would need more complex implementation for real transaction support
        logger.warning("Transaction rollback not fully implemented in SQLite store")
        return True

    # ========== Vector Operations ==========

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
                if j < len(indices[i]):
                    idx = indices[i][j]
                    # Check if it's a valid index (not -1)
                    if isinstance(idx, (int, np.integer)) and idx != -1:
                        valid_indices.append(idx)
                        valid_distances.append(distances[i][j])
                    elif isinstance(idx, str):
                        # Already mapped to string ID
                        valid_indices.append(idx)
                        valid_distances.append(distances[i][j])

            results.append((valid_indices, valid_distances))

        return results

    # ========== Additional Sync Methods ==========

    def save_vectors(
        self,
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]],
        namespace: str = "vectors",
    ) -> bool:
        """Save vectors with metadata"""
        # Convert to episodes format and save
        episodes = []
        for i, (vec, meta) in enumerate(zip(vectors, metadata)):
            episodes.append(
                {
                    "id": meta.get("id", f"{namespace}_{i}"),
                    "text": meta.get("text", ""),
                    "vec": vec,
                    "metadata": meta,
                }
            )

        return self.save_episodes(episodes, namespace)

    def search_vectors(
        self, query_vector: np.ndarray, k: int = 10, namespace: str = "vectors"
    ) -> Tuple[List[int], List[float]]:
        """Search for similar vectors"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(
                self.search_episodes_by_vector(query_vector, k, 0.0, namespace)
            )
        finally:
            loop.close()

        if not results:
            return [], []

        indices = [int(r["id"].split("_")[-1]) for r in results]
        # Calculate distances (L2) from similarities
        distances = []
        for r in results:
            similarity = np.dot(query_vector, r["vec"]) / (
                np.linalg.norm(query_vector) * np.linalg.norm(r["vec"])
            )
            distance = 2 * (1 - similarity)  # Convert to L2-like distance
            distances.append(distance)

        return indices, distances

    def save_graph(
        self, graph_data: Any, graph_id: str, namespace: str = "graphs"
    ) -> bool:
        """Save graph data"""
        try:
            # Assuming graph_data is a dict with 'nodes' and 'edges'
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    self._save_graph(graph_data, graph_id, namespace)
                )
            finally:
                loop.close()
            return True
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
            return False

    async def _save_graph(
        self, graph_data: Dict[str, Any], graph_id: str, namespace: str
    ):
        """Internal async graph save"""
        async with aiosqlite.connect(self.db_path) as db:
            # Save nodes
            for node_id, node_data in graph_data.get("nodes", {}).items():
                await db.execute(
                    """
                    INSERT OR REPLACE INTO graph_nodes (id, namespace, node_type, attributes)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        node_id,
                        namespace,
                        node_data.get("type", "default"),
                        json.dumps(node_data.get("attributes", {})),
                    ),
                )

            # Save edges
            for edge in graph_data.get("edges", []):
                await db.execute(
                    """
                    INSERT INTO graph_edges (namespace, source_id, target_id, edge_type, attributes)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        namespace,
                        edge["source"],
                        edge["target"],
                        edge.get("type", "default"),
                        json.dumps(edge.get("attributes", {})),
                    ),
                )

            # Save graph metadata
            await db.execute(
                """
                INSERT OR REPLACE INTO metadata (key, namespace, value)
                VALUES (?, ?, ?)
            """,
                (
                    f"graph_{graph_id}",
                    namespace,
                    json.dumps(
                        {"id": graph_id, "node_count": len(graph_data.get("nodes", {}))}
                    ),
                ),
            )

            await db.commit()

    def load_graph(self, graph_id: str, namespace: str = "graphs") -> Optional[Any]:
        """Load graph data"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._load_graph(graph_id, namespace))
        finally:
            loop.close()

    async def _load_graph(
        self, graph_id: str, namespace: str
    ) -> Optional[Dict[str, Any]]:
        """Internal async graph load"""
        graph_data = {"nodes": {}, "edges": []}

        async with aiosqlite.connect(self.db_path) as db:
            # Load all nodes in namespace
            cursor = await db.execute(
                """
                SELECT id, node_type, attributes
                FROM graph_nodes
                WHERE namespace = ?
            """,
                (namespace,),
            )

            rows = await cursor.fetchall()
            for row in rows:
                graph_data["nodes"][row[0]] = {
                    "type": row[1],
                    "attributes": json.loads(row[2]) if row[2] else {},
                }

            # Load all edges in namespace
            cursor = await db.execute(
                """
                SELECT source_id, target_id, edge_type, attributes
                FROM graph_edges
                WHERE namespace = ?
            """,
                (namespace,),
            )

            rows = await cursor.fetchall()
            for row in rows:
                graph_data["edges"].append(
                    {
                        "source": row[0],
                        "target": row[1],
                        "type": row[2],
                        "attributes": json.loads(row[3]) if row[3] else {},
                    }
                )

        return graph_data if graph_data["nodes"] else None

    def save_metadata(
        self, metadata: Dict[str, Any], key: str, namespace: str = "metadata"
    ) -> bool:
        """Save arbitrary metadata"""
        try:
            # Use sync SQLite for sync methods
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO metadata (key, namespace, value)
                VALUES (?, ?, ?)
            """,
                (key, namespace, json.dumps(metadata)),
            )
            conn.commit()
            conn.close()
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
        try:
            # Use sync SQLite for sync methods
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT value FROM metadata
                WHERE key = ? AND namespace = ?
            """,
                (key, namespace),
            )

            row = cursor.fetchone()
            conn.close()

            if row:
                return json.loads(row[0])
            return None
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return None

    async def _load_metadata(
        self, key: str, namespace: str
    ) -> Optional[Dict[str, Any]]:
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
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._delete(key, namespace))
                return True
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Failed to delete: {e}")
            return False

    async def _delete(self, key: str, namespace: str):
        """Internal async delete"""
        async with aiosqlite.connect(self.db_path) as db:
            # Delete from appropriate table based on namespace
            if namespace == "metadata":
                await db.execute(
                    "DELETE FROM metadata WHERE key = ? AND namespace = ?",
                    (key, namespace),
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
                    "DELETE FROM episodes WHERE id = ? AND namespace = ?",
                    (key, namespace),
                )

            await db.commit()

    def list_keys(self, namespace: str, pattern: Optional[str] = None) -> List[str]:
        """List keys in namespace"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._list_keys(namespace, pattern))
        finally:
            loop.close()

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
            if namespace == "metadata":
                if pattern:
                    query += " AND key LIKE ?"
                    params.append(f"%{pattern}%")
            elif namespace == "graphs":
                if pattern:
                    query += " AND id LIKE ?"
                    params.append(f"%{pattern}%")
            else:
                if pattern:
                    query += " AND id LIKE ?"
                    params.append(f"%{pattern}%")

            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

            keys = [row[0] for row in rows]

        return keys

    def clear_namespace(self, namespace: str) -> bool:
        """Clear all data in a namespace"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._clear_namespace(namespace))
                return True
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Failed to clear namespace: {e}")
            return False

    async def _clear_namespace(self, namespace: str):
        """Internal async clear namespace"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM episodes WHERE namespace = ?", (namespace,))
            await db.execute(
                "DELETE FROM graph_edges WHERE namespace = ?", (namespace,)
            )
            await db.execute(
                "DELETE FROM graph_nodes WHERE namespace = ?", (namespace,)
            )
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
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._get_stats())
        finally:
            loop.close()

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

    # ========== Sleep/Forget GC (Edge pooling + orphan pruning) ==========

    def run_sleep_gc(
        self,
        namespace: str = "graphs",
        max_degree: int = 12,
        min_weight: float = 0.2,
        decay: float = 0.98,
        keep_pinned: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform a lightweight "sleep/forget" maintenance:
        - Edge pooling: keep top-K strongest undirected pairs per node, drop the rest
        - Prune weak edges (weight < min_weight)
        - Decay retained edge weights by `decay`
        - Remove orphan nodes (no incident edges), unless pinned

        Weight source priority in attributes JSON: 'weight' > 'adjusted_similarity' > 'similarity' > 'w'.
        Treat edges as undirected pairs for pooling and keep a pair if it appears in the
        union of top-K sets across endpoints.
        """
        import sqlite3

        def _edge_weight(attr_json: Optional[str]) -> float:
            try:
                if not attr_json:
                    return 1.0
                d = json.loads(attr_json)
                for k in ("weight", "adjusted_similarity", "similarity", "w"):
                    if k in d and isinstance(d[k], (int, float)):
                        return float(d[k])
            except Exception:
                pass
            return 1.0

        removed_edges = 0
        updated_edges = 0
        removed_nodes = 0

        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()

            # Load all edges for namespace
            cur.execute(
                """
                SELECT id, source_id, target_id, attributes
                FROM graph_edges
                WHERE namespace = ?
                """,
                (namespace,),
            )
            rows = cur.fetchall()

            # Build undirected pair map and adjacency
            pair_map: Dict[tuple[str, str], List[tuple[int, str, str, float, str]]] = {}
            for eid, s, t, attrs in rows:
                if s == t:
                    key = (s, t)
                else:
                    key = (s, t) if s < t else (t, s)
                w = _edge_weight(attrs)
                pair_map.setdefault(key, []).append((eid, s, t, w, attrs))

            # Pair representative weight (max of directions)
            pair_weight: Dict[tuple[str, str], float] = {
                key: max((w for (_eid, _s, _t, w, _a) in lst), default=0.0)
                for key, lst in pair_map.items()
            }

            # Build adjacency per node
            adj: Dict[str, List[tuple[tuple[str, str], float]]] = {}
            for (u, v), w in pair_weight.items():
                adj.setdefault(u, []).append(((u, v), w))
                if v != u:
                    adj.setdefault(v, []).append(((u, v), w))

            # Determine pairs to keep
            keep_pairs: set[tuple[str, str]] = set()
            for node, lst in adj.items():
                filtered = [(pair, w) for pair, w in lst if w >= min_weight]
                filtered.sort(key=lambda x: x[1], reverse=True)
                for pair, _w in filtered[: max_degree]:
                    keep_pairs.add(pair)

            # Compute edge deletions and updates
            delete_edge_ids: List[int] = []
            update_ids: List[int] = []

            for key, edges in pair_map.items():
                w = pair_weight.get(key, 0.0)
                if key not in keep_pairs or w < min_weight:
                    delete_edge_ids.extend([eid for (eid, *_rest) in edges])
                else:
                    update_ids.extend([eid for (eid, *_rest) in edges])

            # Apply deletions in chunks
            if delete_edge_ids:
                for i in range(0, len(delete_edge_ids), 500):
                    chunk = delete_edge_ids[i : i + 500]
                    cur.execute(
                        f"DELETE FROM graph_edges WHERE id IN ({','.join(['?']*len(chunk))})",
                        chunk,
                    )
                removed_edges = len(delete_edge_ids)

            # Update retained edge weights (attributes JSON)
            if update_ids:
                # Fetch existing attributes
                attr_map: Dict[int, str] = {}
                for i in range(0, len(update_ids), 500):
                    chunk = update_ids[i : i + 500]
                    cur.execute(
                        f"SELECT id, attributes FROM graph_edges WHERE id IN ({','.join(['?']*len(chunk))})",
                        chunk,
                    )
                    for rid, ajson in cur.fetchall():
                        attr_map[rid] = ajson

                updates: List[tuple[str, int]] = []
                for (u, v), w in pair_weight.items():
                    if (u, v) not in keep_pairs:
                        continue
                    new_w = float(w * decay)
                    for (eid, _s, _t, _ow, _a) in pair_map[(u, v)]:
                        try:
                            obj = json.loads(attr_map.get(eid, "{}"))
                            if not isinstance(obj, dict):
                                obj = {}
                        except Exception:
                            obj = {}
                        obj["weight"] = new_w
                        updates.append((json.dumps(obj), eid))

                for i in range(0, len(updates), 500):
                    chunk = updates[i : i + 500]
                    cur.executemany(
                        "UPDATE graph_edges SET attributes = ? WHERE id = ?",
                        chunk,
                    )
                updated_edges = len(updates)

            # Orphan nodes (no incident edges)
            cur.execute(
                """
                SELECT id, attributes FROM graph_nodes AS n
                WHERE namespace = ?
                  AND NOT EXISTS (
                    SELECT 1 FROM graph_edges e
                    WHERE e.namespace = ? AND (e.source_id = n.id OR e.target_id = n.id)
                  )
                """,
                (namespace, namespace),
            )
            orphan_rows = cur.fetchall()
            to_delete_nodes: List[str] = []
            for nid, attrs in orphan_rows:
                if keep_pinned:
                    try:
                        a = json.loads(attrs) if attrs else {}
                        if isinstance(a, dict) and a.get("pinned"):
                            continue
                    except Exception:
                        pass
                to_delete_nodes.append(nid)

            if to_delete_nodes:
                for i in range(0, len(to_delete_nodes), 500):
                    chunk = to_delete_nodes[i : i + 500]
                    cur.execute(
                        f"DELETE FROM graph_nodes WHERE id IN ({','.join(['?']*len(chunk))})",
                        chunk,
                    )
                removed_nodes = len(to_delete_nodes)

            conn.commit()

            return {
                "removed_edges": removed_edges,
                "updated_edges": updated_edges,
                "removed_nodes": removed_nodes,
                "kept_pairs": len(keep_pairs),
                "total_pairs": len(pair_map),
            }
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # ========== Utility Methods ==========

    def save_indices(self):
        """Save all vector indices to disk"""
        for namespace, index in self.vector_indices.items():
            index_path = f"{self.db_path}.{namespace}.faiss"
            index.save_index(index_path)
            logger.info(f"Saved index for namespace {namespace} to {index_path}")

    def close(self):
        """Close datastore and save indices"""
        self.save_indices()
        logger.info("SQLiteDataStore closed")

    # ========== Query Operations ==========

    def save_queries(
        self, queries: List[Dict[str, Any]], namespace: str = "queries"
    ) -> bool:
        """Save query records to storage"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._save_queries(queries, namespace))
        finally:
            loop.close()

    async def _save_queries(
        self, queries: List[Dict[str, Any]], namespace: str
    ) -> bool:
        """Internal async save queries"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Create queries table if it doesn't exist
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS queries (
                        id TEXT PRIMARY KEY,
                        namespace TEXT NOT NULL,
                        text TEXT NOT NULL,
                        vector BLOB,
                        has_spike INTEGER DEFAULT 0,
                        spike_episode_id TEXT,
                        response TEXT,
                        metadata TEXT DEFAULT '{}',
                        timestamp REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )
                
                # Create index on namespace and has_spike
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_queries_namespace ON queries(namespace)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_queries_spike ON queries(has_spike)"
                )
                
                # Insert queries
                for query in queries:
                    await db.execute(
                        """
                        INSERT OR REPLACE INTO queries 
                        (id, namespace, text, vector, has_spike, spike_episode_id, response, metadata, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            query.get("id", str(uuid4())),
                            namespace,
                            query["text"],
                            query.get("vec", np.array([])).tobytes() if "vec" in query else None,
                            1 if query.get("has_spike", False) else 0,
                            query.get("spike_episode_id"),
                            query.get("response", ""),
                            json.dumps(query.get("metadata", {})),
                            query.get("timestamp", time.time())
                        ),
                    )
                
                await db.commit()
                logger.info(f"Saved {len(queries)} queries to {namespace}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save queries: {e}")
            return False

    def load_queries(
        self, 
        namespace: str = "queries",
        has_spike: Optional[bool] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load query records from storage"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self._load_queries(namespace, has_spike, limit)
            )
        finally:
            loop.close()

    async def _load_queries(
        self, namespace: str, has_spike: Optional[bool], limit: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Internal async load queries"""
        queries = []
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Build query
                query = "SELECT * FROM queries WHERE namespace = ?"
                params = [namespace]
                
                # Add has_spike filter if specified
                if has_spike is not None:
                    query += " AND has_spike = ?"
                    params.append(1 if has_spike else 0)
                
                # Order by timestamp descending
                query += " ORDER BY timestamp DESC"
                
                # Add limit if specified
                if limit is not None:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()
                
                # Get column names
                columns = [desc[0] for desc in cursor.description]
                
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    
                    # Convert stored data
                    if row_dict.get("vector"):
                        row_dict["vec"] = np.frombuffer(row_dict["vector"], dtype=np.float32)
                        del row_dict["vector"]
                    
                    row_dict["has_spike"] = bool(row_dict.get("has_spike", 0))
                    
                    if row_dict.get("metadata"):
                        row_dict["metadata"] = json.loads(row_dict["metadata"])
                    
                    queries.append(row_dict)
                
                logger.info(f"Loaded {len(queries)} queries from {namespace}")
                return queries
                
        except Exception as e:
            logger.error(f"Failed to load queries: {e}")
            return []
