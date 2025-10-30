"""
SQLite Data Store Implementation with Configurable Vector Index (deprecated)
==========================================================================

DEPRECATION NOTICE:
- This module is being consolidated into the canonical `sqlite_store.SQLiteDataStore`.
- Prefer using the factory entrypoint:
    from insightspike.implementations.datastore.factory import DataStoreFactory
    store = DataStoreFactory.create("sqlite", db_path=..., vector_dim=...)

Existing imports continue to work during the transition window but will be removed
after consolidation completes. See docs/development/SQLITE_CONSOLIDATION_PLAN_2025_09.md
for details and migration guidance.
"""
import warnings as _warnings
_warnings.warn(
    "sqlite_store_configurable is deprecated; use DataStoreFactory.create('sqlite')",
    DeprecationWarning,
    stacklevel=2,
)

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import aiosqlite
import numpy as np

from ...core.base.async_datastore import AsyncDataStore
from ...core.base.datastore import VectorIndex
from .configurable_vector_index import ConfigurableVectorIndex

logger = logging.getLogger(__name__)


class SQLiteDataStore(AsyncDataStore):
    """
    SQLite-based async datastore with configurable vector search
    
    Features:
    - Async SQLite operations
    - Configurable vector search (FAISS/NumPy)
    - Episode and fact management
    - Batch operations for performance
    - Automatic schema migrations
    """

    SCHEMA_VERSION = 2

    def __init__(
        self,
        db_path: str = "./data/insightspike.db",
        vector_dim: int = 384,
        batch_size: int = 100,
        vector_backend: str = "auto",
        **kwargs,
    ):
        """
        Initialize SQLite datastore

        Args:
            db_path: Path to SQLite database file
            vector_dim: Dimension of vectors
            batch_size: Batch size for operations
            vector_backend: Vector index backend ("auto", "numpy", "faiss")
            **kwargs: Additional arguments
        """
        self.db_path = db_path
        self.vector_dim = vector_dim
        self.batch_size = batch_size
        self.vector_backend = vector_backend
        self.vector_indices: Dict[str, ConfigurableVectorIndex] = {}
        self._conn: Optional[aiosqlite.Connection] = None
        self._write_lock = asyncio.Lock()
        self._index_update_queue = asyncio.Queue()
        self._update_task = None

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized SQLiteDataStore with backend={vector_backend}, "
            f"vector_dim={vector_dim}, batch_size={batch_size}"
        )

    async def initialize(self):
        """Initialize database connection and create tables"""
        self._conn = await aiosqlite.connect(self.db_path)
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA cache_size=10000")
        await self._conn.execute("PRAGMA temp_store=MEMORY")

        await self._create_tables()
        await self._check_schema_version()

        # Start background update task
        self._update_task = asyncio.create_task(self._process_index_updates())

        logger.info("SQLiteDataStore initialized successfully")

    async def _create_tables(self):
        """Create database tables"""
        async with self._write_lock:
            # Metadata table
            await self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Episodes table
            await self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    content TEXT NOT NULL,
                    summary TEXT,
                    vector BLOB NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    episode_type TEXT DEFAULT 'regular',
                    c_value REAL DEFAULT 0.0,
                    activation_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Facts table
            await self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    fact TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source_episode_id TEXT,
                    vector BLOB,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_episode_id) REFERENCES episodes(id)
                )
            """
            )

            # Insights table (for storing detected insights)
            await self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS insights (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    content TEXT NOT NULL,
                    insight_type TEXT,
                    confidence REAL DEFAULT 1.0,
                    source_episodes TEXT,  -- JSON array of episode IDs
                    vector BLOB,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create indices
            indices = [
                "CREATE INDEX IF NOT EXISTS idx_episodes_namespace ON episodes(namespace)",
                "CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_episodes_type ON episodes(episode_type)",
                "CREATE INDEX IF NOT EXISTS idx_facts_namespace ON facts(namespace)",
                "CREATE INDEX IF NOT EXISTS idx_facts_confidence ON facts(confidence)",
                "CREATE INDEX IF NOT EXISTS idx_insights_namespace ON insights(namespace)",
                "CREATE INDEX IF NOT EXISTS idx_insights_type ON insights(insight_type)",
            ]

            for idx in indices:
                await self._conn.execute(idx)

            await self._conn.commit()

    async def _check_schema_version(self):
        """Check and update schema version"""
        cursor = await self._conn.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        )
        row = await cursor.fetchone()

        if row is None:
            # First time setup
            await self._conn.execute(
                "INSERT INTO metadata (key, value) VALUES ('schema_version', ?)",
                (str(self.SCHEMA_VERSION),),
            )
            await self._conn.commit()
        else:
            version = int(row[0])
            if version < self.SCHEMA_VERSION:
                await self._migrate_schema(version)

    async def _migrate_schema(self, from_version: int):
        """Migrate schema to latest version"""
        logger.info(f"Migrating schema from version {from_version} to {self.SCHEMA_VERSION}")

        # Add migration logic here as schema evolves
        if from_version < 2:
            # Add insights table
            await self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS insights (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    content TEXT NOT NULL,
                    insight_type TEXT,
                    confidence REAL DEFAULT 1.0,
                    source_episodes TEXT,
                    vector BLOB,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

        # Update version
        await self._conn.execute(
            "UPDATE metadata SET value = ?, updated_at = CURRENT_TIMESTAMP WHERE key = 'schema_version'",
            (str(self.SCHEMA_VERSION),),
        )
        await self._conn.commit()

    # ========== Episode Management ==========

    async def add_episode(
        self,
        namespace: str,
        episode_id: str,
        content: str,
        vector: np.ndarray,
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        episode_type: str = "regular",
        c_value: float = 0.0,
    ) -> bool:
        """Add a new episode"""
        try:
            async with self._write_lock:
                await self._conn.execute(
                    """
                    INSERT INTO episodes 
                    (id, namespace, content, summary, vector, metadata, episode_type, c_value)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        episode_id,
                        namespace,
                        content,
                        summary,
                        vector.tobytes(),
                        json.dumps(metadata) if metadata else None,
                        episode_type,
                        c_value,
                    ),
                )
                await self._conn.commit()

            # Queue vector index update
            await self._index_update_queue.put((namespace, episode_id, vector))

            logger.debug(f"Added episode {episode_id} to namespace {namespace}")
            return True

        except sqlite3.IntegrityError:
            logger.warning(f"Episode {episode_id} already exists")
            return False
        except Exception as e:
            logger.error(f"Failed to add episode: {e}")
            return False

    async def get_episode(
        self, namespace: str, episode_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get episode by ID"""
        cursor = await self._conn.execute(
            """
            SELECT id, content, summary, vector, timestamp, metadata, 
                   episode_type, c_value, activation_count, last_accessed
            FROM episodes
            WHERE namespace = ? AND id = ?
            """,
            (namespace, episode_id),
        )
        row = await cursor.fetchone()

        if row is None:
            return None

        # Update last accessed
        await self._conn.execute(
            "UPDATE episodes SET last_accessed = CURRENT_TIMESTAMP WHERE id = ?",
            (episode_id,),
        )

        return {
            "id": row[0],
            "content": row[1],
            "summary": row[2],
            "vec": np.frombuffer(row[3], dtype=np.float32),
            "timestamp": row[4],
            "metadata": json.loads(row[5]) if row[5] else {},
            "episode_type": row[6],
            "c_value": row[7],
            "activation_count": row[8],
            "last_accessed": row[9],
        }

    async def list_episodes(
        self,
        namespace: str,
        limit: int = 100,
        offset: int = 0,
        episode_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List episodes in namespace"""
        query = "SELECT id, content, summary, timestamp, episode_type, c_value FROM episodes WHERE namespace = ?"
        params = [namespace]

        if episode_type:
            query += " AND episode_type = ?"
            params.append(episode_type)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = await self._conn.execute(query, params)
        episodes = []
        async for row in cursor:
            episodes.append(
                {
                    "id": row[0],
                    "content": row[1],
                    "summary": row[2],
                    "timestamp": row[3],
                    "episode_type": row[4],
                    "c_value": row[5],
                }
            )

        return episodes

    async def search_episodes(
        self,
        namespace: str,
        query_vector: np.ndarray,
        k: int = 10,
        episode_type: Optional[str] = None,
        min_c_value: Optional[float] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search episodes by vector similarity"""
        # Get all episodes (we'll filter after vector search)
        cursor = await self._conn.execute(
            """
            SELECT id, content, summary, vector, timestamp, metadata, 
                   episode_type, c_value, activation_count
            FROM episodes
            WHERE namespace = ?
            """,
            (namespace,),
        )

        episodes = []
        vectors = []
        async for row in cursor:
            episode = {
                "id": row[0],
                "content": row[1],
                "summary": row[2],
                "timestamp": row[4],
                "metadata": json.loads(row[5]) if row[5] else {},
                "episode_type": row[6],
                "c_value": row[7],
                "activation_count": row[8],
            }

            # Apply filters
            if episode_type and episode["episode_type"] != episode_type:
                continue
            if min_c_value is not None and episode["c_value"] < min_c_value:
                continue

            episodes.append(episode)
            vectors.append(np.frombuffer(row[3], dtype=np.float32))

        if not episodes:
            return []

        # Perform vector search
        vectors = np.array(vectors)
        index = await self._get_vector_index(namespace)
        if index and index.ntotal > 0:
            distances, indices = index.search(query_vector.reshape(1, -1), min(k, len(episodes)))
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(episodes):
                    # Convert distance to similarity score
                    similarity = 1.0 / (1.0 + distances[0][i])
                    results.append((episodes[idx], similarity))
            
            return results
        else:
            # Fallback to brute force search
            similarities = np.dot(vectors, query_vector)
            top_k = np.argsort(similarities)[-k:][::-1]
            
            return [(episodes[i], float(similarities[i])) for i in top_k]

    async def update_episode(
        self, namespace: str, episode_id: str, updates: Dict[str, Any]
    ) -> bool:
        """Update episode fields"""
        try:
            # Build update query
            fields = []
            values = []
            for key, value in updates.items():
                if key in ["content", "summary", "metadata", "c_value", "episode_type"]:
                    fields.append(f"{key} = ?")
                    if key == "metadata":
                        values.append(json.dumps(value))
                    else:
                        values.append(value)

            if not fields:
                return True

            fields.append("updated_at = CURRENT_TIMESTAMP")
            query = f"UPDATE episodes SET {', '.join(fields)} WHERE namespace = ? AND id = ?"
            values.extend([namespace, episode_id])

            async with self._write_lock:
                await self._conn.execute(query, values)
                await self._conn.commit()

            return True

        except Exception as e:
            logger.error(f"Failed to update episode: {e}")
            return False

    # ========== Vector Index Management ==========

    async def _get_vector_index(self, namespace: str) -> Optional[ConfigurableVectorIndex]:
        """Get or create vector index for namespace"""
        if namespace not in self.vector_indices:
            # Try to load from disk
            index_path = f"{self.db_path}.{namespace}.index"
            if os.path.exists(index_path):
                index = ConfigurableVectorIndex(
                    self.vector_dim, 
                    index_type=self.vector_backend
                )
                if index.load_index(index_path):
                    self.vector_indices[namespace] = index
                else:
                    # Create new index
                    self.vector_indices[namespace] = ConfigurableVectorIndex(
                        self.vector_dim,
                        index_type=self.vector_backend
                    )
            else:
                # Create new index
                self.vector_indices[namespace] = ConfigurableVectorIndex(
                    self.vector_dim,
                    index_type=self.vector_backend
                )

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
        """Batch update vector index"""
        index = await self._get_vector_index(namespace)
        if index:
            index.add_vectors(vectors, episode_ids)

    async def _process_index_updates(self):
        """Background task to process vector index updates"""
        batch = []
        while True:
            try:
                # Collect updates for batching
                timeout = 0.1 if batch else None
                try:
                    update = await asyncio.wait_for(
                        self._index_update_queue.get(), timeout=timeout
                    )
                    batch.append(update)
                except asyncio.TimeoutError:
                    pass

                # Process batch if we have enough or timeout
                if len(batch) >= self.batch_size or (batch and timeout):
                    # Group by namespace
                    by_namespace = {}
                    for ns, eid, vec in batch:
                        if ns not in by_namespace:
                            by_namespace[ns] = ([], [])
                        by_namespace[ns][0].append(eid)
                        by_namespace[ns][1].append(vec)

                    # Update indices
                    for ns, (ids, vecs) in by_namespace.items():
                        vecs = np.array(vecs)
                        await self._batch_update_vector_index(ns, ids, vecs)

                    batch.clear()

            except Exception as e:
                logger.error(f"Error in index update task: {e}")
                await asyncio.sleep(1)

    # ========== Fact Management ==========

    async def add_fact(
        self,
        namespace: str,
        fact_id: str,
        fact: str,
        confidence: float,
        source_episode_id: Optional[str] = None,
        vector: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add a new fact"""
        try:
            async with self._write_lock:
                await self._conn.execute(
                    """
                    INSERT INTO facts 
                    (id, namespace, fact, confidence, source_episode_id, vector, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fact_id,
                        namespace,
                        fact,
                        confidence,
                        source_episode_id,
                        vector.tobytes() if vector is not None else None,
                        json.dumps(metadata) if metadata else None,
                    ),
                )
                await self._conn.commit()

            logger.debug(f"Added fact {fact_id} to namespace {namespace}")
            return True

        except sqlite3.IntegrityError:
            logger.warning(f"Fact {fact_id} already exists")
            return False
        except Exception as e:
            logger.error(f"Failed to add fact: {e}")
            return False

    async def get_fact(self, namespace: str, fact_id: str) -> Optional[Dict[str, Any]]:
        """Get fact by ID"""
        cursor = await self._conn.execute(
            """
            SELECT id, fact, confidence, source_episode_id, vector, metadata, created_at
            FROM facts
            WHERE namespace = ? AND id = ?
            """,
            (namespace, fact_id),
        )
        row = await cursor.fetchone()

        if row is None:
            return None

        return {
            "id": row[0],
            "fact": row[1],
            "confidence": row[2],
            "source_episode_id": row[3],
            "vec": np.frombuffer(row[4], dtype=np.float32) if row[4] else None,
            "metadata": json.loads(row[5]) if row[5] else {},
            "created_at": row[6],
        }

    async def list_facts(
        self, namespace: str, min_confidence: float = 0.0, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List facts in namespace"""
        cursor = await self._conn.execute(
            """
            SELECT id, fact, confidence, source_episode_id, created_at
            FROM facts
            WHERE namespace = ? AND confidence >= ?
            ORDER BY confidence DESC, created_at DESC
            LIMIT ?
            """,
            (namespace, min_confidence, limit),
        )

        facts = []
        async for row in cursor:
            facts.append(
                {
                    "id": row[0],
                    "fact": row[1],
                    "confidence": row[2],
                    "source_episode_id": row[3],
                    "created_at": row[4],
                }
            )

        return facts

    # ========== Insight Management ==========

    async def add_insight(
        self,
        namespace: str,
        insight_id: str,
        content: str,
        insight_type: Optional[str] = None,
        confidence: float = 1.0,
        source_episodes: Optional[List[str]] = None,
        vector: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add a new insight"""
        try:
            async with self._write_lock:
                await self._conn.execute(
                    """
                    INSERT INTO insights 
                    (id, namespace, content, insight_type, confidence, source_episodes, vector, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        insight_id,
                        namespace,
                        content,
                        insight_type,
                        confidence,
                        json.dumps(source_episodes) if source_episodes else None,
                        vector.tobytes() if vector is not None else None,
                        json.dumps(metadata) if metadata else None,
                    ),
                )
                await self._conn.commit()

            logger.info(f"Added insight {insight_id} to namespace {namespace}")
            return True

        except sqlite3.IntegrityError:
            logger.warning(f"Insight {insight_id} already exists")
            return False
        except Exception as e:
            logger.error(f"Failed to add insight: {e}")
            return False

    async def search_insights(
        self,
        namespace: str,
        query_vector: np.ndarray,
        k: int = 5,
        insight_type: Optional[str] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search insights by vector similarity"""
        # Get insights
        query = """
            SELECT id, content, insight_type, confidence, source_episodes, vector, metadata
            FROM insights
            WHERE namespace = ?
        """
        params = [namespace]
        
        if insight_type:
            query += " AND insight_type = ?"
            params.append(insight_type)
            
        cursor = await self._conn.execute(query, params)
        
        insights = []
        vectors = []
        async for row in cursor:
            if row[5]:  # Has vector
                insights.append({
                    "id": row[0],
                    "content": row[1],
                    "insight_type": row[2],
                    "confidence": row[3],
                    "source_episodes": json.loads(row[4]) if row[4] else [],
                    "metadata": json.loads(row[6]) if row[6] else {},
                })
                vectors.append(np.frombuffer(row[5], dtype=np.float32))
        
        if not insights:
            return []
            
        # Vector similarity search
        vectors = np.array(vectors)
        similarities = np.dot(vectors, query_vector)
        top_k = np.argsort(similarities)[-k:][::-1]
        
        return [(insights[i], float(similarities[i])) for i in top_k]

    # ========== Cleanup & Stats ==========

    async def get_stats(self) -> Dict[str, Any]:
        """Get datastore statistics"""
        stats = {}

        # Episode counts by namespace and type
        cursor = await self._conn.execute(
            """
            SELECT namespace, episode_type, COUNT(*) as count
            FROM episodes
            GROUP BY namespace, episode_type
            """
        )
        episode_stats = {}
        async for row in cursor:
            ns, ep_type, count = row
            if ns not in episode_stats:
                episode_stats[ns] = {}
            episode_stats[ns][ep_type] = count
        stats["episodes"] = episode_stats

        # Fact counts
        cursor = await self._conn.execute(
            """
            SELECT namespace, COUNT(*) as count, AVG(confidence) as avg_confidence
            FROM facts
            GROUP BY namespace
            """
        )
        fact_stats = {}
        async for row in cursor:
            ns, count, avg_conf = row
            fact_stats[ns] = {"count": count, "avg_confidence": avg_conf}
        stats["facts"] = fact_stats

        # Insight counts
        cursor = await self._conn.execute(
            """
            SELECT namespace, COUNT(*) as count
            FROM insights
            GROUP BY namespace
            """
        )
        insight_stats = {}
        async for row in cursor:
            ns, count = row
            insight_stats[ns] = count
        stats["insights"] = insight_stats

        # Vector index stats
        index_stats = {}
        for ns, index in self.vector_indices.items():
            index_stats[ns] = {"vectors": index.ntotal}
        stats["indices"] = index_stats

        return stats

    async def clear_namespace(self, namespace: str) -> bool:
        """Clear all data for a namespace"""
        try:
            async with self._write_lock:
                await self._conn.execute(
                    "DELETE FROM episodes WHERE namespace = ?", (namespace,)
                )
                await self._conn.execute(
                    "DELETE FROM facts WHERE namespace = ?", (namespace,)
                )
                await self._conn.execute(
                    "DELETE FROM insights WHERE namespace = ?", (namespace,)
                )
                await self._conn.commit()

            # Clear vector index
            if namespace in self.vector_indices:
                self.vector_indices[namespace].clear()
                del self.vector_indices[namespace]

            logger.info(f"Cleared namespace {namespace}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear namespace: {e}")
            return False

    async def save_indices(self):
        """Save all vector indices to disk"""
        for namespace, index in self.vector_indices.items():
            index_path = f"{self.db_path}.{namespace}.index"
            index.save_index(index_path)

    async def close(self):
        """Close database connection"""
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        await self.save_indices()

        if self._conn:
            await self._conn.close()

        logger.info("SQLiteDataStore closed")
