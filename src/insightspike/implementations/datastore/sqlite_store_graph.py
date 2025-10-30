"""
SQLite Data Store - Graph Operations Extension
==============================================

Graph-related operations for SQLiteDataStore.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite
import numpy as np

logger = logging.getLogger(__name__)

# This file contains the graph-related methods for SQLiteDataStore
# To be merged with sqlite_store.py


class SQLiteDataStoreGraphMixin:
    """Mixin for graph operations in SQLiteDataStore"""

    # ========== Graph Operations ==========

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

    # ========== Additional Operations ==========

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

    # ========== Sync method implementations ==========

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
        results = asyncio.run(
            self.search_episodes_by_vector(query_vector, k, 0.0, namespace)
        )

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
            asyncio.run(self._save_graph(graph_data, graph_id, namespace))
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
        return asyncio.run(self._load_graph(graph_id, namespace))

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
