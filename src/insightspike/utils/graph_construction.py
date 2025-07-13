"""
Graph Construction Utilities
===========================

Build knowledge graphs from documents and episodes.
"""

import logging
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build knowledge graphs from documents"""

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def build_from_documents(self, documents: List[Dict[str, Any]]) -> nx.Graph:
        """Build a knowledge graph from documents"""

        G = nx.Graph()

        # Add nodes for each document
        for i, doc in enumerate(documents):
            node_id = f"doc_{i}"

            # Extract text and metadata
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})

            # Generate embedding
            embedding = self.embedding_model.encode(text)

            # Extract concepts (simple keyword extraction for now)
            concepts = self._extract_concepts(text)

            # Add node with attributes
            G.add_node(
                node_id,
                text=text[:200],  # Store truncated text
                embedding=embedding,
                concepts=concepts,
                category=metadata.get("category", "general"),
                importance=metadata.get("importance", 0.5),
                timestamp=metadata.get("timestamp", 0),
            )

        # Add edges based on similarity
        self._add_similarity_edges(G)

        # Add concept-based edges
        self._add_concept_edges(G)

        return G

    def _extract_concepts(self, text: str) -> List[str]:
        """Simple concept extraction from text"""
        # This is a placeholder - in production, use NER or keyword extraction
        concepts = []

        # Look for key terms
        key_terms = [
            "entropy",
            "information",
            "thermodynamic",
            "energy",
            "system",
            "quantum",
            "probability",
            "disorder",
            "state",
            "microstate",
            "conservation",
            "transformation",
        ]

        text_lower = text.lower()
        for term in key_terms:
            if term in text_lower:
                concepts.append(term)

        return concepts

    def _add_similarity_edges(self, G: nx.Graph, threshold: float = 0.3):
        """Add edges based on embedding similarity"""

        nodes = list(G.nodes())

        for i, node1 in enumerate(nodes):
            for node2 in nodes[i + 1 :]:
                # Get embeddings
                emb1 = G.nodes[node1]["embedding"]
                emb2 = G.nodes[node2]["embedding"]

                # Calculate cosine similarity
                similarity = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2)
                )

                if similarity > threshold:
                    G.add_edge(node1, node2, weight=similarity, type="similarity")

    def _add_concept_edges(self, G: nx.Graph):
        """Add edges based on shared concepts"""

        nodes = list(G.nodes())

        for i, node1 in enumerate(nodes):
            for node2 in nodes[i + 1 :]:
                # Get concepts
                concepts1 = set(G.nodes[node1]["concepts"])
                concepts2 = set(G.nodes[node2]["concepts"])

                # Check for shared concepts
                shared = concepts1.intersection(concepts2)

                if shared:
                    # Weight based on number of shared concepts
                    weight = len(shared) / max(len(concepts1), len(concepts2))

                    # Add edge if not already present
                    if not G.has_edge(node1, node2):
                        G.add_edge(
                            node1,
                            node2,
                            weight=weight,
                            type="concept",
                            shared_concepts=list(shared),
                        )
                    else:
                        # Update weight if concept connection is stronger
                        current_weight = G[node1][node2]["weight"]
                        if weight > current_weight:
                            G[node1][node2]["weight"] = weight
                            G[node1][node2]["shared_concepts"] = list(shared)

    def add_query_node(
        self, G: nx.Graph, query: str, query_embedding: Optional[np.ndarray] = None
    ) -> str:
        """Add a query as a special node in the graph"""

        query_node_id = "QUERY"

        # Generate embedding if not provided
        if query_embedding is None:
            query_embedding = self.embedding_model.encode(query)

        # Extract concepts from query
        concepts = self._extract_concepts(query)

        # Add query node
        G.add_node(
            query_node_id,
            text=query,
            embedding=query_embedding,
            concepts=concepts,
            category="query",
            node_type="query",
            color="yellow",  # Visual indicator
        )

        # Connect to relevant nodes
        self._connect_query_to_graph(G, query_node_id)

        return query_node_id

    def _connect_query_to_graph(self, G: nx.Graph, query_node_id: str):
        """Connect query node to relevant nodes in the graph"""

        query_embedding = G.nodes[query_node_id]["embedding"]
        query_concepts = set(G.nodes[query_node_id]["concepts"])

        connections = []

        for node in G.nodes():
            if node == query_node_id:
                continue

            # Similarity-based connection
            node_embedding = G.nodes[node]["embedding"]
            similarity = np.dot(query_embedding, node_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding)
            )

            # Concept-based boost
            node_concepts = set(G.nodes[node]["concepts"])
            shared_concepts = query_concepts.intersection(node_concepts)
            concept_boost = len(shared_concepts) * 0.1

            # Combined score
            score = similarity + concept_boost

            if score > 0.2:  # Threshold for connection
                connections.append((node, score))

        # Add top connections
        connections.sort(key=lambda x: x[1], reverse=True)
        for node, score in connections[:5]:  # Top 5 connections
            G.add_edge(query_node_id, node, weight=score, type="query_connection")
