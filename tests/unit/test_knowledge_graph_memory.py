import numpy as np
from insightspike.core.learning.knowledge_graph_memory import KnowledgeGraphMemory


def test_add_nodes():
    kg = KnowledgeGraphMemory(embedding_dim=4, similarity_threshold=0.0)
    emb1 = np.array([1, 0, 0, 0], dtype=np.float32)
    emb2 = np.array([0, 1, 0, 0], dtype=np.float32)
    kg.add_episode_node(emb1, 0)
    kg.add_episode_node(emb2, 1)

    assert kg.graph.x.size(0) == 2
    # similarity threshold 0.0 ensures an edge between nodes
    assert kg.graph.edge_index.size(1) >= 2
