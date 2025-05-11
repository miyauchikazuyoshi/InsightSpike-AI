from typing import List, Any

import numpy as np

class SimpleRetrieval:
    def __init__(self, embeddings):
        # embeddings: np.ndarray [N×D]
        # ここでは単純に保存
        self.emb = embeddings

    def query(self, vec, topk=5):
        # コサイン類似度で上位 topk の index を返す
        normed = self.emb / np.linalg.norm(self.emb, axis=1, keepdims=True)
        qn = vec / np.linalg.norm(vec)
        sims = normed.dot(qn)
        ids = np.argsort(-sims)[:topk]
        return ids, sims[ids]


def build_index(docs: List[str]) -> Any:
    """ドキュメントから検索インデックスを構築"""
    index = None
    # TODO: 実装
    return index


def search(query: str, index: Any, top_k: int = 5) -> List[str]:
    """インデックスを用いた検索結果トップKを返す"""
    results = []
    # TODO: 実装
    return results