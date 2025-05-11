from insightspike.data_loader import load_raw_documents
from insightspike.retrieval import build_index
from insightspike.quantizer import Quantizer

import numpy as np

def embeddings(texts, dim=16):
    # texts の数 × dim のランダムベクトルを返す
    return np.random.RandomState(0).randn(len(texts), dim).astype(np.float32)

if __name__ == "__main__":
    docs = load_raw_documents("data/raw/")
    index = build_index(docs)
    # embeddings = ... # ここで埋め込み計算
    q = Quantizer(codebook=...)
    q_indices = q.encode(embeddings)
    # 保存
    import joblib
    joblib.dump(index, "data/processed/index.joblib")
    joblib.dump(q, "data/processed/episodes/quantizer.joblib")