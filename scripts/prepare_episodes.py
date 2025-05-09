from insightspike.data_loader import load_raw_documents
from insightspike.retrieval import build_index
from insightspike.quantizer import Quantizer

def embeddings(texts):
    # TODO: ここに Embedding モデル呼び出しを書く
    return []

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