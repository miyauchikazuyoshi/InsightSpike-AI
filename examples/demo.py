from scripts.prepare_episodes import embeddings
from insightspike.data_loader import load_raw_documents
from insightspike.retrieval import SimpleRetrieval
# from insightspike.quantizer import Quantizer  
# → 後で「量子化をかました例」を入れるときに使います
from insightspike.quantizer import Quantizer  

# 1) 実際のファイルからドキュメントを読み込む
docs = load_raw_documents("data/raw")

# 2) 埋め込みを計算
vecs = embeddings(docs, dim=16)

# 3) インデックスを作成
retriever = SimpleRetrieval(vecs)

# 4) クエリに対する検索例
q = "テスト用クエリ"
qvec = embeddings([q], dim=16)[0]

ids, sims = retriever.query(qvec, topk=3)
print("Top results:")
for i, s in zip(ids, sims):
    print(f"  doc#{i} (score={s:.3f}): {docs[i]}")
