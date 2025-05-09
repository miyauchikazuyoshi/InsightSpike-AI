from insightspike.data_loader import load_sample_data
from insightspike.quantizer    import Quantizer
from insightspike.retrieval    import Retriever
from insightspike.predict      import Predictor

def main():
    # 1) サンプルデータ読み込み
    data = load_sample_data("examples/sample_input.json")

    # 2) ベクトル量子化
    q = Quantizer(...)
    episodes = q.quantize(data)

    # 3) 検索・RAG
    r = Retriever(...)
    context = r.retrieve(episodes)

    # 4) 推論
    p = Predictor(...)
    result = p.predict(context)

    print("=== PoC Demo Result ===")
    print(result)

if __name__ == "__main__":
    main()
