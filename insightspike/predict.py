import joblib

def predict(query: str):
    index = joblib.load("data/processed/index.joblib")
    q = joblib.load("data/processed/episodes/quantizer.joblib")
    # TODO: L1で検索 → L2で量子化処理 → 出力整形

if __name__ == "__main__":
    print(predict("example query"))