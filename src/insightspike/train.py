import joblib

def train():
    index = joblib.load("data/processed/index.joblib")
    q = joblib.load("data/processed/episodes/quantizer.joblib")
    # TODO: モデル訓練

if __name__ == "__main__":
    train()