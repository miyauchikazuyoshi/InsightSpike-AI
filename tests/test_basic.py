import importlib

def test_train_callable_exists():
    """insightspike.train モジュールに train 関数が定義されていることを確認する"""
    module = importlib.import_module("insightspike.train")
    assert hasattr(module, "train"), "train.py に train() が見つかりません"
    assert callable(module.train), "train は呼び出し可能であるべきです"
