import importlib

MODULES = [
    "insightspike.train",
    "insightspike.quantizer",
    "insightspike.retrieval",
    "insightspike.data_loader",
    "insightspike.predict",
]

def test_import_all_modules():
    for name in MODULES:
        module = importlib.import_module(name)
        assert module is not None, f"モジュール {name} がインポートできません"
