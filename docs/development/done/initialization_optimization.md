---
status: active
category: meta
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# InsightSpike 初期化最適化ガイド

## 現在の問題点
- DistilGPT2の初期化に時間がかかる（特に初回実行時）
- 毎回モデルをロードするオーバーヘッド
- 実験の繰り返しが困難

## 実装済みの対策

### 1. モデルの事前ダウンロード
```bash
poetry run python scripts/setup_models.py
```

これにより以下がキャッシュされます：
- DistilGPT2
- GPT2
- sentence-transformers/all-MiniLM-L6-v2
- paraphrase-MiniLM-L6-v2

### 2. Poetry依存関係の最適化
`pyproject.toml`に必要なパッケージを明記：
- transformers ^4.30.0
- sentence-transformers ^2.2.0
- torch 2.2.2（CUDA自動検出）

## 追加の最適化案

### 1. モデルの遅延読み込み
```python
class L4LLMInterface:
    def __init__(self, config):
        self._model = None  # 遅延初期化
        
    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model
```

### 2. モデルのシングルトン化
```python
class ModelCache:
    _instance = None
    _models = {}
    
    @classmethod
    def get_model(cls, model_name):
        if model_name not in cls._models:
            cls._models[model_name] = cls._load_model(model_name)
        return cls._models[model_name]
```

### 3. 環境変数の設定
```bash
# .envファイルに追加
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false  # Fork警告を回避
export TRANSFORMERS_OFFLINE=1  # オフラインモード（キャッシュのみ使用）
```

### 4. 実験用の軽量設定
```python
# config/presets.py に追加
"quick_experiment": {
    "llm": {
        "provider": "mock",  # 初期テストはMockで
        "model": "mock"
    },
    "memory": {
        "max_retrieved_docs": 5  # 少ないドキュメント数
    },
    "graph": {
        "spike_ged_threshold": -999,  # 連続スコアリング
        "spike_ig_threshold": -999
    }
}
```

### 5. プロファイリング用デコレータ
```python
import time
from functools import wraps

def profile_initialization(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger.info(f"{func.__name__} took {duration:.2f}s")
        return result
    return wrapper
```

## 使用例

### 高速実験の実行
```python
# 1. MockLLMで動作確認
config = load_config(preset="quick_experiment")

# 2. 動作確認後、DistilGPT2に切り替え
config.llm.provider = "local"
config.llm.model = "distilgpt2"
```

### バッチ実験の実行
```python
# モデルを一度だけロード
model_cache = ModelCache()

for experiment in experiments:
    # キャッシュからモデルを取得
    model = model_cache.get_model("distilgpt2")
    run_experiment(model, experiment)
```

## まとめ

これらの最適化により：
1. **初回起動**: 〜1分（モデルダウンロード済み）
2. **2回目以降**: 〜10秒（キャッシュから読み込み）
3. **MockLLMテスト**: 〜1秒（即座に開始）

実験の繰り返しが大幅に効率化されます！