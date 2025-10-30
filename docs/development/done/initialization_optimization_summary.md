---
status: active
category: meta
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# Initialization Optimization Summary

## Problem Identified
実験を実行するたびに、LLMの初期化（特に`from_pretrained`）に時間がかかり、開発サイクルが遅くなっていた。

## Root Cause
- 毎回MainAgentが作成されるたびに、新しいLLMインスタンスが作成されていた
- `AutoModelForCausalLM.from_pretrained()`が毎回実行され、重いモデルファイルをディスクから読み込んでいた
- アプリケーションのライフサイクルとオブジェクトのライフサイクルが一致していた

## Solution Implemented

### 1. LLMProviderRegistry (Singleton Pattern)
`src/insightspike/implementations/layers/layer4_llm_interface.py`に追加:
- プロバイダーとモデル名の組み合わせでインスタンスをキャッシュ
- スレッドセーフな実装
- 一度初期化されたLLMは再利用される

### 2. Pre-warming in __main__.py
`src/insightspike/__main__.py`に追加:
- アプリケーション起動時に使用予定のモデルを事前にロード
- Clean provider（フォールバック用）を常にプリロード
- 設定に基づいてLocal/OpenAIモデルもプリロード

### 3. MainAgent Optimization
`src/insightspike/implementations/agents/main_agent.py`を更新:
- 既に初期化済みのLLMは再初期化をスキップ
- キャッシュされたインスタンスを使用

### 4. Configuration Option
`src/insightspike/config/models.py`に追加:
- `pre_warm_models`フラグ（デフォルト: True）
- 必要に応じてプリウォーミングを無効化可能

## Expected Benefits

1. **初回起動時**: 従来通りの時間がかかる（モデルロードが必要）
2. **2回目以降の実験**: 
   - LLMの再ロードが不要
   - 実験の開始が大幅に高速化（数秒 → ミリ秒）
   - 思考の流れを妨げない高速なイテレーション

## Usage

### テストスクリプト
```bash
python scripts/test_initialization_speed.py
```

### プリウォーミングを無効化する場合
`config.yaml`に追加:
```yaml
pre_warm_models: false
```

## Implementation Details

### Cache Key
- Provider type + Model name でユニークに識別
- 例: ("local", "distilgpt2"), ("openai", "gpt-4")

### Thread Safety
- `threading.Lock()`で同期制御
- 複数スレッドからの同時アクセスに対応

### Backward Compatibility
- 既存のコードは変更不要
- `use_cache=False`オプションで従来の動作も可能
- DependencyFactoryの既存のキャッシュ機構と協調動作

## Next Steps

1. 実際の実験での速度改善を測定
2. 必要に応じてキャッシュの有効期限機能を追加
3. メモリ使用量のモニタリング機能を検討