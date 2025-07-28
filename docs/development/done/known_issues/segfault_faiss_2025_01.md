# Known Issue: FAISS Segmentation Fault

## Issue Description

セグメンテーションフォルトが発生する場合があります。

```
Fatal Python error: Segmentation fault
...
File "/src/insightspike/implementations/layers/scalable_graph_builder.py", line 211 in _build_from_scratch
File "/faiss/swigfaiss_avx2.py", line 2688 in search
```

## Root Cause

FAISSのインデックス検索時にメモリアクセス違反が発生しています。

### 詳細なスタックトレース
- `scalable_graph_builder.py:211` - `_build_from_scratch`メソッド内
- FAISSの`search`メソッド呼び出し時
- 複数のテストを連続実行した際に発生

## Workaround

### 1. 個別テスト実行
```bash
# 個別に実行すると成功
poetry run pytest tests/integration/test_config_quick_check.py -k minimal -v
poetry run pytest tests/integration/test_config_quick_check.py -k multihop -v
```

### 2. FAISS無効化（一時的）
config.yamlで:
```yaml
graph:
  use_faiss: false  # FAISSを無効化
```

## Long-term Solution

1. **FAISSのバージョン確認**
   - 現在のバージョンとの互換性確認
   - 必要に応じてダウングレード

2. **メモリ管理の改善**
   - FAISSインデックスの適切な初期化
   - インデックスサイズの制限

3. **代替実装**
   - FAISSの代わりにNumPyベースの実装
   - より安定したベクトル検索ライブラリの使用

## Impact

- パフォーマンステストには影響
- 機能的には個別実行で問題なし
- CI/CDでは注意が必要

## Status

- 調査中
- 機能への影響は限定的
- Phase 4のメモリ最適化で対処予定