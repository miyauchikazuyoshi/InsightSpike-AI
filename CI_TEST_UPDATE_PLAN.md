# CI/テスト更新計画

## 現状の問題点

### 1. CI設定 (`.github/workflows/ci.yml`)
- 存在しない実験ファイルを参照
- Phase 2/3の新実装に対応していない

### 2. テストファイル
- C値を参照する古いモック（`tests/conftest.py`）
- 古いLayer2/Layer3実装をテスト
- 新しいスケーラブル実装のテストが不足

## 更新が必要なファイル

### 高優先度（動作に影響）
1. **tests/conftest.py**
   - MockMainAgentからc_value引数を削除
   - MockMemoryからc_value関連を削除

2. **.github/workflows/ci.yml**
   - 存在しない実験ファイルの参照を削除
   - 新しいPhase 2/3テストを追加

### 中優先度（整合性）
1. **tests/unit/test_layer2_memory_manager.py**
   - GraphCentricMemoryManagerのテストに更新
   - C値なしの統合ロジックをテスト

2. **tests/unit/test_layer3_graph_pyg.py**
   - ScalableGraphBuilderのテストを追加
   - 階層的グラフ構造のテスト

### 低優先度（クリーンアップ）
1. **古い実験ファイル**
   - experiment_1～4のC値参照を更新
   - または`experiments/archive/`に移動

## 新規テストの追加

### Phase 2: スケーラブルグラフ
```python
# tests/unit/test_scalable_graph_builder.py
- FAISSベースの近傍探索
- O(n log n)のパフォーマンス確認
- Top-k設定のテスト
```

### Phase 3: 階層的グラフ
```python
# tests/unit/test_hierarchical_graph.py
- 3層構造の構築
- O(log n)検索のテスト
- 動的ドキュメント追加
```

### 統合テスト
```python
# tests/integration/test_graph_centric_system.py
- C値なしのエピソード管理
- グラフベースの統合/分裂
- 動的重要度計算
```

## 実行手順

1. **即座に修正すべき**
   - CI設定から存在しないファイル参照を削除
   - conftest.pyのモックを更新

2. **次のPRで対応**
   - 新しい実装に対応したユニットテスト追加
   - 古いテストの更新または削除

3. **将来的な整理**
   - 実験ファイルのアーカイブ化
   - ドキュメントの整合性確認

## テスト戦略

### 保持すべきテスト
- 基本的なエピソード追加/検索
- グラフ構造の整合性
- メモリ効率性

### 追加すべきテスト
- スケーラビリティ（1000, 10000エピソード）
- 階層的検索の正確性
- グラフベース統合の動作

### 削除可能なテスト
- C値関連のすべてのテスト
- 古いGraphBuilder実装のテスト
- Phase 1の単純な実装テスト