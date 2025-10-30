---
status: active
category: meta
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# 包括的修正計画 (2025年1月)

## 1. 問題の整理

### A. Episode作成の問題
- **問題**: MainAgent.add_knowledge()でEpisode作成時に`c`パラメータを使用しているが、Episodeクラスは`confidence`を期待
- **影響**: エピソードの追加が全て失敗する
- **ファイル**: `/src/insightspike/implementations/agents/main_agent.py`

### B. 埋め込みベクトルの形状問題
- **問題**: EmbeddingManager.get_embedding()が(1,384)の形状を返すが、全ての処理は(384,)を期待
- **影響**: 類似度計算が失敗し、グラフのエッジが作成されない
- **ファイル**: `/src/insightspike/processing/embedder.py`

### C. グラフ型の混在問題
- **問題**: NetworkXとPyTorch Geometricのグラフ型が混在し、GED/IG計算が失敗
- **影響**: "Adjacency matrix not square"エラーが発生
- **ファイル**: 複数（GraphAnalyzer, ProperDeltaGED, InformationGain）

### D. 設定パイプラインの問題
- **問題**: dict形式とPydantic形式の設定が混在
- **影響**: 設定の受け渡しで型エラーが発生
- **ファイル**: 複数のレイヤー実装

### E. DataStore依存の問題
- **問題**: MainAgentはDataStoreなしでは動作しない
- **影響**: 簡単なテストが困難
- **ファイル**: `/src/insightspike/implementations/agents/main_agent.py`

## 2. 修正戦略

### Phase 1: 即時修正（コアの動作を回復）
1. **Episode作成の修正**
   - MainAgent.add_knowledge()を修正
   - `c=c_value` → `confidence=c_value`

2. **埋め込み形状の修正**
   - EmbeddingManager.get_embedding()の出力を常に1D化
   - StandardizedEmbedderクラスの作成

3. **基本的なパッチの統合**
   - 既存のapply_fixes.pyにEpisode修正を追加

### Phase 2: グラフ処理の修正
1. **GraphTypeAdapterの完全実装**
   - NetworkX ↔ PyG の相互変換
   - 全てのグラフ操作の前で型を統一

2. **GED/IG計算の修正**
   - 入力グラフの型チェックと変換
   - エラーハンドリングの改善

### Phase 3: 設定システムの統一
1. **ConfigNormalizerの拡張**
   - 全ての設定形式を統一的に処理
   - 後方互換性の維持

2. **各レイヤーの設定処理を統一**
   - dict形式を内部でPydantic形式に変換

### Phase 4: テストと検証
1. **統合テストの作成**
   - 全パイプラインのエンドツーエンドテスト
   - 各修正の効果を検証

2. **実験の再実行**
   - 数学的概念進化実験
   - スパイク検出の検証

## 3. 実装手順

### Step 1: MainAgent.add_knowledge()の修正
```python
# main_agent.py の該当箇所を修正
episode = Episode(
    text=text,
    vec=embedding,
    confidence=c_value,  # c → confidence に変更
    timestamp=time.time(),
    metadata={"c_value": c_value}
)
```

### Step 2: パッチシステムの更新
```python
# apply_fixes.py に追加
def apply_episode_creation_fix():
    """Fix Episode creation in MainAgent"""
    from ..implementations.agents.main_agent import MainAgent
    # ... パッチ実装
```

### Step 3: 統合パッチファイルの作成
```python
# comprehensive_fixes.py
def apply_comprehensive_fixes():
    """Apply all fixes in correct order"""
    apply_episode_creation_fix()
    apply_embedder_fix()
    apply_graph_type_fix()
    apply_config_normalization_fix()
```

## 4. 検証計画

### A. ユニットテスト
- Episode作成のテスト
- 埋め込み形状のテスト
- グラフ型変換のテスト

### B. 統合テスト
- MainAgentの知識追加フロー
- 質問処理フロー
- スパイク検出フロー

### C. 実験による検証
- 基本的な数学知識での動作確認
- スパイク検出の確認
- DataStore保存の確認

## 5. リスクと対策

### リスク
1. パッチの適用順序による予期しない動作
2. 既存の動作への影響
3. パフォーマンスの低下

### 対策
1. パッチ適用のログ記録
2. フォールバック機構の実装
3. パフォーマンス測定の実施

## 6. タイムライン

1. **即時**: Episode作成とembedding形状の修正（30分）
2. **短期**: グラフ型統一の実装（1時間）
3. **中期**: 設定システムの統一（2時間）
4. **検証**: テストと実験（1時間）

## 7. 成功基準

- [ ] MainAgent.add_knowledge()が正常に動作
- [ ] 埋め込みベクトルが常に(384,)形状
- [ ] GED/IG計算でエラーが発生しない
- [ ] 数学実験でスパイクが検出される
- [ ] DataStoreに正しくデータが保存される

## 8. 次のステップ

1. この計画の承認を得る
2. Phase 1の即時修正を実施
3. 基本動作の確認
4. 段階的に残りのPhaseを実施