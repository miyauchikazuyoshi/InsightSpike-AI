# InsightSpike パイプライン問題分析

## 発見された主要な問題

### 1. Embedding形状の不一致
**問題**: 
- HuggingFaceEmbedder.get_embedding() → (1, 384) shape
- Episode, search_episodes, build_graph → (384,) shape expected

**影響**:
- 類似度計算でエラー: `shapes (1,384) and (1,384) not aligned`
- グラフにエッジが作成されない → GED/IG値が常に0
- スパイク検出が機能しない

**修正箇所**:
1. HuggingFaceEmbedderの出力を修正
2. CachedMemoryManager.add_episodeでflatten処理
3. Episode作成時にvecの形状チェック

### 2. グラフ型の混在
**問題**:
- ScalableGraphBuilder → NetworkX Graph
- L3GraphReasoner.previous_graph → PyTorch Geometric Data
- ProperDeltaGED, InformationGain → NetworkX期待、PyG Data受信

**影響**:
- "Adjacency matrix not square" エラー
- GED/IG計算の失敗

**修正箇所**:
1. PyGAdapter使用の徹底
2. グラフ型変換の統一
3. GraphAnalyzerのインターフェース修正

### 3. Config処理の不整合
**問題**:
- LLMProviderRegistry.get_instance() → dict.providerアクセス
- LLMConfig → prompt_style, use_simple_prompt非対応
- dict/Pydantic混在処理

**影響**:
- LLMプロバイダー初期化エラー
- 設定が正しく伝播しない

**修正箇所**:
1. LLMProviderRegistryのdict処理追加
2. L4LLMInterfaceの属性保存
3. Config変換処理の統一

## データフロー図

```
1. add_knowledge(text)
   ↓
2. embedder.get_embedding(text) → (1,384) ❌
   ↓
3. Episode(vec=(1,384)) ❌
   ↓
4. datastore.save_episodes()
   ↓
5. L3GraphReasoner.analyze_documents()
   ↓
6. ScalableGraphBuilder.build_graph() → No edges ❌
   ↓
7. GraphAnalyzer.calculate_metrics() → GED=0, IG=0 ❌
   ↓
8. No spike detected ❌
```

## 修正優先順位

1. **高**: Embedding形状修正（根本原因）
2. **高**: グラフ型統一（計算エラー）
3. **中**: Config処理統一（機能制限）
4. **低**: エラーハンドリング改善

## テスト戦略

1. 単体テスト
   - Embedder出力形状
   - Episode作成
   - グラフ構築
   - 類似度計算

2. 統合テスト
   - add_knowledge → グラフ作成
   - process_question → メトリクス計算
   - スパイク検出フロー

3. リグレッションテスト
   - 既存の実験スクリプト
   - 異なる設定での動作確認