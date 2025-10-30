---
status: active
category: meta
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# リグレッションテスト計画書

## 概要

メッセージパッシングとエッジ再評価機能の実装後、既存機能が正しく動作することを確認するための包括的なリグレッションテスト計画。

## テストレベル

### 1. ユニットテスト（コンポーネント単位）

#### 1.1 コアモジュール
- [ ] **VectorIntegrator**
  - 洞察ベクトル生成
  - エピソード分岐ベクトル生成
  - 各種集約方法（weighted_mean, mean, max）

- [ ] **MessagePassing**
  - 基本的なメッセージ伝播
  - クエリベクトルの影響
  - 複数イテレーション

- [ ] **EdgeReevaluator**
  - エッジ保持判定
  - 新規エッジ発見
  - エッジ統計計算

#### 1.2 既存アルゴリズム
- [ ] **GraphEditDistance (GED)**
  - 基本的なGED計算
  - エラーハンドリング（Dimension error）
  - 近似計算フォールバック

- [ ] **InformationGain (IG)**
  - 基本的なIG計算
  - クラスタリング方式
  - エントロピー計算

- [ ] **MetricsSelector**
  - アルゴリズム選択ロジック
  - フォールバック動作

### 2. レイヤーテスト（Layer単位）

#### 2.1 Layer2 (Memory Manager)
- [ ] **基本機能**
  - エピソード追加
  - ベクトル検索
  - メモリ統計

- [ ] **検索機能**
  - 標準検索（メッセージパッシングなし）
  - グラフベース検索
  - query_embeddingの返却

#### 2.2 Layer3 (Graph Reasoner)
- [ ] **グラフ構築**
  - 基本的なグラフ構築
  - インクリメンタル更新
  - エッジ作成ロジック

- [ ] **メッセージパッシング統合**
  - 有効/無効の切り替え
  - query_vectorの受け取り
  - エッジ再評価の実行

- [ ] **メトリクス計算**
  - GED/IG計算（メッセージパッシングあり/なし）
  - スパイク検出
  - conflict計算

#### 2.3 Layer4 (LLM Interface)
- [ ] **基本機能**
  - プロンプト構築
  - レスポンス生成
  - キャッシング

- [ ] **洞察ベクトル生成**
  - query_vectorあり/なし
  - VectorIntegrator使用
  - 後方互換性

### 3. 統合テスト（Layer間連携）

#### 3.1 Layer2→Layer3
- [ ] **データフロー**
  - documentの受け渡し
  - query_embeddingの伝播
  - グラフコンテキスト

#### 3.2 Layer3→Layer4
- [ ] **データフロー**
  - graph_analysisの受け渡し
  - query_vectorの伝播
  - メトリクス情報

#### 3.3 MainAgent統合
- [ ] **初期化**
  - 各種config形式のサポート
  - レガシー設定の互換性
  - デフォルト値の適用

- [ ] **サイクル実行**
  - _execute_memory_search
  - _execute_cycle
  - query_embeddingのスコープ

### 4. パイプラインテスト（E2E）

#### 4.1 基本設定パターン

##### Config 1: ベースライン（すべて無効）
```yaml
graph:
  enable_message_passing: false
  enable_graph_search: false
  use_gnn: false
```
- [ ] 知識追加 → 質問 → 応答
- [ ] スパイク検出なし
- [ ] グラフ更新の確認

##### Config 2: メッセージパッシングのみ
```yaml
graph:
  enable_message_passing: true
  message_passing:
    alpha: 0.3
    iterations: 3
  enable_graph_search: false
  use_gnn: false
```
- [ ] メッセージパッシング実行確認
- [ ] エッジ再評価の実行
- [ ] 新規エッジ発見の確認

##### Config 3: グラフ検索のみ
```yaml
graph:
  enable_message_passing: false
  enable_graph_search: true
  use_gnn: false
```
- [ ] グラフベース検索の動作
- [ ] 標準検索との違い
- [ ] マルチホップ検索

##### Config 4: すべて有効
```yaml
graph:
  enable_message_passing: true
  message_passing:
    alpha: 0.5
    iterations: 3
  enable_graph_search: true
  use_gnn: true
```
- [ ] 全機能の協調動作
- [ ] パフォーマンスへの影響
- [ ] エラーなし

#### 4.2 エッジケース

- [ ] **空の知識ベース**
  - 初回質問時の動作
  - グラフなしでのメッセージパッシング

- [ ] **単一エピソード**
  - 自己ループの処理
  - エッジ再評価

- [ ] **大量エピソード（100+）**
  - スケーラビリティ
  - メモリ使用量

### 5. 後方互換性テスト

#### 5.1 設定フォーマット
- [ ] **Dict形式config**
- [ ] **Pydanticモデル形式**
- [ ] **YAMLファイル読み込み**
- [ ] **デフォルト値の適用**

#### 5.2 API互換性
- [ ] **MainAgent.add_knowledge()**
- [ ] **MainAgent.process_question()**
- [ ] **CycleResultオブジェクト**

### 6. エラーケーステスト

#### 6.1 設定エラー
- [ ] 不正なalpha値（<0 or >1）
- [ ] 不正なiterations（負の値）
- [ ] 存在しないプロバイダー

#### 6.2 実行時エラー
- [ ] None値の処理
- [ ] 空配列の処理
- [ ] 次元不一致

## テスト実装優先順位

1. **High Priority**
   - Layer3のメッセージパッシング統合
   - MainAgentのquery_embedding伝播
   - 基本的なパイプラインテスト

2. **Medium Priority**
   - 各種config形式のテスト
   - エッジケーステスト
   - Layer間連携テスト

3. **Low Priority**
   - パフォーマンステスト
   - 大規模データテスト
   - GNN統合テスト

## 成功基準

- すべてのユニットテストがPASS
- 既存機能の動作に変更なし
- 新機能が有効/無効で切り替え可能
- エラーハンドリングが適切
- ドキュメントとの一貫性

## テスト環境

- Python 3.11+
- Poetry環境
- Mock LLMプロバイダー（API依存を避ける）
- 小規模テストデータ

## 実行計画

1. テストコードの作成（2時間）
2. ユニットテスト実行（30分）
3. レイヤーテスト実行（1時間）
4. パイプラインテスト実行（1時間）
5. 結果分析とバグ修正（2時間）

合計予想時間: 6.5時間