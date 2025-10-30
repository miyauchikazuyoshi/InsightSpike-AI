# InsightSpike用語辞書 (Glossary)

## コア概念

### Episode (エピソード)
**定義**: 観測と行動の単位的な記録。1つの経験を表す最小単位。

**構成要素**:
- `text`: テキスト表現
- `vector`: ベクトル表現（埋め込み）
- `metadata`: メタデータ（タイプ、成功/失敗、位置など）
- `c_value`: 信頼度/重要度スコア

**例**:
```python
episode = {
    "text": "Move from (3,4) right: success",
    "vector": [0.2, 0.3, 0.5, ...],
    "metadata": {"type": "movement", "success": True},
    "c_value": 0.8
}
```

### Experience (経験)
**定義**: エピソードに評価やコンテキストが付与された拡張構造。グラフのノードとして扱われる。

**Episodeとの違い**:
- Episode: 生の観測データ
- Experience: エピソードに関連性や価値判断が加わったもの

### Memory (記憶)
**定義**: エピソードとエクスペリエンスの集合体。検索可能な知識ベース。

**種類**:
- Short-term memory: 最近のエピソード
- Long-term memory: 永続化されたエピソード
- Working memory: 現在のタスクに関連するエピソード

### Spike (スパイク)
**定義**: ΔGED < 閾値 かつ ΔIG > 閾値 となる状態。新しい洞察の発見を示す。

**検出条件**:
```python
spike = (delta_ged < ged_threshold) and (delta_ig > ig_threshold)
```

### Insight (洞察)
**定義**: スパイク検出時に生成される新しい理解や関連性。エピソード間の隠れた繋がり。

**生成プロセス**:
1. スパイク検出
2. メッセージパッシング
3. ベクトル集約
4. 新エピソードとして記録

## 評価指標

### geDIG (Generalized Differential Information Gain)
**定義**: `GED - IG`。グラフ編集距離と情報利得の差分。

**解釈**:
- geDIG < 0: 良好な学習（情報利得が編集距離を上回る）
- geDIG > 0: 構造的差異が大きい

### ΔGED (Delta Graph Edit Distance)
**定義**: グラフ構造の変化量。ノード/エッジの追加・削除コスト。

### ΔIG (Delta Information Gain)
**定義**: 情報エントロピーの減少量。知識の整理度合い。

## アーキテクチャ層

### L1: Document Processing
**役割**: テキスト処理とベクトル化
**主要クラス**: `L1DocumentProcessor`

### L2: Memory Management  
**役割**: エピソード管理と検索
**主要クラス**: `L2MemoryManager`

### L3: Graph Reasoning
**役割**: グラフ構築とスパイク検出
**主要クラス**: `L3GraphReasoner`

### L4: Planning & QA
**役割**: 高次推論と応答生成
**主要クラス**: `L4PlanningAssistant`

## インデックス種別

### OptimizedNumpyIndex
**定義**: NumPy実装の高速ベクトル検索。O(k)の計算量。
**用途**: メモリ上での高速類似度検索

### FaissIndex
**定義**: Facebook AI Similarity Searchを使用した大規模ベクトル検索。
**用途**: 大規模データセットでの効率的な検索

### GraphMemorySearch
**定義**: グラフ構造を活用したメッセージパッシング検索。
**用途**: 関連性の深い探索

## 実験用語

### Pure Memory Agent
**定義**: LLMを使わず純粋に記憶駆動で動作するエージェント。
**特徴**: エピソードベクトルの類似度検索のみで行動決定

### Message Passing
**定義**: グラフ上でノード間で情報を伝播させる処理。
**用途**: クエリからの多段階推論

### Edge Re-evaluation  
**定義**: メッセージパッシング後にエッジの重みを再評価する処理。
**効果**: 新しい関連性の発見

## 設定・モード

### LITE Mode
**定義**: 軽量実行モード。L1処理をスキップし、事前計算済みベクトルを使用。
**設定**: `INSIGHTSPIKE_LITE_MODE=1`

### L1 Bypass
**定義**: L1層を迂回し、直接ベクトルを入力する最適化。
**条件**: キャッシュヒット時または事前ベクトル化済みデータ

## データ永続化

### DataStore
**定義**: エピソードとグラフの永続化層。
**実装**:
- `FileSystemDataStore`: ファイルシステムベース
- `MemoryDataStore`: メモリベース（テスト用）

### Snapshot
**定義**: 実験データの特定時点でのバックアップ。
**保存先**: `experiments/{name}/data_snapshots/{timestamp}/`

## その他

### c_value (Confidence Value)
**定義**: エピソードの信頼度/重要度スコア。0.0〜1.0の値。
**用途**: 優先度付き検索、知識の選別

### Contradiction Detection
**定義**: エピソード間の矛盾を検出する処理。
**手法**: 
- 構造的矛盾: グラフ構造の不整合
- 意味的矛盾: ベクトル空間での対立
- 時間的矛盾: 時系列での不整合

### Triangle Inequality
**定義**: 距離空間での三角不等式 `d(A,C) ≤ d(A,B) + d(B,C)`
**用途**: エピソード間の距離関係の整合性チェック

## 命名規則

### ファイル名
- 実験スクリプト: `test_{実験名}.py`
- エージェント: `{名前}_agent.py`
- 結果: `{YYYYMMDD-HHMMSS}_seed{xxxx}/`

### 変数名
- ベクトル: `vec`, `vector`, `embedding`
- 距離/類似度: `dist`, `similarity`, `sim`
- インデックス: `idx`, `index`, `indices`

## 更新履歴

- 2025-08-08: 初版作成
- 基本用語の定義
- 命名規則の明文化