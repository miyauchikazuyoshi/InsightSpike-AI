# ハイブリッドエピソード分割実装まとめ

## 実装完了内容

### 1. HybridEpisodeSplitter (`src/insightspike/episodic/hybrid_episode_splitter.py`)

洞察検出と同じ二段階評価フローを実装：
- **メッセージパッシング**によるベクトル生成
- **LLM意味解析**による重み付け（オプション）
- **品質評価**（Quality + GED/IG改善）

#### 主要メソッド：
- `split_episode()`: メインエントリーポイント
- `_generate_message_passing_vectors()`: グラフ構造を考慮したベクトル生成
- `_analyze_with_llm()`: LLMによる意味的分析
- `_evaluate_split_quality()`: GED/IG改善度の評価

### 2. CachedMemoryManagerへの統合

`split_episode()`と`merge_episodes()`メソッドを追加：
- DataStoreベースの永続化
- 親子関係のメタデータ保持
- キャッシュの効率的な管理

### 3. メッセージパッシングの重み配分

```python
# ベクトル生成の重み配分
親ベクトル: 50%      # 元の文脈を最重視
近傍ノード: 30%      # グラフ構造の文脈
基本ベクトル: 20%    # セグメント固有の意味
```

### 4. 品質評価基準

以下の3つの基準をすべて満たした場合に分割：
1. **品質スコア** ≥ 0.6（ベクトル多様性、長さバランス、重みエントロピー）
2. **GED改善** ≥ 0.1（グラフ構造の単純化）
3. **IG改善** ≥ 0.05（情報利得の向上）

## テスト結果

`test_splitter_direct.py`での検証結果：
- 親ベクトルとの類似度: 0.85-0.89（適切な継承）
- セグメント間類似度: 0.79-0.86（適度な多様性）
- 品質スコア: 0.80（高品質）

## 重要な設計判断

### 1. エッジベースの統合分裂
- テキスト長ベースの分割を廃止
- グラフ構造の変化（ΔGED/ΔIG）で判断
- エピソード境界の動的定義

### 2. ハイブリッドアプローチ
- 場合分けではなく**両方実行**
- メッセージパッシング結果をベースに
- LLM分析で意味的調整

### 3. 洞察検出との統一性
- 同じ評価フロー（品質→最適化）
- 再利用可能なアルゴリズム
- 一貫した品質保証

## 今後の拡張可能性

1. **適応的分割**
   - グラフ密度に応じた重み調整
   - ドメイン特化の境界検出

2. **階層的分割**
   - 再帰的な分割による階層構造
   - 概念の粒度管理

3. **学習的改善**
   - 分割成功/失敗パターンの学習
   - パラメータの自動調整

## 使用例

```python
from insightspike.implementations.layers.cached_memory_manager import CachedMemoryManager
from insightspike.core.datastore import DataStore

# セットアップ
datastore = DataStore(storage_path="data/episodes.db")
memory_manager = CachedMemoryManager(datastore=datastore)

# エピソード追加
episode_id = memory_manager.add_episode(
    text="長いテキスト...",
    c_value=0.8
)

# 分割実行
split_ids = memory_manager.split_episode(
    episode_id=episode_id,
    graph=knowledge_graph,  # 現在のナレッジグラフ
    llm_provider=llm,       # オプション
    force_split=False       # 品質チェックを有効化
)

# 結果確認
for split_id in split_ids:
    episode = memory_manager.get_episode(split_id)
    print(f"Split: {episode.text[:50]}...")
    print(f"C-value: {episode.c}")
```

## まとめ

このハイブリッドエピソード分割により：
- **グラフ構造を改善**しながら
- **意味的な一貫性を保持**した
- **動的なエピソード境界定義**

が実現されました。エピソードの統合分裂がエッジベースの評価に一本化され、より本質的な知識組織化が可能になりました。