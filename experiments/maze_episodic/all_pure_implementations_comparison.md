# 全チートなし実装の比較

## 発見された純粋な実装

### 1. experiments/maze_episodic/
- **pure_episodic_navigator.py** - 純粋エピソード記憶（6次元）
- **pure_episodic_donut.py** - ドーナツ検索付き（7次元、訪問回数追加）
- **true_gedig_flow_navigator.py** - 正確なgeDIG定義実装
- **true_pure_gedig_navigator.py** - 真の純粋geDIG実装
- **pure_episodic_v2.py** - 改良版純粋エピソード
- **pure_episodic_true_flow.py** - 正確なフロー実装

### 2. experiments/maze-sota-comparison/
- **true_episodic_2hop.py** - GED減少検出による構造的洞察
- **pure_multihop_deep_message.py** - 深層メッセージパッシング
- **truly_pure_navigator.py** - 完全に純粋な実装
- **true_episodic_gedig.py** - エピソード的geDIG
- **pure_multihop_navigator.py** - 純粋マルチホップ
- **pure_local_embeddings.py** - ローカル埋め込み

### 3. src/insightspike/maze_experimental/navigators/
- **pure_gediq_navigator.py** - geDIGの基本実装
- **simple_gediq_navigator.py** - シンプルgeDIG
- **blind_experience_navigator.py** - 視覚情報なし

## 特徴比較

### メモリ構造の違い

| 実装 | 次元 | 特徴 |
|-----|------|------|
| pure_episodic_navigator | 6D | 位置、方向、結果、トポロジー、ゴール信号 |
| pure_episodic_donut | 7D | 6D + 訪問回数 |
| true_gedig_flow | 6D | 厳密な正規化、離散方向エンコーディング |
| true_episodic_2hop | 7D | Noneを許容、グラフ構造重視 |
| pure_gediq_navigator | 可変 | MemoryNodeクラス、IG値保持 |

### 検索戦略

1. **全エピソード検索型**（初期実装）
   - O(n²)の計算量問題
   - 小規模迷路のみ成功

2. **ドーナツ検索型**
   - 内径・外径による効率化
   - 中規模迷路まで対応

3. **グラフベース検索**
   - NetworkXによる構造解析
   - GED減少パターン検出

### 成功パターン

| 迷路サイズ | 成功した純粋実装 | 特徴 |
|-----------|----------------|------|
| 5x5 | ほぼ全て | 基本的な記憶ベース探索で十分 |
| 15x15 | pure_episodic系、true_episodic_2hop | マルチホップ or ドーナツ検索必須 |
| 25x25 | true_episodic_2hop、一部のドーナツ実装 | 効率的な検索が必須 |
| 50x50 | なし（タイムアウト） | 計算量の壁 |

### 独自の革新点

1. **pure_episodic_navigator**
   - 適応的1/2/3ホップ選択
   - 次元別メッセージパッシング率

2. **true_episodic_2hop**
   - GED減少による構造的洞察検出
   - 行き止まりパターンの明示的認識

3. **pure_gediq_navigator（src内）**
   - MemoryNodeクラスでIG値を保持
   - 温度パラメータk_igで探索調整

4. **blind_experience_navigator**
   - 視覚情報完全排除
   - 純粋な経験のみで探索

## 共通の課題

1. **初期探索問題**
   - エピソード不足で検索失敗
   - 内径0のドーナツ検索で対処

2. **ゴールエピソード選択**
   - 常にゴールを選んでしまう
   - 適切な重み付けが必要

3. **スケーリング**
   - 50x50では全て失敗
   - 根本的な効率化が必要

## 結論

チートなし実装は**少なくとも15種類以上**存在し、それぞれ異なるアプローチで純粋な記憶ベース探索を実現しています。小〜中規模迷路では成功例があり、理論的健全性が証明されています。