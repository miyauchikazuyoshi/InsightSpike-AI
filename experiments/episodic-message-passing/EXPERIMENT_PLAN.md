# 次世代geDIG実験計画（更新版）

## 新しい方向性：メインコードの洞察ベクトル活用

### ユーザー提案（優先度：高）
> 洞察ベクトル生成まではメインコードでいけるでしょ。洞察エピソード内の移動方向成分だけ切り出して、4方向に正規化したら、自律移動になるんじゃない？

**核心アイデア：**
- メインコードの洞察ベクトル生成機能（384次元）を活用
- エピソード内の移動方向成分を抽出
- 4方向（上下左右）に正規化して自律移動を実現

### 実装アプローチ
```python
# メインコードのMainAgentを使って洞察生成
agent = MainAgent(config, datastore)
agent.add_knowledge("迷路探索の知識...")

# 洞察ベクトルから移動方向を抽出
def extract_movement_direction(insight_vector, episode_content):
    # エピソードの移動情報
    from_pos = episode_content['from']
    to_pos = episode_content['to']
    
    # 移動ベクトル（4方向に正規化）
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    
    # 上下左右への確率分布に変換
    movement_probs = softmax([
        insight_vector @ up_template,
        insight_vector @ right_template,
        insight_vector @ down_template,
        insight_vector @ left_template
    ])
    
    return movement_probs
```

## 背景：なぜ複数アプローチか？

### 現在の問題
1. **純粋な勾配ベース**: 局所的すぎて性能悪化（カバレッジ12.2%）
2. **経験ベース**: そこそこ良いが頭打ち（カバレッジ26.8%）
3. **ループ問題**: 同じ場所を何度も訪問（最大530回）
4. **実験コードとメインコードの分離**: 統合の機会を逃している

### 3つのアプローチ

#### 1. メインコード統合アプローチ（NEW - 最優先）
- MainAgentの洞察ベクトル生成を活用
- 移動方向成分の抽出と4方向正規化
- 既存の高度な機能（geDIG計算、スパイク検出）を再利用

#### 2. ドーナツ検索アプローチ
```
  遠すぎる（到達困難）
    ↓
  ○ ○ ○ ○ ○  ← 外径（探索限界）
  ○ ● ● ● ○
  ○ ● 現在 ● ○  ← 内径（既知領域）
  ○ ● ● ● ○
  ○ ○ ○ ○ ○  ← 理想的な探索範囲
```

- **内径**: 既に知っている近い場所を除外
- **外径**: 遠すぎて現実的でない場所を除外
- **ドーナツ領域**: 「適度に新しい」探索対象

#### 3. 5次元コンパクトベクトルアプローチ
- 既に26.8%のカバレッジ達成
- メモリ効率的（1/77のサイズ）
- 迷路タスクに特化

## 実験計画（更新版）

### Phase 1: メインコード統合実験（最優先 - 3日）

**目的：** MainAgentの洞察ベクトルを活用した自律移動の実現

```python
class InsightBasedNavigator:
    def __init__(self, main_agent: MainAgent):
        self.agent = main_agent
        self.direction_templates = {
            'up': np.array([0, -1, 0, 0, ...]),    # 上方向テンプレート
            'right': np.array([1, 0, 0, 0, ...]),  # 右方向テンプレート
            'down': np.array([0, 1, 0, 0, ...]),   # 下方向テンプレート
            'left': np.array([-1, 0, 0, 0, ...])   # 左方向テンプレート
        }
    
    def decide_action(self, current_state):
        # 現在の状態を自然言語化
        query = f"位置({current_state.x}, {current_state.y})から次にどの方向に進むべきか？"
        
        # MainAgentで洞察生成
        result = self.agent.process_question(query)
        
        # 洞察ベクトルから移動方向を抽出
        insight_vector = result.insight_vector  # 384次元
        
        # 各方向への類似度計算
        direction_scores = {}
        for direction, template in self.direction_templates.items():
            score = cosine_similarity(insight_vector, template)
            direction_scores[direction] = score
        
        # ソフトマックスで確率化
        return softmax(direction_scores)
```

**評価指標：**
- メインコードとの統合容易性
- 洞察ベクトルの方向成分抽出精度
- 実際の迷路探索性能

### Phase 2: ドーナツ検索実験（1週間）

**目的：** 適度な新規性を持つ探索対象の選択

```python
class DonutGedigNavigator:
    def __init__(self):
        self.inner_radius = 0.3  # 内径（既知領域）
        self.outer_radius = 0.7  # 外径（探索限界）
        
    def find_donut_targets(self, current_vec):
        # 全エピソードから距離計算
        distances = [cosine_distance(current_vec, ep.vector) 
                    for ep in self.episodes]
        
        # ドーナツ内のエピソードを選択
        donut_episodes = [ep for ep, d in zip(self.episodes, distances)
                         if self.inner_radius < d < self.outer_radius]
        
        return donut_episodes
```

**最適化項目：**
- 内径・外径の動的調整
- FAISSによる高速化
- 他手法とのハイブリッド化

### Phase 3: 統合評価（3日）

**3つのアプローチの比較：**

| アプローチ | 利点 | 欠点 | 期待性能 |
|-----------|------|------|----------|
| メインコード統合 | 既存機能活用、理論的整合性 | 計算コスト高 | 未知数 |
| ドーナツ検索 | 探索効率、パラメータ調整可能 | 実装複雑 | 40-60% |
| 5次元コンパクト | 実績あり、高速 | 拡張性低 | 26.8% |

**ハイブリッド実験：**
```python
# 3つのアプローチを統合
final_score = (
    0.4 * insight_based_score +    # メインコード由来
    0.3 * donut_search_score +      # ドーナツ検索
    0.3 * compact_vector_score      # 5次元実績
)

```

## 実装の優先順位（更新版）

### 今すぐやること（Phase 1: 3日間）
1. **メインコード統合ナビゲーターの実装**
   - MainAgentの洞察ベクトル生成機能の調査
   - 方向テンプレートベクトルの設計
   - 移動方向抽出アルゴリズムの実装
2. **小規模実験での検証**
   - 5×5迷路での基本動作確認
   - 洞察ベクトルの可視化
3. **既存手法との比較**
   - 5次元コンパクト版（26.8%）との性能比較

### 次にやること（Phase 2: 1週間）
1. **ドーナツ検索の実装**
   - 基本アルゴリズム実装
   - パラメータチューニング
2. **スケーラビリティテスト**
   - 10×10 → 20×20への拡張
3. **ハイブリッド化の検討**

### 将来的にやること
1. **FAISS統合による高速化**
2. **完全なメインコード統合**
3. **論文用の理論的裏付けと図表作成**

## リスクと対策（更新版）

### リスク1: メインコード統合の複雑性
対策：
- 段階的な統合（まずは洞察ベクトル生成のみ）
- フォールバック機構の実装
- 十分なテストカバレッジ

### リスク2: 計算コストの増大
対策：
- キャッシュ機構の活用
- バッチ処理による効率化
- 必要に応じて5次元版へのフォールバック

### リスク3: 方向抽出の精度
対策：
- 複数の方向テンプレート設計を試行
- 機械学習による最適化
- 経験的な重み調整

## 成功基準（更新版）

### Phase 1（メインコード統合）
- **必須**: 基本的な自律移動の実現
- **目標**: 5次元版（26.8%）と同等以上の性能
- **理想**: 理論的に正当化可能な実装

### Phase 2（ドーナツ検索）
- **必須**: カバレッジ40%以上
- **目標**: カバレッジ60%以上、最大訪問50回以下
- **理想**: 全手法を統合した最強版

## タイムライン（更新版）

**Week 1:**
- Day 1-3: メインコード統合実験（Phase 1）
- Day 4-5: 結果分析と改善
- Day 6-7: ドーナツ検索の基本実装開始

**Week 2:**
- ドーナツ検索の最適化
- 3つのアプローチの統合実験

**Week 3:**
- 大規模評価実験
- 論文執筆準備
- 実装の文書化

## まとめ

ユーザーの提案を最優先に、メインコードの洞察ベクトル活用から始めます。これにより：

1. **理論と実装の統合**: geDIG理論の実装がメインコードと融合
2. **既存資産の活用**: MainAgentの高度な機能を再利用
3. **段階的な改善**: 実績ある5次元版からの着実な進化

次は、メインコード統合ナビゲーターの実装から始めましょう！