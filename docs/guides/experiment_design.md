# InsightSpike-AI実験設計書

## 実験1: パラドックス解決タスク (Paradox Resolution)

### 目的
認知的な「aha!」モーメントが発生する問題設定で、InsightSpike-AIの洞察検出能力を評価

### データセット設計
```
実験データ: パラドックスとその解決説明のペア
例:
1. バンディットパラドックス → 測度論的解説
2. ツェノンのパラドックス → 無限級数の収束
3. モンティ・ホール問題 → 条件付き確率
4. 船頭多くして船山に上る → ゲーム理論
```

### 仮説
- ΔGED: パラドックス説明中に構造変化が起きてスパイク発生
- ΔIG: 認知的な情報統合により情報ゲインが急激に増加

### 評価方法
- **人手評価**: 人間が「納得した瞬間」をタイムスタンプで記録
- **InsightSpike検出**: 同じタイミングでΔGED/ΔIGスパイクが発生するか
- **一致率測定**: 人間の「aha!」とシステムのスパイクの相関係数

---

## 実験2: 段階的概念理解タスク (Scaffolded Learning)

### 目的
段階的に複雑さが増す学習タスクで、概念の階層的理解をモデル化

### データセット設計
```
数学概念の段階的説明:
Level 1: 基本算術 (1+1=2)
Level 2: 代数方程式 (x+1=2)
Level 3: 微分方程式 (dx/dt = -x)
Level 4: 偏微分方程式 (∂u/∂t = ∇²u)

物理概念:
Level 1: ニュートン力学 (F=ma)
Level 2: 相対性理論 (E=mc²)
Level 3: 量子力学 (HΨ=EΨ)
Level 4: 場の量子論 (Lagrangian formalism)
```

### 仮説
- **概念ジャンプ**: レベル間遷移でΔGEDが負になる（構造簡化）
- **抽象化ゲイン**: ΔIGが大きくなる（高次概念の獲得）

---

## 実験3: 創発的問題解決タスク (Emergent Problem-Solving)

### 目的
複数の知識領域を統合して新しい解決策を発見するタスク

### データセット設計
```
学際的問題:
- 生物学 + 工学 → バイオミメティクス
- 心理学 + AI → 認知アーキテクチャ
- 物理学 + 経済学 → エコノフィジクス
- 数学 + 芸術 → フラクタルアート
```

### 評価軸
1. **創発度**: 既存知識の線形結合を超えた新規性
2. **関連性**: 異なる領域間の概念的距離
3. **有用性**: 実際の問題解決への貢献度

---

## 実験4: ベースライン比較実験

### 比較対象
1. **Standard RAG**: FAISS検索 + GPT生成
2. **Multi-hop RAG**: 複数回検索を行うRAG
3. **Graph RAG**: グラフベースのRAG
4. **InsightSpike-AI**: 提案手法

### 評価指標
- **回答品質**: BLEU, BERTScore, 人手評価
- **洞察発見率**: 新規パターン発見の頻度
- **効率性**: 同じ品質に到達するまでの計算コスト
- **説明可能性**: 答えに至るプロセスの追跡可能性

---

## 実験5: リアルタイム洞察検出

### 目的
実際の人間の思考プロセスとリアルタイムで比較

### 手法
1. **Think-aloud Protocol**: 被験者に思考を音声化してもらう
2. **Eye-tracking**: 視線移動パターンから認知負荷を測定
3. **Concurrent InsightSpike**: 同時にシステムでもタスクを実行
4. **Correlation Analysis**: 人間の認知状態とシステムの内部状態の相関

---

## データ生成スクリプト案

### 高品質な洞察誘発データの生成

```python
# experiments/data_generators/paradox_generator.py
def generate_paradox_dataset():
    """認知的パラドックスデータセットを生成"""
    paradoxes = [
        {
            "name": "Banach-Tarski Paradox",
            "setup": "A ball can be decomposed and reassembled into two balls of the same size",
            "resolution": "Uses axiom of choice and non-measurable sets in measure theory",
            "cognitive_shift": "discrete_to_continuous",
            "expected_spike_timing": [0.3, 0.7]  # 説明の30%と70%地点
        },
        # ... more paradoxes
    ]
    return paradoxes

# experiments/data_generators/concept_hierarchy.py  
def generate_concept_ladder():
    """概念の階層的学習データを生成"""
    math_ladder = [
        {"level": 1, "concept": "arithmetic", "prerequisite": None},
        {"level": 2, "concept": "algebra", "prerequisite": "arithmetic"},
        {"level": 3, "concept": "calculus", "prerequisite": "algebra"},
        {"level": 4, "concept": "analysis", "prerequisite": "calculus"},
    ]
    return math_ladder
```

---

## 期待される成果

### 1. **洞察の定量化**
- ΔGED/ΔIGによる「aha!」モーメントの客観的測定
- 人間の認知過程との相関関係の証明

### 2. **創発的知識発見**
- 既存RAGを超えた新規知識の発見能力
- 異分野統合による創造的解決策の生成

### 3. **説明可能AI**
- 洞察に至るプロセスの可視化
- エピソード記憶のC値変化による学習過程の追跡

### 4. **実用性の証明**
- 研究・教育・創薬等の実問題での有効性
- 計算効率とスケーラビリティの実証

---

## 次のステップ

1. **実験環境構築**: データ生成とベースライン実装
2. **評価フレームワーク**: 定量・定性評価の自動化
3. **被験者実験**: 認知科学的検証の実施
4. **論文執筆**: 査読付き国際会議への投稿

この設計により、InsightSpike-AIの科学的妥当性と実用性を包括的に証明できると考えられます。
