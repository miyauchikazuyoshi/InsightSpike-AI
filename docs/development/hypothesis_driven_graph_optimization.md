# Hypothesis-Driven Graph Optimization

## 概要

現在のInsightSpikeのグラフ構築（類似度閾値による即時確定）を、仮説駆動型の最適化アプローチに拡張する提案です。仮結線→評価→採用/棄却のフローにより、より柔軟で創造的なグラフ構築が可能になります。

## 統合アーキテクチャ

```
┌─────────────────────────────────────────────────┐
│          仮説駆動型グラフ構築システム              │
├─────────────────────────────────────────────────┤
│                                                 │
│  1. 仮結線フェーズ                               │
│     ├─ 直接エッジ（閾値を緩和: 0.5）             │
│     ├─ 既存エピソードブリッジ                    │
│     └─ 仮説ブリッジ（新規生成）                  │
│                                                 │
│  2. 評価フェーズ                                │
│     ├─ ΔGED計算（構造の改善度）                 │
│     ├─ ΔIG計算（情報利得）                      │
│     └─ 仮説妥当性検証                           │
│                                                 │
│  3. 最適化フェーズ                              │
│     ├─ スコアベースの選択                        │
│     ├─ 採用/棄却の決定                          │
│     └─ 信頼度の更新                             │
│                                                 │
│  4. 学習フェーズ                                │
│     ├─ 成功パターンの記録                        │
│     ├─ 仮説の知識化                             │
│     └─ 閾値の動的調整                           │
└─────────────────────────────────────────────────┘
```

## 実装の統合

### 1. エッジタイプの拡張

```python
class EdgeType(Enum):
    DIRECT = "direct"                    # 直接的な類似性
    BRIDGE_EXISTING = "bridge_existing"  # 既存エピソード経由
    BRIDGE_HYPOTHESIS = "bridge_hypothesis"  # 仮説による橋渡し

class EdgeStatus(Enum):
    TENTATIVE = "tentative"    # 仮結線
    EVALUATED = "evaluated"    # 評価済み
    ACCEPTED = "accepted"      # 採用
    REJECTED = "rejected"      # 棄却
```

### 2. 仮説管理システム

```python
class HypothesisManager:
    def __init__(self):
        self.hypotheses = {}  # 仮説のプール
        self.validation_queue = []  # 検証待ちキュー
        
    def generate_hypothesis(self, ep1, ep2):
        """仮説生成（既存エピソード検索失敗時）"""
        # LLMによる仮説生成
        hypothesis = {
            "concept": generated_concept,
            "confidence": 0.3,  # 初期信頼度
            "created_at": timestamp,
            "validation_count": 0,
            "success_count": 0
        }
        return hypothesis
    
    def update_confidence(self, hypothesis_id, success):
        """検証結果に基づく信頼度更新"""
        h = self.hypotheses[hypothesis_id]
        h["validation_count"] += 1
        
        if success:
            h["success_count"] += 1
            h["confidence"] = min(1.0, h["confidence"] * 1.2)
        else:
            h["confidence"] = max(0.1, h["confidence"] * 0.8)
            
        # 十分検証されたら知識化
        if h["validation_count"] > 10 and h["confidence"] > 0.8:
            self.promote_to_knowledge(h)
```

### 3. 統合評価システム

```python
def evaluate_graph_with_edge(self, current_graph, candidate_edge):
    """エッジ追加による影響を評価"""
    
    # 仮想的にエッジを追加
    temp_graph = current_graph.copy()
    temp_graph.add_edge(candidate_edge)
    
    # メトリクス計算
    delta_ged = self.calculate_ged_change(current_graph, temp_graph)
    delta_ig = self.calculate_ig_change(current_graph, temp_graph)
    
    # エッジタイプ別の重み付け
    type_weight = {
        EdgeType.DIRECT: 1.0,
        EdgeType.BRIDGE_EXISTING: 0.8,
        EdgeType.BRIDGE_HYPOTHESIS: 0.6
    }
    
    # 総合スコア
    base_score = -delta_ged + delta_ig  # GED減少とIG増加が良い
    edge_score = base_score * type_weight[candidate_edge.type]
    
    # 仮説の場合は追加検証
    if candidate_edge.type == EdgeType.BRIDGE_HYPOTHESIS:
        validity_score = self.validate_hypothesis(candidate_edge)
        edge_score *= validity_score
    
    return {
        "delta_ged": delta_ged,
        "delta_ig": delta_ig,
        "edge_score": edge_score,
        "recommendation": "accept" if edge_score > 0.1 else "reject"
    }
```

## 期待される効果

### 1. より豊かなグラフ構造
- 直接つながらない概念も関連付け可能
- 創造的な洞察の発見
- 知識の隙間を埋める

### 2. 継続的な学習
- 成功/失敗からの学習
- 仮説の信頼度管理
- 検証済み仮説の知識化

### 3. 最適化された構造
- GED/IGによる客観的評価
- ノイズの少ないグラフ
- 意味のある関連性の強調

## 実装ロードマップ

### Phase 1: 基盤整備（2週間）
- [ ] EdgeType, EdgeStatusの実装
- [ ] 仮結線メカニズムの追加
- [ ] 評価履歴の記録システム

### Phase 2: 仮説システム（3週間）
- [ ] HypothesisManagerの実装
- [ ] 既存エピソード検索の強化
- [ ] 仮説生成インターフェース

### Phase 3: 評価システム（3週間）
- [ ] 統合評価関数の実装
- [ ] バッチ評価の最適化
- [ ] 並列処理対応

### Phase 4: 統合とテスト（2週間）
- [ ] 既存システムとの統合
- [ ] パフォーマンステスト
- [ ] A/Bテスト実施

## 技術的考慮事項

### 1. パフォーマンス
- 仮結線により評価対象が増加
- バッチ処理と並列化で対応
- 重要なエッジから優先評価

### 2. 品質管理
- 仮説の暴走を防ぐ制限
- 最大仮説数の設定
- 定期的なガベージコレクション

### 3. 説明可能性
- なぜそのエッジが採用/棄却されたか
- 仮説の根拠の記録
- 評価プロセスの可視化

## まとめ

この統合アプローチにより、InsightSpikeは：
1. **探索的**: 仮結線による柔軟な構造探索
2. **創造的**: 仮説による新しい関連性の発見
3. **学習的**: 成功/失敗からの継続的改善
4. **最適化**: GED/IGによる客観的な構造改善

を実現し、より人間的で創造的な知識グラフ構築が可能になります。

---

*Created: 2024-01-19*
*Authors: Claude & User*
*Status: Design Integration Phase*