# 覚醒モード（Wake Mode）でのgeDIG最小化設計

## 概念的背景

### Sleep Mode vs Wake Mode

**Sleep Mode（睡眠モード）:**
- **目的**: 新しい洞察の発見（スパイク検知）
- **戦略**: geDIG最大化 - 構造的新規性×情報利得を追求
- **動作**: 探索的、創造的、発散的思考

**Wake Mode（覚醒モード）:**
- **目的**: 既知の知識を活用した効率的な問題解決
- **戦略**: geDIG最小化 - 最小限の構造変更で目標達成
- **動作**: 収束的、効率的、目的指向

## 迷路探索における実装

### 現状の問題
```python
# 現在のgeDIG計算は常に「新規性」を追求
score = ged_delta * ig_delta  # 大きいほど良い
```

### 覚醒モードでの修正
```python
class WakeModeNavigator:
    def __init__(self, mode='wake'):
        self.mode = mode
        self.known_paths = {}  # 既知の効率的経路
        
    def calculate_action_score(self, state, action):
        if self.mode == 'wake':
            # 覚醒モード：既知の最適経路に従う
            return self.calculate_wake_score(state, action)
        else:
            # 睡眠モード：新しい経路を探索
            return self.calculate_sleep_score(state, action)
    
    def calculate_wake_score(self, state, action):
        """
        覚醒モード：geDIG最小化
        - 既知の効率的経路への近さを評価
        - 構造的変更を最小限に抑える
        """
        # 既知の成功経路との距離
        path_distance = self.distance_to_known_path(state, action)
        
        # 予測可能性（低エントロピー）を重視
        predictability = 1.0 / (1.0 + self.state_entropy(state))
        
        # geDIG最小化スコア（小さいほど良い）
        ged_minimal = 1.0 / (1.0 + self.structural_change(state, action))
        ig_minimal = 1.0 / (1.0 + self.information_novelty(state, action))
        
        return predictability * ged_minimal * ig_minimal / (1.0 + path_distance)
```

## MainAgentへの統合

### 1. モード切り替えロジック
```python
class MainAgent:
    def __init__(self, config, datastore):
        self.mode = 'sleep'  # デフォルトは探索モード
        self.confidence_threshold = 0.8
        self.switch_cooldown = 10  # モード切り替えのクールダウン
        
    def should_switch_mode(self, task_context):
        """タスクの性質に応じてモードを切り替え"""
        if self.mode == 'sleep':
            # 十分な知識が蓄積されたら覚醒モードへ
            if self.has_sufficient_knowledge(task_context):
                return 'wake'
        else:
            # 行き詰まったら睡眠モードへ
            if self.is_stuck() or self.needs_exploration():
                return 'sleep'
        return self.mode
```

### 2. geDIG計算の修正
```python
def calculate_gedig(self, graph, focal_nodes, mode='sleep'):
    """モードに応じたgeDIG計算"""
    
    if mode == 'wake':
        # 覚醒モード：最小化を目指す
        # 既存の接続を維持し、最小限の変更で目標達成
        ged = self.calculate_minimal_ged(graph, focal_nodes)
        ig = self.calculate_convergent_ig(graph, focal_nodes)
        
        # 小さい値が良い（効率性）
        gedig = 1.0 / (1.0 + ged * ig)
        
    else:  # sleep mode
        # 睡眠モード：最大化を目指す（現在の実装）
        ged = self.calculate_normalized_ged(graph, focal_nodes)
        ig = self.calculate_entropy_variance_ig(graph, focal_nodes)
        
        # 大きい値が良い（新規性）
        gedig = ged * ig
        
    return GeDIGResult(
        gedig_value=gedig,
        ged_value=ged,
        ig_value=ig,
        mode=mode
    )
```

### 3. 迷路探索での実証実装
```python
class AdaptiveMazeNavigator:
    """覚醒/睡眠モードを切り替える適応的ナビゲーター"""
    
    def __init__(self):
        self.mode = 'sleep'
        self.known_segments = {}  # 既知の経路セグメント
        self.exploration_history = []
        
    def navigate(self, current_pos, goal_pos):
        # モード判定
        if self.has_path_to_goal(current_pos, goal_pos):
            self.mode = 'wake'
            action = self.follow_known_path(current_pos, goal_pos)
        else:
            self.mode = 'sleep'
            action = self.explore_new_path(current_pos)
            
        return action
    
    def follow_known_path(self, current_pos, goal_pos):
        """覚醒モード：既知の経路を効率的に辿る"""
        path = self.known_segments.get((current_pos, goal_pos))
        if path:
            return path[0]  # 次の一手
        else:
            # 部分的な経路を組み合わせる
            return self.combine_path_segments(current_pos, goal_pos)
    
    def explore_new_path(self, current_pos):
        """睡眠モード：新しい経路を探索"""
        # 現在の実装（geDIG最大化）を使用
        return self.gedig_based_exploration(current_pos)
```

## 期待される効果

1. **効率性の向上**
   - 既知の領域では高速な移動
   - 無駄な探索の削減

2. **学習の活用**
   - 過去の成功経験を再利用
   - 経路の組み合わせによる新しい解法

3. **適応的な振る舞い**
   - 状況に応じた探索と活用のバランス
   - 行き詰まり時の自動的な探索モード切り替え

## 実装優先順位

1. **Phase 1**: 基本的なモード切り替え機構
2. **Phase 2**: 覚醒モードでのgeDIG最小化実装
3. **Phase 3**: 迷路探索での実証実験
4. **Phase 4**: MainAgentへの統合

これにより、単なる探索エージェントから、学習と活用を使い分ける知的エージェントへの進化が期待できます。