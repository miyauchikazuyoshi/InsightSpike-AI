---
status: active
category: memory
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# 既存reorganizeコマンドからの学び

## 発見されたコード（2024年5月頃）

```python
@app.command()
def reorganize(iterations: int = 5):
    """インプットなしでメモリの再編成とグラフ最適化を実行"""
```

このコマンドは以下の最適化を繰り返し実行：

### 1. 重要ノードの特定と強化
```python
# グラフから重要なノードを特定
important_nodes = [i for i in range(min(10, len(mem.episodes)))]
# 重要なノードに対応するエピソードの重要度を上げる
for node_id in important_nodes:
    mem.update_c([node_id], reward=0.2)  # 自己報酬
```

### 2. 類似エピソードのマージ
```python
# 類似度の高いエピソードをマージ検討
if len(mem.episodes) > 5:
    mem.merge([0, 1])
```

### 3. 低価値エピソードの刈り取り
```python
# C値が低い/非アクティブなエピソードをprune
mem.prune(c_thresh=0.1, inactive_n=10)
```

### 4. 高C値エピソードの分裂
```python
# C値が高すぎるノードをsplit（例: c>0.95）
for idx, ep in enumerate(mem.episodes):
    if ep.c > 0.95:
        mem.split(idx)
```

### 5. geDIG評価によるフィードバック
```python
# GED/IGの変化を計算
ged_change = delta_ged(g, new_g)
ig_change = delta_ig(old_vecs, vecs)
print(f"  ΔGED: {ged_change:.5f}, ΔIG: {ig_change:.5f}")
```

## 現在の設計への応用

### 1. 自動最適化サイクル

```python
class MemoryOptimizer:
    def __init__(self, memory: Memory, graph_builder: GraphBuilder):
        self.memory = memory
        self.graph_builder = graph_builder
        self.optimization_history = []
        
    def auto_optimize(self, target_ged: float = -0.5):
        """geDIG基準での自動最適化"""
        current_graph = self.graph_builder.get_current_graph()
        
        for iteration in range(5):
            # 1. 重要ノードを中心性で特定
            important_nodes = self.identify_hub_nodes(current_graph)
            
            # 2. 類似エピソードを検出してマージ候補を特定
            merge_candidates = self.find_merge_candidates()
            
            # 3. 低価値エピソードを特定
            prune_candidates = self.find_prune_candidates()
            
            # 4. 最適化アクションを実行
            actions = self.plan_optimization_actions(
                important_nodes, 
                merge_candidates, 
                prune_candidates
            )
            
            # 5. アクションを実行して新グラフを構築
            new_graph = self.execute_actions(actions)
            
            # 6. geDIG評価
            delta_ged = self.calculate_delta_ged(current_graph, new_graph)
            delta_ig = self.calculate_delta_ig(current_graph, new_graph)
            
            # 7. 目標達成したら終了
            if delta_ged <= target_ged:
                logger.info(f"Target GED achieved: {delta_ged}")
                break
                
            current_graph = new_graph
```

### 2. 核保持型マージ

```python
def intelligent_merge(self, episode_indices: List[int]):
    """核を保持しながらエピソードをマージ"""
    episodes = [self.memory.episodes[i] for i in episode_indices]
    
    # 共通構造を抽出
    common_structure = self.extract_common_structure(episodes)
    
    # 各エピソードの核を保持
    cores = [self.extract_core(ep) for ep in episodes]
    
    # 差分表現で新エピソードを作成
    merged_episode = DifferentialEpisode(
        pattern=common_structure,
        cores=cores,
        c_value=np.mean([ep.c_value for ep in episodes])
    )
    
    # 元エピソードを削除せず、アクセス重みを下げる
    for idx in episode_indices:
        self.memory.episodes[idx].access_weight *= 0.1
        
    return merged_episode
```

### 3. 適応的しきい値

```python
def adaptive_thresholds(self, current_stats: Dict):
    """メモリ使用状況に基づいてしきい値を調整"""
    memory_usage = current_stats['episode_count'] / self.max_episodes
    
    if memory_usage > 0.8:
        # メモリ逼迫時は積極的に最適化
        return {
            'merge_threshold': 0.85,  # より緩い基準でマージ
            'prune_threshold': 0.2,   # より積極的に刈り取り
            'split_threshold': 0.98   # 分裂は抑制
        }
    else:
        # 余裕がある時は品質重視
        return {
            'merge_threshold': 0.95,
            'prune_threshold': 0.1,
            'split_threshold': 0.95
        }
```

### 4. バックグラウンド最適化

```python
def background_optimization_task(self):
    """アイドル時にバックグラウンドで最適化"""
    while True:
        if self.is_idle():
            # 小規模な最適化を実行
            self.mini_reorganize()
            
        # 定期的にフルreorganize
        if self.cycles_since_last_reorg > 1000:
            self.full_reorganize()
            self.cycles_since_last_reorg = 0
            
        time.sleep(60)  # 1分待機
```

## 実装提案

1. **reorganizeコマンドの復活**
   - 現在のアーキテクチャに合わせて再実装
   - ScalableGraphManagerと統合

2. **自動トリガー**
   - メモリ使用率が閾値を超えたら自動実行
   - スパイク検出時に部分的reorganize

3. **評価メトリクス**
   - ΔGED: 構造の簡潔性
   - ΔIG: 情報の保持
   - 圧縮率: メモリ効率

4. **ユーザー制御**
   ```bash
   insightspike reorganize --iterations 10 --aggressive
   insightspike reorganize --preserve-recent --threshold 0.8
   ```

## 結論

既存のreorganizeコマンドは、geDIGベースの最適化の優れた実装例。これを現在のアーキテクチャに適応させることで、メモリ爆発を防ぎながら知識を保持できる。