# 目的関数とGED定義の整理

## 1. ベースラインの目的関数

### Grid-World環境での外発的報酬（External Reward）
```python
# 実験での設定
goal_reward = 10.0      # ゴール到達
step_penalty = -0.01    # 各ステップ
collision_penalty = -0.1 # 壁/障害物への衝突
timeout_penalty = -1.0   # タイムアウト
```

### 強化学習の目的関数
**ベースライン（内発的動機なし）:**
```
maximize E[Σ γ^t R_external(s_t, a_t)]
```
- R_external: 外発的報酬のみ
- γ = 0.99: 割引率

**内発的動機付きの目的関数:**
```
maximize E[Σ γ^t (R_external(s_t, a_t) + λ × R_intrinsic(s_t, s_{t+1}))]
```
- R_intrinsic = ΔGED × ΔIG
- λ = 0.1: 内発的報酬の重み

## 2. GEDの定義に関する混乱

### 本来のInsightSpike-AIでのGED定義
```
GED(G1, G2) = min Σ(cost(edit_operation))
```
- グラフG1をG2に変換する最小編集コスト
- 編集操作：ノード/エッジの追加・削除

**洞察検出のためのΔGED:**
```
ΔGED = GED(G_after, G_reference) - GED(G_before, G_reference)
```
- **ΔGED < 0**: 構造の単純化（洞察の兆候）
- **ΔGED > 0**: 構造の複雑化

### 実験での簡略化されたGED
```python
# 実際のコード
diversity = np.linalg.norm(state - self.prev_state)
```
- 単純な状態間のユークリッド距離
- 常に非負の値
- グラフ構造を考慮していない

## 3. なぜGEDの定義が違うのか

### 概念的な違い
1. **本来のGED**: 知識構造の「単純化」を測定
   - 洞察 = 複雑な知識が簡潔に再構成される
   - 負の値が重要（構造の単純化）

2. **実験での"diversity"**: 状態空間の「探索」を測定
   - 多様性 = 異なる状態への移動
   - 正の値が探索を促進

### 実験での意図
実験では「ΔGED × ΔIG」を以下のように再解釈：
- **ΔGED → diversity（多様性）**: どれだけ新しい状態を探索したか
- **ΔIG → novelty（新奇性）**: その状態がどれだけ新しいか
- **内発的報酬 = 多様性 × 新奇性**

## 4. 正しいGED実装の例

```python
def calculate_true_ged(self, state1, state2):
    """本来のGEDに近い実装"""
    # 状態をグラフ表現に変換
    g1 = self.state_to_graph(state1)
    g2 = self.state_to_graph(state2)
    
    # 簡易的な編集距離（ノード位置の差）
    pos1 = self.extract_agent_position(state1)
    pos2 = self.extract_agent_position(state2)
    
    # マンハッタン距離を編集コストとして使用
    edit_cost = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    # 参照グラフ（ゴール状態）との距離
    goal_dist1 = abs(pos1[0] - (self.size-1)) + abs(pos1[1] - (self.size-1))
    goal_dist2 = abs(pos2[0] - (self.size-1)) + abs(pos2[1] - (self.size-1))
    
    # ΔGED = 新状態のゴール距離 - 旧状態のゴール距離
    delta_ged = goal_dist2 - goal_dist1
    
    return delta_ged  # 負の値 = ゴールに近づいた
```

## 5. まとめ

### 実験での簡略化の影響
1. **GEDの意味が変化**: 構造単純化 → 状態多様性
2. **符号が逆転**: 負値（良い） → 正値（良い）
3. **脳科学的意味の喪失**: 洞察の瞬間を検出できない

### それでも実験が機能した理由
- 「新奇性 × 多様性」は探索促進に有効
- 内発的動機付けの基本概念は保持
- Grid-World環境では探索が重要なため、簡略版でも効果的

### 今後の改善方向
1. 真のGED実装（ゴールへの距離減少を測定）
2. 洞察検出の実装（ΔGED < -0.5 かつ ΔIG > 0.2）
3. EurkaSpike発生時の特別な報酬システム