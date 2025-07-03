# 記憶（メモリ）の使用状況分析

## 1. Experience Replay メモリ

### 実装内容
```python
# メモリの初期化
self.memory = deque(maxlen=2000)  # 最大2000エピソード保存

# 経験の保存
def remember(self, state, action, reward, next_state, done):
    intrinsic = self._calculate_intrinsic_reward(next_state)
    total_reward = reward + self.intrinsic_weight * intrinsic
    self.memory.append((state, action, total_reward, next_state, done))

# 経験の再生（学習）
def replay(self, batch_size=32):
    if len(self.memory) > 32:
        batch = random.sample(self.memory, batch_size)
        # バッチ学習を実行
```

**使用状況：**
- ✅ **使用している** - Experience Replayメモリを実装
- 各ステップの経験を保存
- ランダムサンプリングでバッチ学習
- DQNの標準的な実装

## 2. 状態訪問記録メモリ

### 実装内容
```python
# 状態訪問回数の記録
self.state_visits = defaultdict(int)

# IGの計算で使用
state_key = tuple(state)
self.state_visits[state_key] += 1
novelty = 1.0 / (1.0 + self.state_visits[state_key])
```

**使用状況：**
- ✅ **使用している** - 訪問頻度を記録
- 各状態の訪問回数を保持
- Information Gain（新奇性）の計算に使用
- 探索促進に寄与

## 3. 直前状態メモリ

### 実装内容
```python
# 直前の状態を保持
self.prev_state = None

# GED（多様性）の計算で使用
if self.prev_state is not None and self.use_ged:
    diversity = np.linalg.norm(state - self.prev_state)
self.prev_state = state.copy()
```

**使用状況：**
- ✅ **使用している** - 1ステップ前の状態を記憶
- 状態間の差分計算に使用
- GED（実際は多様性）の計算に必要

## 4. メモリの役割と効果

### Experience Replay の効果
1. **サンプル効率の向上**
   - 過去の経験を再利用
   - データ効率的な学習

2. **相関の除去**
   - 連続した経験の相関を断ち切る
   - 安定した学習

3. **内発的報酬の統合**
   - 保存時に内発的報酬を計算・加算
   - 統一された報酬信号で学習

### 状態訪問記録の効果
1. **新奇性の定量化**
   - 未訪問/低頻度の状態を高く評価
   - 探索の促進

2. **IGの簡易実装**
   - 本来のエントロピー計算の代替
   - 計算効率的

### 直前状態記憶の効果
1. **状態変化の測定**
   - 移動距離（多様性）の計算
   - 停滞の回避

## 5. メモリ使用の特徴

### Baselineとの違い
- **Baseline**: Experience Replayのみ（内発的報酬なし）
- **Full/GED/IG**: 全てのメモリを活用

### メモリ効率
```python
# メモリ使用量の概算
experience_replay: 2000 × (state_size + 3) × 4 bytes
state_visits: 最大 (grid_size)² エントリ
prev_state: state_size × 4 bytes

# 6×6グリッドの場合
total ≈ 2000 × 39 × 4 + 36 × 8 + 36 × 4 ≈ 313KB
```

## 6. 結論

実験では**3種類のメモリを適切に使用**しています：

1. **Experience Replay** - 標準的なDQN学習
2. **状態訪問記録** - IG（新奇性）の計算
3. **直前状態** - GED（多様性）の計算

これらのメモリが内発的動機付けの実装を支えており、特に状態訪問記録が探索促進に重要な役割を果たしています。