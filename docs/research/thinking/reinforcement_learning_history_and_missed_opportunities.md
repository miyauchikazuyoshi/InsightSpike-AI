# 強化学習の歴史とgeDIG的アプローチの可能性

## 強化学習の歴史

### 黎明期（1950s-1980s）
```
1954: Minsky's SNARC（最初のニューラルネット強化学習）
1959: Samuel's Checkers（自己対戦で学習）
1970s: 適応制御理論の発展
```

### 理論確立期（1980s-1990s）
```
1989: Watkins - Q-Learning
1992: Tesauro - TD-Gammon（バックギャモン世界レベル）
1994: Rummery & Niranjan - SARSA
1998: Sutton & Barto「強化学習」教科書
```

### 停滞期（2000s-2010）
```
・関数近似の難しさ
・サンプル効率の悪さ
・実世界応用の限界
→ 研究者の多くが他分野へ
```

### 深層強化学習革命（2013-）
```
2013: DQN（Atariゲーム）
2015: A3C（非同期学習）
2016: AlphaGo（囲碁世界チャンピオン撃破）
2017: AlphaZero（汎用ゲームAI）
```

## 迷路探索の歴史

### 古典的アプローチ
```
1960s: A*アルゴリズム（最短経路）
1970s: Dynamic Programming（全状態評価）
1980s: Reinforcement Learning（試行錯誤）
```

### なぜgeDIG的アプローチが生まれなかったか

## 見逃された可能性：特徴点記憶と動的最適化

### あなたの洞察
```
迷路の特徴点のみ記憶 + 記憶の最適化
→ 試行回数の大幅削減
```

### 実は近い研究はあった！

#### 1. Cognitive Maps（1990s）
```
Tolman's Cognitive Map（1948）の計算機実装
・ラットは迷路の「重要な場所」だけ記憶
・でも「どれが重要か」の判断が難しかった
```

#### 2. Hierarchical RL（2000s）
```
Options Framework（Sutton, 1999）
・サブゴールを設定
・でも手動設定が必要
→ 自動的な特徴点発見には至らず
```

#### 3. Memory-based RL（2010s）
```
Neural Turing Machine + RL
Differentiable Neural Computer
・外部メモリを持つ
・でも「何を記憶すべきか」が不明確
```

## なぜgeDIG的発想に至らなかったか

### 1. 状態表現の問題
```
従来：グリッド座標 (x, y)
必要：意味的な埋め込み空間
→ 2010年代まで良い埋め込みがなかった
```

### 2. 記憶の最適化の概念不足
```
従来：全て記憶 or ランダムサンプリング
geDIG：構造的重要性（GED）で選択
→ グラフ的な見方が欠けていた
```

### 3. 探索と記憶の分離
```
従来：探索アルゴリズム ≠ 記憶システム
geDIG：探索しながら記憶を最適化
→ 統合的な視点の欠如
```

## もしgeDIG的アプローチが使われていたら

### 迷路探索の革命
```python
# 従来のQ-Learning
for episode in range(10000):  # 大量の試行
    state = start
    while state != goal:
        action = epsilon_greedy(Q[state])
        next_state, reward = step(action)
        Q[state][action] += alpha * (reward + gamma * max(Q[next_state]) - Q[state][action])

# geDIG的アプローチ
memory = PriorityMemory()  # 特徴点のみ保持
for episode in range(100):  # 少ない試行
    state = start
    while state != goal:
        # 記憶から関連する特徴点を検索
        relevant_points = memory.search_sphere(state, radius=r)
        
        # 構造的に重要な点を発見したら記憶
        if is_structural_keypoint(state, relevant_points):
            memory.add_with_priority(state, calculate_GED(state, relevant_points))
        
        # 記憶を使って効率的に行動選択
        action = plan_with_memory(state, relevant_points, goal)
```

### 期待される効果
1. **試行回数：10000回 → 100回**
2. **記憶量：全状態 → 特徴点のみ**
3. **汎化性能：新しい迷路にも適応**

## 実際に近づいた研究

### 1. World Models（2018）
```
・環境の内部モデルを学習
・夢の中で訓練
→ Wake-Sleepに近い！
```

### 2. Episodic Control（2017）
```
・成功体験を記憶
・類似状況で再利用
→ でも構造的な選択はない
```

### 3. Graph Neural Networks + RL（2020-）
```
・状態をグラフとして表現
・でもgeDIG的な最適化はまだ
```

## なぜ今geDIGが可能か

### 技術的条件の成熟
1. **埋め込み技術**：状態の意味的表現
2. **グラフニューラルネット**：構造の理解
3. **メモリ効率的な計算**：スパースな記憶

### 概念的ブレークスルー
1. **特徴点の自動発見**：GEDによる重要性判定
2. **動的な記憶最適化**：不要な記憶の削除
3. **探索と学習の統合**：Wake-Sleepサイクル

## 歴史的教訓

### 見逃されたアイデア
```
1990s: 「重要な場所だけ覚える」→ 実装困難
2000s: 「階層的に考える」→ 手動設計
2010s: 「経験を記憶する」→ 全部記憶

2024: geDIG「構造的に重要な点を自動選択して動的に最適化」
→ ついに実現可能に！
```

### タイミングの重要性
- 早すぎた：技術基盤（埋め込み、グラフ）が未熟
- 今：ちょうど良い（基盤技術が成熟）

## 結論

強化学習の歴史を見ると、geDIG的なアイデアの断片は存在していた：
- 認知地図（重要な場所の記憶）
- 階層強化学習（構造の活用）
- エピソード記憶（経験の再利用）

しかし、これらを統合し、**自動的に特徴点を発見し、動的に最適化する**という発想は実現されなかった。

geDIGは、70年の強化学習の歴史で見逃されてきた「**構造的に賢い探索**」を実現しようとしている。