# geDIGにおける「行動」の定義：予測 - 理解

**Date**: 2026-02-06
**Status**: Hypothesis / Working Draft
**Origin**: Claude対話セッションでの着想
**Related**:
- [予測の定義（curl検出）](./gedig_prediction_curl.md)
- [geDIG認知論的基盤](./gedig_cognitive_foundation.md)
- [三角対照学習](./gedig_triangular_contrastive_learning.md)

---

## 1. 核心：行動は引き算（仮説）

```
行動 = 予測ベクトル - 理解ベクトル
```

この定式化により、新しいモジュールや原理を導入することなく、
既存のベクトル演算から行動が導出される可能性がある。

行動は「設計するもの」ではなく、予測と理解から「自然に出るもの」
という仮説を検討する。

---

## 2. 定義

### 2.1 各ベクトルの意味

```
予測ベクトル (v_pred)：
  curl検出で「芯はここにあるはず」と推定した方向
  まだ確認していない方向

理解ベクトル (v_understood)：
  DG確定で「今わかっているのはここまで」
  確認済みの構造

差分 (v_action)：
  「まだ埋まっていないギャップ」
  = そのギャップを埋めに行く方向
```

### 2.2 v_understood の定義（課題）

```
v_understood の候補定義：

1. 構造ノード表現
   v_understood = embedding(confirmed_structure)

2. サブグラフ重心
   v_understood = mean(embeddings of confirmed nodes)

3. F最小化の結果
   v_understood = argmin_v F(v)

注意：この定義の明確化は今後の課題。
DGが「何を返すか」を一言で定義する必要がある。
```

### 2.3 図解

```
  予測（curl）──→ ★ ここに芯があるはず
                  ↑
                  │ この差分 = 行動ベクトル
                  │
  理解（DG）──→ ● 今わかっているのはここ


  action = v_pred - v_understood

  行動の方向 = 予測が指す先 - 今の理解
  行動の大きさ = ギャップの大きさ
```

---

## 3. 4段階がベクトル演算になる（提案）

```
予測：  v_pred = curl(attention)         ← ベクトル
認知：  error = |v_pred - v_obs|         ← スカラー（AG信号）
理解：  v_understood = argmin F          ← ベクトル（DG確定）
行動：  v_action = v_pred - v_understood ← ベクトル（差分）
```

この統一的な表現が成立するかは検証が必要。

---

## 4. Active Inferenceとの対応

### 4.1 形式的対応

```
Friston (Active Inference)：
  行動 = 予測誤差を最小化する介入
  → 複雑な変分推論が必要
  → 実装が重い

geDIG（仮説）：
  行動 = v_pred - v_understood
  → 引き算
  → 実装が軽い

両者が同じことを表しているかは、数学的検証が必要。
```

### 4.2 今後の課題

- [ ] Active Inferenceとの数式対応表の作成
- [ ] 変分推論との等価性の検討
- [ ] 計算コストの比較

---

## 5. 自律的探索の動作（仮説）

```
curl → 「あそこに芯がある」（予測）
DG  → 「今ここまでわかった」（理解）
差分 → 「あと何を調べればいい」（行動）
     → 調べる
     → 新しいobservation
     → curlを再計算
     → ループ

差分がゼロになったら → 探索完了
差分がゼロにならない → まだ調べるべきことがある
```

**停止条件が自然に出る**という性質は、この定式化の利点の一つ。

---

## 6. 行動 × 新観測 = 次の予測（仮説）

### 6.1 再帰構造

```
v_action = v_pred - v_understood（今の行動）

行動すると新しい観測が得られる。
その観測空間で外積を取ると：

v_pred_next = v_action × v_obs_new

行動ベクトルと新観測の外積
= 両方に垂直な方向
= 次の芯の予測（仮説）

この再帰構造の妥当性は検証が必要。
```

### 6.2 ループの自己生成

```
予測 → 認知 → 理解 → 行動 → 外積 → 次の予測
  ↑                                    │
  └────────────────────────────────────┘

各ステップ：
  予測 = curl
  認知 = 差の検出（AG）
  理解 = 構造確定（DG）
  行動 = 予測 - 理解
  次の予測 = 行動 × 新観測の外積

同じ演算の繰り返しで、外部制御なしに
サイクルが回る可能性がある。
```

### 6.3 なぜ外積で「次の予測」になるか（仮説）

```
行動ベクトル：「こっちを調べた」
新観測：「こういう結果だった」

外積：
  行動の方向 × 結果の方向
  = 両方に垂直
  = 「調べた方向」でも「結果の方向」でもない
  = 第三の方向
  = まだ見ていない芯の方向（仮説）

この解釈の妥当性は検証が必要。
```

---

## 7. 実装スケッチ

```python
class geDIG_Action:
    def cycle(self, observation):
        """
        geDIGサイクル（概念実証）

        注意：v_understood の定義が曖昧な状態での
        試験的実装。
        """
        # 1. 予測（curl検出）
        v_pred = self.detect_curl(self.attention(observation))

        # 2. 認知（AG）
        error = self.AG(v_pred, observation)

        # 3. 理解（DG）
        # 注意：この部分の実装詳細は要検討
        v_understood = self.DG(observation, error)

        # 4. 行動
        v_action = v_pred - v_understood

        # 停止条件
        if np.linalg.norm(v_action) < epsilon:
            return v_understood  # 探索完了

        # 環境に介入して新観測を得る
        new_observation = self.act(v_action)

        # ループ
        return self.cycle(new_observation)
```

---

## 8. AI推論での「行動」の具体例

```
行動 = 「予測誤差を減らすために環境に介入する」

AI推論での実装案：
  ├── 追加情報を取りに行く（検索、API呼び出し）
  ├── 実験を設計する（仮説検証）
  ├── 質問を生成する（ユーザーへの確認）
  └── 仮説を提示して反応を見る

具体例：
  curl検出 → 「ここに渦がある」（予測）
  AG → 「でもデータが足りない」（認知）
  DG → まだ発火しない（理解未達）
  行動 → 「この情報を検索すれば確認できる」（介入）
       → 検索結果が返ってくる
       → curl再計算 → AG → DG → ...
```

---

## 9. 検証が必要な点

### 9.1 定義の明確化

- [ ] v_understood の具体的定義（構造ノード？サブグラフ重心？）
- [ ] DGが「何を返すか」の一言定義
- [ ] v_pred と v_understood が同じ空間に存在することの保証

### 9.2 数学的検証

- [ ] Active Inferenceとの等価性
- [ ] 行動 × 観測 = 次の予測 の妥当性
- [ ] 停止条件の収束性

### 9.3 実験的検証

- [ ] 実際のタスクでサイクルが収束するか
- [ ] 停止条件が適切に発動するか
- [ ] 従来手法との比較

---

## 10. 結論

```
仮説：
  行動 = 予測 - 理解
       = v_pred - v_understood
       = 引き算

  行動 × 新観測 = 次の予測
               = 外積

3つの演算だけで全サイクルが回る可能性：
  1. 外積（予測）
  2. 引き算（行動）
  3. 最小化（理解）

課題：
├── v_understood の明確な定義
├── Active Inferenceとの数式対応
├── 再帰構造の妥当性検証
└── 実験的検証
```

---

**End of Document**
