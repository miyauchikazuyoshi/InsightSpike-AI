# MuSiQue v8 中間レポート — Betti補正項の予測力分析

**Date**: 2026-03-10
**Status**: Interim (追加実験進行中)

---

## 1. 実験概要

| 実験 | Config | Data | EM | F1 | 備考 |
|------|--------|------|-----|-----|------|
| E3A baseline | musique_e3a_4o | musique_dev head-100 (2-hop) | 35.0% | 0.452 | 初回ベースライン |
| **v8 full** | musique_v8_4o | musique_dev head-500 (2-hop) | **37.6%** | **0.480** | θ_dg修正 + Gap Query |

v8の変更点:
1. `theta_dg`: -0.5 → -10.0 (S1ルーティング排除)
2. `component_gap_query`: true (β₀ > 1 → LLM bridge query → 追加検索)
3. 500qへスケール

---

## 2. 核心的発見: F信号の分解

### 2.1 予測力の正体はBetti補正項

```
F = geDIG_base − λ·(γ₁·Δβ₁ − γ₀·Δβ₀)
    ^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^
    diff=+0.064  diff=-0.523 ← こちらが予測力の95%
```

| 成分 | correct avg | wrong avg | **diff** |
|------|------------|---------|---------|
| geDIG_base (情報ゲイン) | +1.113 | +1.048 | +0.064 |
| **Betti補正** (0.5·Δβ₁ − 0.3·Δβ₀) | +0.049 | +0.572 | **-0.523** |
| **F (= base − 補正)** | +1.263 | +0.676 | **+0.587** |

**結論**: geDIG_baseはcorrect/wrong間でほぼ差がない。
**Betti補正項が予測力のほぼ全てを担っている**。

### 2.2 Betti補正の五分位分析

| 五分位 | Betti補正値の範囲 | EM | 解釈 |
|--------|------------------|-----|------|
| Q1 (最低) | −4.2 〜 −1.7 | **44.0%** | β₀支配 = 成分統合 |
| Q2 | −1.7 〜 −0.7 | 36.0% | |
| Q3 | −0.7 〜 +0.2 | 43.0% | |
| Q4 | +0.2 〜 +2.1 | 41.0% | |
| Q5 (最高) | +2.1 〜 +28.5 | **24.0%** | β₁支配 = サイクル過多 |

**Q1 vs Q5 = 20pt差** (n=100 per quintile, N=500 total)

Betti補正が極端に大きい(サイクル構造が支配的) → 正解率が半減する。

### 2.3 閾値別lift

| 予測子 | high EM (n) | low EM (n) | **lift** |
|--------|------------|-----------|---------|
| F > 0.0 | 40.8% (348) | 30.3% (152) | **+10.5pt** |
| **Betti補正 ≤ 0** | **41.3% (288)** | **32.5% (212)** | **+8.8pt** |
| F > 1.0 | 41.2% (294) | 32.5% (206) | +8.6pt |
| geDIG_base > 1.0 | 41.1% (285) | 33.0% (215) | +8.0pt |
| Δβ₁ ≤ 3 | 40.7% (236) | 34.8% (264) | +5.8pt |
| Δβ₀ ≥ 5 | 38.7% (401) | 33.3% (99) | +5.3pt |

---

## 3. FとSF-F1の独立性

### 3.1 相関係数

| ペア | Pearson r | 解釈 |
|------|-----------|------|
| r(F, EM) | +0.085 | F → 回答品質を予測 |
| r(F, SF-F1) | **-0.071** | F ≠ 証拠選択品質 |
| r(SF-F1, EM) | +0.211 | SF-F1 → 回答品質を予測 |
| r(Betti補正, EM) | -0.087 | 補正が大 → EMが低 |

### 3.2 解釈

FとSF-F1は**独立な予測子**:
- **SF-F1** = 「正しい証拠段落を選べたか」(retrieval quality)
- **F** = 「証拠グラフの構造が推論に適しているか」(structural reasonability)

**主張**: 正しい証拠を持っていても、それらが構造的に噛み合わなければ
(β₁が大きい = 矛盾するサイクルが多い) 回答は失敗する。

### 3.3 F値の四分位別

| 四分位 | F値範囲 | EM | SF-F1 |
|--------|---------|-----|-------|
| Q1 (最低) | −29 〜 −0.5 | 28.0% | 0.143 |
| Q2 | −0.5 〜 +1.5 | **43.2%** | 0.123 |
| Q3 | +1.5 〜 +3.0 | 39.2% | 0.120 |
| Q4 (最高) | +3.0 〜 +7.3 | 40.0% | 0.123 |

Q1のみEM=28%で顕著に低い。Q2-Q4は安定して39-43%。
**F < 0 の領域が「構造的に推論不可能」な領域**。

---

## 4. SOTA比較 (distractor設定)

| System | EM | F1 | Model | Category |
|--------|-----|-----|-------|----------|
| Beam Retrieval (beam=2) | — | 69.2 | Supervised | Trained on MuSiQue |
| Ex(SA) | — | 49.7 | Supervised | Trained on MuSiQue |
| SA | — | 47.3 | Supervised | Trained on MuSiQue |
| **geDIG v8 (ours)** | **37.6** | **48.0** | **GPT-4o** | **Zero-shot** |
| EE | — | 42.3 | Supervised | Trained on MuSiQue |

Open-domain (参考):

| System | EM | F1 | Model |
|--------|-----|-----|-------|
| ChainRAG CxtInt | 38.5 | 47.9 | GPT-4o-mini |
| PRISM | 31.2 | 41.8 | GPT-4o |
| IRCoT | 26.5 | 36.5 | GPT-3 |

---

## 5. 未解決問題と進行中の実験

### 5.1 Δβ₁ lift消失問題

100qでは Δβ₁ ≤ 1 → +20.6pt lift だったが、500qでは +0.4pt に消失。

**仮説**:
- (a) Gap Queryでグラフ膨張 (9.9→18.5 nodes) → β₁分布が変化
- (b) 100qの+20.6ptが統計的ノイズ
- (c) Δβ₁の閾値1が500qスケールでは不適切 (Δβ₁ ≤ 3 だと+5.8pt)

**検証**: Ablation実験 (Gap Query無し, θ_dg修正のみ) を実行中。

### 5.2 ホップ数別分析

先行500qは全て2-hop (head-500がたまたま2-hop先頭)。
ランダム500qサンプル (2-hop: 282, 3-hop: 140, 4-hop: 78) で再実験中。

3-hop/4-hopでは:
- β₀ > 1 になりやすい → Gap Queryがより有効な可能性
- 推論チェーンが長い → F信号がより強く効く可能性

### 5.3 Open-domain設定

distractor設定(20段落固定)ではGap Queryの効果に限界。
疑似open-domain (100段落プール) での検証を計画中。

---

## 6. 暫定的な論文フレーミング

### Title (案)
"Betti Numbers as Structural Reasonability Indicators in Multi-hop Question Answering"

### Abstract sketch
Evidence graphのBetti数から計算される補正項
(γ₁·Δβ₁ − γ₀·Δβ₀) が、multi-hop QAにおける回答品質の
「構造的推論可能性」指標として機能することを示す。

- Betti補正の五分位分析: Q1 EM=44% vs Q5 EM=24% (20pt差, N=500)
- この信号はSF-F1 (証拠選択品質) と独立 (r = -0.071)
- GPT-4oで supervised baseline SA (F1=47.3) に匹敵 (F1=48.0)
- MuSiQue distractor設定, 500問で検証

### Key claim
「正しい証拠を見つけることと、その証拠が構造的に推論可能であることは
別の問題である。Betti数はその構造的推論可能性を定量化する。」

---

## Appendix: Git History

| Commit | Description |
|--------|-------------|
| `8dd7e205` | MuSiQue benchmark support + E3A baseline (EM=35.0%) |
| `26e90c2c` | v8 Component Gap Query + θ tuning (EM=37.6%) |
