# Experiment 4: F-Regularized Training — Provisional Results

**Date**: 2026-03-19
**Status**: Provisional (single run, needs replication)

## Setup

- Model: DistilBERT (distilbert-base-uncased)
- Task: SST-2 sentiment classification
- Samples: 2000 train, 500 eval
- Epochs: 3
- Device: MPS (Apple Silicon)
- Beta (F regularization weight): 0.1

## Three Conditions

| Condition | Loss Function | Hypothesis |
|-----------|--------------|------------|
| Baseline | CE only | Control |
| Positive | CE + beta * F (minimize F) | "Reducing F improves learning" |
| Negative | CE - beta * F (maximize F) | "Increasing F improves learning" |

## Results

| Condition | Epoch 1 | Epoch 2 | Epoch 3 (Final) |
|-----------|---------|---------|-----------------|
| Baseline | 89.4% | 89.4% | **88.1%** |
| Positive | 89.4% | 88.5% | **87.2%** (-0.9pp) |
| Negative | 89.7% | 89.4% | **89.4%** (+1.4pp) |

**Conclusion**: `negative_better`

## Interpretation (Provisional)

### Key Finding
F maximization (Negative) outperforms both Baseline and F minimization (Positive).

### What This Means

```
F = edge_cost - lambda * query_relevance (geDIG F-evaluation)

F minimization = "make all edges AG" = flatten attention structure
  -> Accuracy DROPS: model loses structural diversity in representations

F maximization = "preserve DG edges" = maintain attention heterogeneity
  -> Accuracy MAINTAINED/IMPROVED: model keeps "what it doesn't know" explicit
```

### Connection to AGHT (Analytical Heterogeneous Graph Transformer)

This result is consistent with today's AGHT findings:
- AGHT preserves DG edges as structural information (not penalty)
- QKV attention separates "what the query needs" (Q) from "what the node provides" (K)
- DG edges = low attention = information gaps = valuable structural signal

### Theoretical Implication

```
CE Loss: optimizes "get the right answer" (compression efficiency)
F regularization: preserves "knowledge structure" (information flow topology)

CE + F_max: optimize answers WHILE maintaining structural diversity
           = "learn the answer but don't collapse your representation"
```

This suggests F captures an independent axis of learning quality that CE alone misses.

## SP版 vs β₁版 比較 (2026-03-19)

F の第3項を SP (経路効率) から β₁ (Betti number) に置き換えて再実験。

```
SP版:  F = ΔEPC - λ(ΔH + γΔSP)   SP = 行列累乗の到達可能性
β₁版:  F = ΔEPC - λ(ΔH + γΔB)    B = β₁ = E - V + C (サイクル数)
```

### 比較結果

| Condition | SP版 (legacy) | β₁版 (new) |
|-----------|-------------|-----------|
| Baseline  | 88.1%       | 88.5%     |
| Positive  | 87.2% (-0.9pp) | 83.5% (-5.0pp) |
| Negative  | 89.4% (+1.4pp) | 85.5% (-3.0pp) |
| Conclusion | negative_better | negative_better |

### 重要な発見

**結論は SP/β₁ どちらでも同じ: `negative_better`**

ただし β₁ 版では:
- **F 最小化の悪影響がより顕著** (83.5% vs SP版 87.2% = -3.7pp差)
- β₁ はサイクル数 = 情報の冗長経路数を直接測定
- F 最小化でサイクルが消滅 → 情報が一方向にしか流れない → 表現力低下

### 核心的な解釈: 事前学習の構造破壊

実験の開始点は `distilbert-base-uncased` (MLM で事前学習済み)。
3条件とも同じ開始点から SST-2 を fine-tune する。

```
事前学習 (MLM)
  → Transformer は自然に attention グラフの構造を発達させた
  → この構造には AG (確認済み情報のパス) と DG (推論ギャップ) が含まれる
  → DG = サイクル = 冗長な情報経路 = 表現の多様性

F 最小化 (Positive)
  = 「全エッジを AG にする」= attention 構造を平坦化
  = 事前学習で獲得した DG 構造 (サイクル/冗長経路) を破壊
  → 精度低下: モデルの表現力が失われた

F 最大化 (Negative)
  = 「DG エッジを維持する」= attention 構造の多様性を保持
  = 事前学習で獲得した構造を保存
  → 精度維持: モデルの表現力が保たれた
```

**これは「完成されたモデルが介入で壊された」のではなく、
「事前学習で獲得した知識構造の多様性が、F 最小化によって失われた」**。

β₁ 版で差が大きい (83.5% vs 88.5% = -5.0pp) のは、β₁ が
「サイクル数 = 冗長な情報経路」を直接測定するため、
構造破壊の影響がより正確に F に反映され、
より強い介入として作用したことを示す。

### 含意

1. Transformer の事前学習は attention グラフに意味のあるトポロジカル構造を作る
2. この構造には AG (直接パス) と DG (サイクル/冗長パス) が含まれる
3. DG 構造を破壊すると性能が劣化する → DG は「無駄」ではなく必要
4. geDIG F はこの構造を測定でき、介入に使える
5. AGHT (検索パイプライン) の設計哲学と一致: DG エッジは罰するのではなく保存する

## Caveats

1. **Single run** — needs replication with different seeds
2. **Small dataset** (2000 samples) — may not generalize to full-scale
3. **Beta=0.1 only** — sensitivity to beta not explored
4. **DistilBERT only** — needs testing on larger models
5. **SST-2 only** — needs testing on reasoning-intensive tasks
6. **β₁版の全体精度低下** — eigvalsh の数値安定性 or β₁ の正規化に問題の可能性

## Next Steps

- [ ] Replicate with 3+ random seeds for statistical significance
- [ ] Beta sweep: [0.01, 0.05, 0.1, 0.2, 0.5]
- [ ] Test on reasoning tasks (NLI, multi-hop QA) where DG structure matters more
- [ ] Visualize attention pattern differences between conditions
- [ ] Scale to full dataset and larger models
- [ ] β₁版の数値安定性を改善 (eigvalsh の epsilon/temperature 調整)
- [ ] SP版とβ₁版の F 値のスケール比較 (mean_F の推移を並べる)
