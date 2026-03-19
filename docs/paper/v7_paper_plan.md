# geDIG v7 Paper Plan

## v6 → v7: What Changes

### v6 の構成 (現行)
```
1. Introduction: F = ΔEPC - λ(ΔH + γΔSP), AG/DG two-stage gating
2. Mechanism: Event-driven control algorithm
3. Evaluation Protocol
4. Experiment I: Maze PoC (25x25, 40 seeds)
5. Experiment II: Static RAG baseline (HotpotQA v2/v3)
6. Experiment III: Dynamic GRAG × geDIG
7. Experiment IV: Insight Vector Alignment (supplemental)
8. Ablation Analysis
9. FEP-MDL Bridge
10. Related Work, Limitations, Conclusion
```

### v7 の主要アップデート

| 項目 | v6 | v7 |
|------|-----|-----|
| **F の定義** | ΔEPC - λ(ΔH + γ·ΔSP) | ΔEPC - λ(ΔH + γ·**Δβ₁**) |
| **構造項** | SP (shortest path shortening) | **β₁ (Betti number)** — 位相不変量 |
| **Graph Transformer** | なし | **AGHT**: QKV attention on heterogeneous graph |
| **Transformer 実験** | なし (Future Work 扱い) | **Exp4: F-regularization** (negative_better) |
| **統一実装** | 各実験で別実装 | **src/gedig/ 統一コア** (71 tests) |
| **BRIGHT** | なし | **nDCG@10 = 0.439** (reasoning-intensive retrieval) |
| **HotpotQA SF** | なし | **SF F1 = 0.334** (sentence-level, zero-shot) |

---

## v7 論文構成

```
§1  Introduction
    - F = ΔEPC - λ(ΔH + γΔβ₁) — β₁ への一般化
    - 「1つの方程式、3つのドメイン」
    - AG/DG = attention の閾値判定

§2  Unified Gauge Theory
    §2.1 Definition of F (v6 §1.1 を更新、SP → β₁)
    §2.2 AG/DG as Attention Classification (NEW)
         - Q·K 内積 → f = cost - λ·α → AG/DG
         - Transformer attention との構造的同型性
    §2.3 Wake-Sleep-Wake Cycle (NEW)
         - 3ドメイン共通のサイクル構造
    §2.4 FEP-MDL Bridge (v6 §9 から移動)

§3  Implementation: Unified Core
    §3.1 Protocol-based architecture
    §3.2 F-eval implementation (f_eval.py)
    §3.3 Adapters: Maze / RAG / Transformer
    §3.4 Test structure (71 unit + E2E)

§4  Experiment I: Maze PoC (v6 §4 更新)
    §4.1 Wake-Sleep-Wake with β₁
    §4.2 25x25 results (v6 と同等 + β₁ 比較)
    §4.3 Three-layer search architecture

§5  Experiment II: BRIGHT Reasoning-Intensive Retrieval (NEW)
    §5.1 AGHT: Analytical Heterogeneous Graph Transformer
         - Sentence-Token unified graph
         - QKV attention with 10 analytical parameters
    §5.2 BRIGHT biology 50q: nDCG@10 = 0.439
    §5.3 AG/DG split analysis
    §5.4 Spec progression (V → W → X → Z)

§6  Experiment III: HotpotQA Paragraph Selection (NEW)
    §6.1 Pure graph ranking (no BM25)
    §6.2 R@2 = 0.405 (+170% vs Legacy)
    §6.3 SF F1 = 0.334 (zero-shot vs fine-tuned SOTA)
    §6.4 Bridge vs Comparison: AG/DG routing validation
    §6.5 Grid search: 10 parameters, analytical tuning

§7  Experiment IV: Transformer F-Regularization (NEW)
    §7.1 Intervention design: CE + β·F (Positive/Negative)
    §7.2 Results: negative_better (both SP and β₁)
         - Baseline 88.1% → Positive 87.2% → Negative 89.4%
    §7.3 Interpretation: DG preservation as structural regularization
    §7.4 Cross-validation with AGHT findings

§8  Experiment V: HotpotQA Dual-Process (v6 §5-6 更新)
    §8.1 GPT-4o: EM 51.2% @ 3.6x fewer LLM calls
    §8.2 Unified core との統合

§9  Ablation and Analysis
    §9.1 SP vs β₁ comparison across all experiments
    §9.2 Parameter sensitivity (λ, γ, mp_alpha)
    §9.3 Computational cost analysis

§10 Related Work
    §10.1 Graph Attention Networks (GAT, HGT, Graphormer)
    §10.2 Reasoning-intensive retrieval (BRIGHT, IRCoT)
    §10.3 Topological data analysis in NLP
    §10.4 Knowledge graph construction

§11 Limitations and Future Work
    §11.1 Statistical significance (single seed → 3+ seeds)
    §11.2 Scale (DistilBERT → GPT-2/LLaMA)
    §11.3 ARC Prize application
    §11.4 F-regularization at pre-training scale

§12 Conclusion
```

---

## 各実験で達成すべき検証条件

### Experiment I: Maze PoC

| ID | 検証項目 | 現状 | v7 で必要 | 優先度 |
|----|---------|------|----------|--------|
| M1 | 25x25 で β₁ ベース F が SP ベースと同等以上 | 未実施 | **3 seeds で比較** | HIGH |
| M2 | Wake-Sleep-Wake サイクルが β₁ で正常動作 | 未実施 | **ゴール到達率 ≥ 95%** | HIGH |
| M3 | 統一コア (src/gedig/) 経由で既存結果を再現 | adapter あり、E2E 未実施 | **既存 40 seeds 再現** | MED |
| M4 | Three-layer search と β₁ F の相互作用 | 未検証 | **L1 attention と F の相関** | LOW |

### Experiment II: BRIGHT (RAG)

| ID | 検証項目 | 現状 | v7 で必要 | 優先度 |
|----|---------|------|----------|--------|
| R1 | AGHT nDCG@10 ≥ 0.43 (biology 50q) | 0.439 (Spec X) ✅ | **3 seeds で再現** | HIGH |
| R2 | AGHT が legacy graph を上回る (HotpotQA) | R@2 0.405 vs 0.150 ✅ | **100q で統計検定** | HIGH |
| R3 | AG/DG split で comparison > bridge | 0.429 vs 0.256 ✅ | **p値算出** | MED |
| R4 | 他ドメインでの汎化 (earth_science, economics) | 未実施 | **2+ ドメインで検証** | HIGH |
| R5 | SF F1 の改善 (grid search 後) | 0.334 | **可能なら 0.40+** | MED |
| R6 | 統一コア経由で同等結果 | E2E 0.0000 diff ✅ | そのまま掲載 | DONE |

### Experiment III: HotpotQA Paragraph Selection

| ID | 検証項目 | 現状 | v7 で必要 | 優先度 |
|----|---------|------|----------|--------|
| H1 | R@2 ≥ 0.40 (optimized, 100q) | 0.405 ✅ | **3 seeds 平均** | HIGH |
| H2 | SF F1 ≥ 0.30 (sentence-level) | 0.334 ✅ | **3 seeds 平均** | HIGH |
| H3 | Bridge vs Comparison 分析 | 定性的のみ | **統計検定 (paired t-test)** | MED |
| H4 | Grid search の安定性 | 1 run | **top-5 configs の分散** | LOW |

### Experiment IV: Transformer F-Regularization

| ID | 検証項目 | 現状 | v7 で必要 | 優先度 |
|----|---------|------|----------|--------|
| T1 | negative_better (SP) | 確認済み ✅ | **3 seeds で再現** | CRITICAL |
| T2 | negative_better (β₁) | 確認済み ✅ | **3 seeds で再現** | CRITICAL |
| T3 | β sweep (β = 0.01, 0.05, 0.1, 0.5) | 0.1 のみ | **4点で感度分析** | HIGH |
| T4 | 別タスクでの再現 (NLI or QA) | 未実施 | **1+ タスクで再現** | HIGH |
| T5 | 別モデルでの再現 (BERT-base or GPT-2) | DistilBERT のみ | **1+ モデルで再現** | HIGH |
| T6 | Attention パターン可視化 | 未実施 | **Pos/Neg の attention 差分** | MED |
| T7 | 統一コア経由で同結論 | negative_better ✅ | そのまま掲載 | DONE |

### Experiment V: HotpotQA Dual-Process

| ID | 検証項目 | 現状 | v7 で必要 | 優先度 |
|----|---------|------|----------|--------|
| D1 | GPT-4o EM 51.2% 再現 | v6 で報告済み ✅ | そのまま引用 | DONE |
| D2 | 統一コア経由での再実装 | 未実施 | **adapter 経由で再現** | LOW |

---

## 検証の優先順位 (実行順)

### Phase 1: CRITICAL (論文の成否を決める)
```
T1+T2: Exp4 3-seed 再現 (SP + β₁)           → ~6時間
       結論: negative_better が 3/3 で再現するか？
       失敗条件: 2/3 以下なら v7 から Exp4 を外す
```

### Phase 2: HIGH (論文の説得力)
```
T3: β sweep (4点)                             → ~8時間
T4: NLI タスクでの再現                         → ~3時間
T5: BERT-base での再現                         → ~3時間
R1: BRIGHT 3-seed                             → ~3時間
R4: BRIGHT 他ドメイン (2+)                     → ~2時間
H1+H2: HotpotQA 3-seed                       → ~30分
M1+M2: Maze β₁ 3-seed                        → ~1時間
```

### Phase 3: MED (論文の完成度)
```
R3: AG/DG split 統計検定
H3: Bridge vs Comparison 統計検定
T6: Attention 可視化
R5: SF F1 改善
M4: L1 attention × F 相関
```

---

## 論文ターゲット

| ターゲット | 〆切 | 適合度 | 必要な Phase |
|-----------|------|--------|-------------|
| **JSAI 2026** | TBD | 高 (国内、短い) | Phase 1 のみ |
| **EMNLP 2026** | ~June 2026 | 中 (NLP メイン会議) | Phase 1+2 |
| **NeurIPS 2026** | ~May 2026 | 中-高 (理論寄り) | Phase 1+2+3 |
| **ICLR 2027** | ~Oct 2026 | 高 (representation learning) | Full |
| **Workshop** (any) | 随時 | 高 | Phase 1 のみ |

---

## v6 → v7 の差分サマリ

```
v6: 「1つの方程式で maze と RAG を制御できる」
v7: 「1つの方程式で maze, RAG, Transformer を統一し、
     F が attention の構造を測定でき、
     F 最大化が学習を改善することを実験的に示す」

キーフレーズ:
  "Attention topology preservation via geDIG F-maximization"
  "AG = high attention, DG = low attention"
  "One equation, three domains, zero training"
```
