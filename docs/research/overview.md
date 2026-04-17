# geDIG — 1 ページ概要

**geDIG** = **g**raph **e**dit **D**istance and **I**nformation **G**ain  
構造情報を確率に押し込めないスカラー F による動的制御フレームワーク。
式: `F = ΔEPC - λ·ΔIG`（cost vs gain の二項収支、canonical reading）。

**最終更新**: 2026-04-17

---

## 一言で

**geDIG は、構造コストと構造利得を確率分布に押し込めずスカラー量として直接扱う工学フレームワークである。**

既存研究（FEP, VAE, GNN, Information Bottleneck, KL ダイバージェンス等）は、
構造情報を確率表現に**写像することで処理**するが、本研究は**構造量を構造量のまま**取り扱う。

これにより、既存手法が検出できない構造的事象（閃き、位相転移、冗長性）を
**工学的に検出・制御可能**にする。

---

## 核心視覚化: Figure 1 — KL の盲点

同じ編集コスト `EPC = 1` でも、位相 `Δβ₁` は **+1 / 0 / −1** に分岐する:

| Case | 操作 | EPC | **Δβ₁** | ΔH | 判定 |
|---|---|---|---|---|---|
| **A** | 三角形完成（閉路成立） | 1 | **+1** | +0.4 | **Aha! 洞察** |
| **B** | 棒の延長（閉路なし） | 1 | **0** | +0.3 | 力仕事 |
| **C** | 四角形崩壊（閉路消滅） | 1 | **−1** | −0.2 | 構造崩壊 |

**KL ダイバージェンスは ΔH しか測れないため、Case A（洞察）と Case B（力仕事）を区別できない**。
geDIG は `Δβ₁` によってこの区別を可能にする。

📊 視覚化: [thinking/matchstick_figure_v2.html](thinking/matchstick_figure_v2.html)

---

## 既存手法との対比

| 手法 | 測る量 | Case A/B 区別 | Case C 検出 | 計算量 |
|---|---|---|---|---|
| KL ダイバージェンス | 確率分布差 | ✗ | △ | O(N) |
| VAE / ELBO | KL + 再構成誤差 | ✗ | ✗ | O(N) |
| FEP / Active Inference | 予測誤差 | ✗ | △ | O(N·d) |
| GNN message passing | 期待値集約 | ✗ | ✗ | O(E·d) |
| Information Bottleneck | 相互情報量 | ✗ | ✗ | O(N²) |
| TDA (persistent homology) | 位相のみ | △ | ✓ | O(N³) |
| **geDIG (スカラー直接)** | **EPC + ΔH + Δβ₁** | **✓** | **✓** | **O(V+E)** |

---

## 三本柱

### 1. 三項の独立性 — 計量 / 測度 / 位相

`EPC`（計量、組合せ論）・`ΔH`（測度、情報理論）・`Δβ₁`（位相、代数的位相幾何学）は
**現代数学の3つの基本空間概念**に対応する原子量。

単一スカラーへの合成:
```
F = ΔEPC − λ · (ΔH + γ · Δβ₁)
```

### 2. AG/DG 二段ゲート

スカラー F による構造制御の最小機構:
- **AG (Attention Gate)**: 曖昧性・新規性の検知 → 探索開始
- **DG (Decision Gate)**: 有効な構造の確認 → 統合確定

これは閃きの認知プロセス（「あれ？」→「なるほど、繋がった」）の工学的実装。

### 3. Wake-Sleep-Wake による動的構造育成

- **Wake**: AG/DG 発火による経験蓄積
- **Sleep**: グラフ再配線・剪定・圧縮（F 最小化）
- **Wake'**: 抽出された構造での効率的推論

これは Complementary Learning Systems 論（McClelland et al. 1995）の工学的実装。
**演繹的 NN** はこの戦略の帰結として導出される。

---

## 実証状況

### ✅ ポジティブな結果

- **迷路 PoC**: 15×15 → 51×51 で**スケール不変性確認**、15×15 で 98% 成功率
- **OCR** ([vector-based-cnn-ocr](https://github.com/miyauchikazuyoshi/vector-based-cnn-ocr)):
  - **18K params で 73.53% 精度**（標準 CNN の数百万 params 相当）
  - **DG は AG の 5 倍の情報効率**（理論予測と整合）
- **RAG (HotpotQA Lite Suite)**: 予備的な改善を観測

### ⚠️ 修正中の結果

- **Transformer 層別解析**: `delta_r2_struct` が全モデルで負値
  - 原因候補: structural probe 依存の SP 指標が高次元で不安定
  - 修正方針: **β₁ 直接指標への切替**（β₁ の次元フリー性を活用）
  - 詳細: [Part 4 §7](gedig_transformer_architecture.md)

---

## 留保（誠実さの明示）

「構造 ≡ 確率」の数学的厳密化は**本研究の範囲外**。
情報幾何学・Kolmogorov 複雑性・MaxEnt 等の専門家に委ねる open problem。

本研究の貢献は、**スカラー F による構造制御の工学的実証**に徹する。

---

## 読み進め方

| 読者タイプ | 推奨ルート |
|---|---|
| 全体像を掴みたい | [INDEX.md](INDEX.md) → [Part 1 §1-2](gedig_core_theory_unified.md) |
| 査読者・批評者 | [Part 1 §9 棄却可能性](gedig_core_theory_unified.md) → [core_theory §9.6 棄却条件表](gedig_core_theory_unified.md) |
| 実装者 | [Part 1 §7](gedig_core_theory_unified.md) → [Part 5 応用・実装](INDEX.md) |
| 認知科学・神経科学 | [Part 2 認知アーキテクチャ](gedig_cognitive_architecture.md) |
| Transformer/LLM 研究者 | [Part 4 Transformer 統合](gedig_transformer_architecture.md) → [insight_beta1_dimension_free](thinking/insight_beta1_dimension_free.md) |

---

## 研究の 0 軸 — 人間の直観

すべての起点は [gedig_origin_story.md](gedig_origin_story.md) にあります:

> アインシュタインの 1905 年（電磁気学 vs ニュートン力学の矛盾）、
> 湯川秀樹の木目（ありえない 2 領域の位相的接続）を観察して得た直観:
>
> **閃き = 既存の知識（記憶）をトポロジカルに再構成すること**

この直観は人間の観察から生まれたもの。数式・実装・精緻化は、
ここから **AI との対話を通じて**展開されています。

---

## 連絡・協力募集

- **GitHub**: [InsightSpike-AI](https://github.com/miyauchikazuyoshi/InsightSpike-AI)
- **外部プロジェクト (OCR 実証)**: [vector-based-cnn-ocr](https://github.com/miyauchikazuyoshi/vector-based-cnn-ocr)
- **査読者募集**: [call-for-reviewers.md](call-for-reviewers.md)

特に以下の方とのコラボレーションを歓迎:
- **情報幾何学 / MDL / MaxEnt**: 構造 ≡ 確率 の等価性の定式化
- **認知神経科学**: AG/DG と神経調節物質の対応の検証
- **Transformer 研究**: 大規模モデル（Llama 8B+）での検証
- **TDA / persistent homology**: differentiable PH の geDIG への適用

---

**一言で: 構造を確率に押し込めない工学フレームワーク。迷路・OCR で動作実証済み、Transformer は修正中。**
