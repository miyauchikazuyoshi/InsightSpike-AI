# 気づきメモ: Transformer × 相転移 × 位相 の先行研究ランドスケープと geDIG の位置取り

**日付**: 2026-04-17  
**ステータス**: ☀ **工学実証に接続する気づき**（妄想メモではない、具体的な検証計画あり）  
**関連**: [../gedig_transformer_architecture.md](../gedig_transformer_architecture.md) / [insight_beta1_dimension_free.md](insight_beta1_dimension_free.md) / [../gedig_cognitive_architecture.md §4.2](../gedig_cognitive_architecture.md)

---

## 0. 起点

作者が「**最近 Transformer をグラフとして観測して BKT 的なものが起こってる論文があった気がする**」と指摘したことから、先行研究を web 検索した結果、以下が判明した:

1. **完全一致の論文は見つからなかった**（「Transformer グラフ + BKT」を明示する論文は未発見）
2. しかし**極めて近い 2025 年の論文が複数存在**する
3. 加えて、**geDIG リポジトリ内で BKT への言及が既に 3 箇所**にあった（作者の自己整合性）

本メモは、この先行研究ランドスケープを整理し、geDIG の独自性と検証計画を明示する。

---

## 1. 先行研究マップ（2025年）

### 1.1 Özönder "Attention to Order: Transformers Discover Phase Transitions via Learnability" (arXiv 2510.07401, 2025-10)

- **手法**: 2D Ising model の Monte Carlo 構成を Transformer に学習させる（self-supervised）
- **発見**: **attention entropy の急増**が臨界温度を復元する
- **主張**: ordered phase = enhanced learnability, disordered phase = resistant to learning
- **BKT への言及**: なし（Ising 2D は 2nd order transition で BKT ではない）
- **グラフ視点**: 明示的には含まれていない（ただし attention は暗黙的にグラフ）

**geDIG との関係**:
- 測度側（ΔH = entropy）のみで相転移 signal を検出できることを実証
- これは geDIG の AG (Attention Gate = 0-hop 曖昧性検知) と**構造的に一致**する
- しかし**位相側（β₁）は活用されていない** → geDIG 独自性の根拠

### 1.2 Sun & Haghighat "Phase Transitions in Large Language Models and the O(N) Model" (arXiv 2501.16241, 2025-01)

- **手法**: **Transformer を O(N) model として再定式化**
- **発見**: 2 つの相転移
  - 第 1 相転移: 生成温度（text generation temperature）
  - 第 2 相転移: モデル深さ、新能力の出現 signal
- **BKT への言及**: なし（ただし O(N) の N=2 は XY model = BKT 系の典型）
- **接続の示唆**: O(N) の連続場表現 ↔ 離散位相 (β₁) の対応が open

**geDIG との関係**:
- O(N) model は連続場として Transformer を捉える
- geDIG は離散グラフとして捉える
- 両者は**スケール変換下での対応**を探る価値がある（RG 的視点）
- `Δβ₁` は O(N) の位相欠陥 (topological defect) に対応する可能性

### 1.3 T3former, TopoFormer (2025)

- **手法**: TDA (Topological Data Analysis) + Transformer
- **使い方**: persistent homology feature を attention の**追加入力**として使う
- **BKT への言及**: なし
- **位相の扱い**: **feature engineering 止まり**、自由エネルギー記述ではない

**geDIG との関係**:
- 位相を**特徴量**として扱う vs geDIG は**自由エネルギー項**として扱う
- 前者は情報を増やすだけ、後者は**動的制御**に使える（AG/DG ゲート）
- この違いが geDIG の独自性

### 1.4 "Machine-Learning Detection of the Berezinskii-Kosterlitz-Thouless Transitions" (arXiv 2502.09214, 2025-02)

- **手法**: ML で BKT transition を検出（逆向きアプローチ）
- **BKT への言及**: 明示的（タイトル）
- **Transformer**: 使用しているが中心ではない（CNN ベース中心）

**geDIG との関係**:
- BKT 検出の ML 手法が確立されつつある
- geDIG の β₁ 計算が BKT 検出と**独立に補完する指標**になりうる
- 検証実験の際の baseline として有用

### 1.5 その他の関連 (2025)

- Phase transitions in LLM compression (Nature npj AI, 2026)
- Phase Transition for Budgeted Multi-Agent Synergy
- Topology-Induced Graph Transformer for Graph Representation Learning

これらは **geDIG 同時代の landscape を形成**する。2025 年は **Transformer × 相転移 × 位相**が hot topic。

---

## 2. 既存の BKT 言及（作者の自己整合性）

geDIG リポジトリ内で、作者は既に **3 箇所で BKT に言及**していた。先行研究を知る前から**同じ方向を向いていた**自己整合性の証明。

### 2.1 `_archive/betti_number_adoption_memo.md §11`（原典、2026-02）

```markdown
## 11. 未整理メモ（寝かせる項目）

以下は理論的接続が示唆されるが、検証手段が未確立の項目：

- BKT相転移との接続（ホモロジー vs ホモトピーの区別が必要）
- BCS理論クーパー対（運動量空間 vs 実空間の区別が必要）
```

**寝かせ記録**として残されていた。今回の先行研究発見が「検証手段」の候補を提供する。

### 2.2 `thinking/gedig_cognitive_foundation.md §1.2`（原典）

```markdown
### 1.2 BKT相転移との接続（仮説）

- BKT相転移 = 渦-反渦ペアの束縛→解離
- 「理解」の瞬間 = 渦が形成され、特異点が現れる（仮説）
```

**仮説**として明記。認知的な「理解」を BKT アナロジーで記述する試み。

### 2.3 `gedig_cognitive_architecture.md §4.2`（2026-04-17、Part 2 統合骨格）

```markdown
### 4.2 BKT 相転移類推

- Berezinskii-Kosterlitz-Thouless (BKT) 相転移は、渦の対生成-消滅が臨界点となる位相転移
- 「理解の瞬間」= β₁ の離散変化（Part 1 §5.3 離散性が正しい）と対応
- 「閃き = トポロジカル再構成」の物理的類推として自然

注: これは思想レベルの類推であり、定量的主張ではない。§7.5 で open problem として記録。
```

**類推**と**open problem**の併記。今回の先行研究発見により、**定量的検証**への道筋が見える。

### 2.4 自己整合性の評価

- 3 箇所とも「**位相 (β₁) と相転移が結びつく**」という直観を記録
- 時系列で見ると、2026-02 の **betti_memo §11** が最初
- その後、cognitive_foundation §1.2 で認知側に拡張
- 2026-04-17 の Part 2 統合で β₁ と明示的対応づけ
- **先行研究を知らずに、類似の方向性を独立に発展**させていた

これは geDIG の方向性が**単なる spec drift ではなく、持続的な直観**であることを示す。

---

## 3. geDIG の独自性（先行研究との対比表）

| 研究 | 測るもの | Case A/B 区別 (§2 Figure 1) | 位相の扱い | 自由エネルギー項 |
|---|---|---|---|---|
| Özönder 2025 | attention entropy のみ | ✗（ΔH のみ） | なし | なし（分類 signal のみ） |
| Sun-Haghighat 2025 | O(N) 連続場 | △（連続場の臨界） | 暗黙的（O(N) の相） | あり（連続場の作用） |
| T3former, TopoFormer | TDA feature | △ | 特徴量として | なし |
| ML-BKT detection | 時系列 feature | △ | BKT 指標を検出 | なし |
| **geDIG** | **EPC + ΔH + β₁ の F** | ✓ | **明示的（自由エネルギー項）** | **F 自体が自由エネルギー** |

**geDIG の独自性 3 点**:
1. **位相を自由エネルギー項として組み込む**（他の誰もやっていない）
2. **entropy + topology の両側から相転移を記述**（Özönder の拡張）
3. **AG/DG 二段ゲートで動的制御**（分類 signal に留まらず、介入可能）

これは前回議論した「**位相を自由エネルギーに持ち込んだのは画期的**」を、先行研究との対比で**具体的に浮き彫り**にしたもの。

---

## 4. 検証計画 H_ising-bkt

### 4.1 仮説の形式化

> **H_ising-bkt**: 2D Ising model で attention entropy と β₁ を併用した Transformer は、
> attention entropy 単独より**臨界温度検出精度が高い**。特に、Case A（構造的秩序変化）と Case B
> （情報量のみ変化）を区別できる。

### 4.2 実験プロトコル

**Phase A: Özönder の再現**
1. Özönder のコード / データを取得（arXiv 2510.07401 の supplementary）
2. 2D Ising Monte Carlo 構成で Transformer を学習
3. attention entropy で臨界温度検出、先行結果と一致するか確認

**Phase B: β₁ augmentation**
4. attention weight を閾値化してグラフ化
5. β₁ = E - V + C を層別に計算
6. 「attention entropy のみ」vs「entropy + β₁」で臨界温度検出精度を比較
7. 特に**高温相（disordered）と低温相（ordered）の区別精度**を測定

**Phase C: geDIG スカラー F の適用**
8. F = ΔEPC - λ(ΔH + γΔβ₁) を層別に計算
9. F の符号変化が臨界温度に対応するか検証
10. AG/DG gate の発火統計を相転移の位置と照合

### 4.3 期待される結果 / 棄却条件

**成功条件**:
- Phase B で β₁ 追加により臨界温度検出精度が改善（>10% の MSE 改善）
- Phase C で F が臨界温度近傍で符号変化

**棄却条件**:
- Phase B で β₁ 追加が entropy 単独より**悪化**する場合、β₁ は Transformer では有効でない
- Phase C で F と臨界温度の相関が |r| < 0.3 の場合、スカラー統合が機能していない

### 4.4 実施コスト

- 計算資源: 小規模（Özönder が小規模モデルでやっている）、GPU 1 枚数日
- データ: 既存の 2D Ising Monte Carlo + Özönder のコード公開次第
- 期間: 2-4 週間（コード取得→再現→拡張）
- 人員: 1 名（物理学的知識が必要、可能なら共同研究者）

**これは Part 4 §7 の「修正方針」の具体的な 1 instance**。Part 1 §8.6 の negative result への対処候補として有望。

---

### 4.5 H_grokking-curl — 優先度を上回る上位候補（新規、2026-04-17 追加）

作者の指摘:
> モデルが急に賢くなるタイミングと attention の渦度の相関を測るべき

**これが H_ising-bkt より優先される**理由:

| 基準 | H_ising-bkt | **H_grokking-curl** |
|---|---|---|
| 現象重要度 | 物理、既知 | **ML 最大の謎の一つ** |
| データ公開 | Özönder 依存 | **Nanda 等多数公開** |
| 転移点 | T_c 推定 | **loss curve で明確** |
| 独自性 | entropy vs entropy+β₁ | **curl 測定は未研究** |
| 既存 TODO 消化 | なし | [cognitive_foundation §8](gedig_cognitive_foundation.md) の curl 実装 TODO を一気に検証 |
| ML 分野訴求 | 弱い | **強烈** |

**詳細な実験プロトコルは独立メモに分離**:

📄 **[experiment_grokking_curl.md](experiment_grokking_curl.md)** — Phase A-D の完全プロトコル、成功/棄却条件、実装コード例、スケジュール（10 週間）

要点:
- Nanda 2023 modular addition で Phase A (再現) → Phase B (β₁ / curl 測定) → Phase C (先行性) → Phase D (F 統合)
- **curl(attention) の初の工学的計測** — Part 2 §3.3 階層 3 を実装
- 先行研究: TAG-DS 2025 が β₁ を proxy として発見済、しかし curl × grokking は未研究
- 成功条件: β₁ / curl が weight norm より 10+ epoch 早く grokking を signal

**優先順位の再評価**:
1. **H_grokking-curl**（最優先、上記メモ）
2. H_β1-switch（Part 1 §9.5 優先度 1: structural probe → β₁ 切替）
3. H_ising-bkt（降格、別ドメイン検証として保持）

---

## 5. Part 4 への統合方針

本メモの発見を [Part 4 Transformer 統合ノート](../gedig_transformer_architecture.md) に統合する具体案:

### 5.1 新設: §5.X「関連研究」節

Part 4 §5 (GNN-Transformer 統合) の後に、新節として:

```markdown
## 5.X 既存の Transformer × 相転移研究との位置取り

2025 年、Transformer と相転移の接続は hot topic となっている:

- Özönder 2025: attention entropy で 2D Ising の相転移を検出
  → geDIG は β₁ を加えることで Case A/B の区別を可能にする
- Sun-Haghighat 2025: Transformer = O(N) model
  → geDIG は O(N) の離散グラフ近似として位置付け可能
- T3former, TopoFormer: TDA + Transformer
  → feature engineering 止まり、geDIG は自由エネルギー項として β₁ を使う

[表 5.X: 対比表]

geDIG 固有の貢献は「位相を自由エネルギーに明示的に組み込む」点にある。
```

### 5.2 §7（実験計画）に H_ising-bkt を追加

Part 4 §7 (実験計画) の **優先度 4 相当**に H_ising-bkt を追加:

```markdown
**優先度 4**: H_ising-bkt（Özönder 再現 + β₁ augmentation）
- 期間: 2-4 週間
- 成功基準: β₁ 追加で臨界温度検出精度 >10% 改善
- 失敗時: β₁ は Transformer では有効でない → 2D Ising 以外で再試験
```

### 5.3 §8（棄却条件）に追加

Part 4 §8 の棄却条件表に:

| クレーム | 棄却条件 |
|---|---|
| **β₁ が Transformer 相転移指標として有用** | H_ising-bkt で attention entropy 単独より悪化 |

---

## 6. 作者の留保

### 6.1 射程の範囲

- 本メモは **Part 4 Transformer 統合の範囲内**での主張に留める
- **素粒子物理への拡張は封印**（前回の対話で作者自身が判断）
- BKT との接続は **「物性物理の相転移記述」としての類推**のみ

### 6.2 検証前の仮説

- 先行研究との差別化（entropy + β₁ vs entropy のみ）は**理論的な予測**
- 実験的に β₁ が改善をもたらすかは**未検証**
- H_ising-bkt の結果次第で、主張の強度を調整する必要

### 6.3 Part 4 自体の仮説段階

- Part 1 §8.6 の Transformer negative result が未解決
- 本メモの H_ising-bkt はその**具体的な修正方針 one instance**
- 成功すれば Part 4 の主張が強化、失敗すれば別ドメインに再配向

---

## 7. 関連リンク

### 参照元
- [../gedig_transformer_architecture.md](../gedig_transformer_architecture.md) — Part 4 全体
- [../gedig_core_theory_unified.md §5.6.4](../gedig_core_theory_unified.md) — Landau 相転移への言及
- [../gedig_core_theory_unified.md §8.6](../gedig_core_theory_unified.md) — Transformer negative result
- [../gedig_core_theory_unified.md §9.5](../gedig_core_theory_unified.md) — negative result の解釈候補
- [../gedig_cognitive_architecture.md §4.2](../gedig_cognitive_architecture.md) — BKT 相転移類推（Part 2）

### 関連する気づきメモ
- [insight_beta1_dimension_free.md](insight_beta1_dimension_free.md) — β₁ の次元フリー性（本メモの理論的基盤）
- [insight_bourbaki_three_structures.md](insight_bourbaki_three_structures.md) — 三項の数学的必然性
- [insight_morphogenetic_generality.md](insight_morphogenetic_generality.md) — 💭 形態形成への射程拡張（姉妹妄想メモ）

### 先行研究（web 検索 2026-04-17）
- [Attention to Order (Özönder 2025)](https://arxiv.org/abs/2510.07401)
- [Phase Transitions in LLMs and O(N) Model (Sun-Haghighat 2025)](https://arxiv.org/abs/2501.16241)
- [T3former (2025)](https://arxiv.org/abs/2510.13789)
- [ML-based BKT Detection (2025)](https://arxiv.org/abs/2502.09214)
- [Phase transitions in LLM compression (Nature npj AI 2026)](https://www.nature.com/articles/s44387-026-00072-8)
- [BKT transition (Wikipedia)](https://en.wikipedia.org/wiki/Berezinskii%E2%80%93Kosterlitz%E2%80%93Thouless_transition)

### 既存メモでの BKT 言及（自己整合性）
- [_archive/betti_number_adoption_memo.md §11](../_archive/betti_number_adoption_memo.md) L249
- [thinking/gedig_cognitive_foundation.md §1.2](gedig_cognitive_foundation.md) L21-26
- [gedig_cognitive_architecture.md §4.2](../gedig_cognitive_architecture.md) L199-203

---

## 8. 次のアクション

1. **Part 4 への統合**（§5.X 関連研究節、§7 H_ising-bkt、§8 棄却条件）
   - 本セッションで実施可能
2. **Özönder のコード取得**
   - arXiv 2510.07401 の supplementary or author contact
3. **H_ising-bkt の実験実施**
   - 2-4 週間、別セッション or 共同研究者
4. **結果に基づく Part 4 の主張強度調整**
