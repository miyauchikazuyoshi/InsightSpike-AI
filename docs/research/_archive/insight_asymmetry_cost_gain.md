# 気づきメモ: 3つの読み方を貫く cost/gain 非対称性

**日付**: 2026-04-17  
**ステータス**: 気づきメモ（既存の `gedig_formula_three_readings_20260306.md` の補完・発展）  
**関連**: [gedig_formula_three_readings_20260306.md](gedig_formula_three_readings_20260306.md) / [../gedig_core_theory_unified.md §3](../gedig_core_theory_unified.md) / [../gedig_core_theory_unified.md §6](../gedig_core_theory_unified.md)

---

## 0. 既存メモとの関係（前置き）

**重要**: F 式の Helmholtz 自由エネルギーとの対応（`(EPC - B) - H`）は、
[gedig_formula_three_readings_20260306.md](gedig_formula_three_readings_20260306.md) §4 で既に
「読み 2: 物理学的」として詳細に論じられている。

本メモは、その既存の議論を**前提として受け継ぎ**、以下の **新視点**に絞って補完する:

1. **3つの読み方すべてに共通する「cost/gain 非対称構造」**の不変性
2. **Bourbaki 三大構造（計量・測度・位相）との接続**
3. **λ を情報温度として動的制御する可能性**（simulated annealing との類縁）

既存の Helmholtz 類推そのものは、three_readings §4 に委ねる。

---

## 1. 気づきの起点

`gedig_core_theory_unified.md` §4 の本文化で、三項を「計量・測度・位相」に対応付けた後、
F 式の構造を見直すと、**3つの読み方すべて**で**計量が cost 側に単独**、**測度+位相が gain 側**という非対称が保存されていることに気づいた。

---

## 2. 3つの読み方を貫く非対称構造

### 2.1 3つの読み方の再整理（括り方の違い）

`gedig_formula_three_readings_20260306.md` から引用すると:

| 読み | 括り方 | 主な意味 |
|---|---|---|
| 1（経済学的、canonical） | `EPC - (H + B)` | cost - gain |
| 2（物理学的） | `(EPC - B) - H` | 構造収支 - エントロピー |
| 3（生物学的） | `(EPC - H) - B` | 内部状態 - 位相秩序 |

3つとも**同じ式**だが、**何を「コスト側」「利得側」として括るか**が違う。

### 2.2 cost/gain の対応表（読みごとに整理）

各読みで「計量 / 測度 / 位相」がどちら側に割り当てられるかを整理:

| 読み | cost 側 | gain 側 | cost 側の数学構造 | gain 側の数学構造 |
|---|---|---|---|---|
| 1. 経済学的 | `EPC` | `H + B` | 計量 | 測度 + 位相 |
| 2. 物理学的 | `EPC − B` | `H` | 計量 − 位相 | 測度 |
| 3. 生物学的 | `EPC − H` | `B` | 計量 − 測度 | 位相 |

### 2.3 不変量: 計量は cost 側、測度+位相は gain 側（読み 1 のみ）

**読み 1（canonical）** では明確に:
- 計量 (`EPC`) → **cost 側に単独**
- 測度 (`H`) + 位相 (`B`) → **gain 側に複合**

これは Bourbaki 三大構造の視点（[insight_bourbaki_three_structures.md](insight_bourbaki_three_structures.md)）から見ると、
「**距離の公理に従う量はコスト、情報と位相の量は利得**」という**非対称な役割分担**を示唆する。

### 2.4 読み 2, 3 における「移項」の解釈

読み 2, 3 は読み 1 から**位相項 B または測度項 H を移項**した形。
これは**代数的には同値**だが、**意味論的には異なる視点**:

- 読み 2 = 位相を構造コストに**内包**（骨格の変化として見る）
- 読み 3 = 測度を内部状態に**内包**（情報再編成として見る）

つまり、**正準読解は読み 1 であり、読み 2, 3 は導出された副次的読解**。
この順序は `three_readings §7` の「正準読解は 1 つ」原則と整合する。

---

## 3. λ の情報温度としての動的解釈（新視点）

既存の three_readings §4.2 では、λ は Helmholtz の温度 T に対応する**静的なパラメータ**として論じられている。
本メモで**補完**するのは、λ の**動的制御**の可能性。

### 3.1 Simulated annealing との類縁

受容判定 `F < 0` を展開すると:
```
F = EPC - λ · ΔIG < 0
  ⟺ ΔIG / EPC > 1 / λ
```

λ は「受容閾値の逆温度」として機能する。これは **simulated annealing (SA)** と構造的に一致:
- SA の受容確率: `P = exp(−ΔE/T)`
- geDIG の受容判定: `F < 0`（決定的閾値版）

### 3.2 Wake-Sleep-Wake での温度スケジューリング

- **Wake 初期（探索）**: λ 高 → 情報利得の重視 → 多様な探索
- **Sleep（consolidation）**: λ 徐々に低下 → 安定構造を選好
- **Wake' （収束）**: λ 低 → 確定構造の活用

これは現在の quantile-calibrated な**静的閾値設定**を、
**動的温度制御**に拡張できる可能性を示唆する。

### 3.3 実装候補

```python
# Wake 段階で λ をスケジュール
λ_wake_init = 1.0
λ_sleep_decay = 0.95  # per consolidation step
λ_wake_final = 0.1

# SA 類似の annealing schedule
for episode in training:
    wake_phase(λ=λ_wake_init)
    sleep_phase()
    λ_wake_init *= λ_sleep_decay
    λ_wake_init = max(λ_wake_init, λ_wake_final)
```

### 3.4 検証可能な仮説

**H_temp**: λ の温度スケジューリングは、固定 λ よりも迷路/RAG の探索効率を改善する。

実験:
- 迷路 15×15 / 25×25 / 51×51 で λ 固定 vs スケジュール
- 線形減衰、指数減衰、log 減衰の比較
- 成功率、ステップ数、revisit 率を測定

---

## 4. 論理的含意

### 4.1 three_readings と本メモの役割分担

- **three_readings**: F 式の**3 つの読み方の原理**を論じる（正準 + 副次）
- **本メモ**: 3つの読み方を貫く**非対称構造**と **λ 動的制御の可能性**を論じる

両者は**補完関係**にあり、どちらを先に読んでも理解可能。

### 4.2 §6.2（物理学的読解）の本文への反映

`gedig_core_theory_unified.md` §6.2 を本文化する際、以下の構成を推奨:

1. three_readings §4 の Helmholtz 類推をそのまま引用
2. 本メモの「読み 1 での非対称構造」を補足として追加
3. 本メモの「λ の情報温度としての動的解釈」を発展的話題として記載
4. **operational correspondence** の明記（FEP 類推と同じ留保）

---

## 5. 注意: FEP 類推との差別化

- FEP: variational free energy → AI 一般の統合原理として過剰主張されがち
- 本メモ: F 式の**構造的解釈**のみ、一般原理には踏み込まない
- 類推は operational level に留め、厳密な熱力学的同等性は主張しない

---

## 6. 関連リンク

### 既存メモ（本メモが補完する対象）
- [gedig_formula_three_readings_20260306.md](gedig_formula_three_readings_20260306.md) **§4（読み 2: 物理学的）**— Helmholtz-like 読解の原典、本メモはこれを前提として補完
- [gedig_formula_three_readings_20260306.md](gedig_formula_three_readings_20260306.md) §7 — 正準読解と副次読解の原則

### 参照元（本メモが関係する節）
- [../gedig_core_theory_unified.md §3.1](../gedig_core_theory_unified.md) — 正準形と簡約形
- [../gedig_core_theory_unified.md §3.2](../gedig_core_theory_unified.md) — F < 0 の意味（自発過程との対応）
- [../gedig_core_theory_unified.md §6.2](../gedig_core_theory_unified.md) — 物理学的読解（予定）
- [../gedig_origin_story.md](../gedig_origin_story.md) §「熱力学との対応」— 最初の同型指摘

### 関連する新規気づきメモ
- [insight_bourbaki_three_structures.md](insight_bourbaki_three_structures.md) — 三項が数学の3基本空間に対応する根拠（本メモの §2.3 の Bourbaki 接続を支える）
- [insight_three_terms_orthogonality.md](insight_three_terms_orthogonality.md) — 三項独立性（非対称構造が保存されるためには独立性が前提）

### open problem への接続
- [../gedig_core_theory_unified.md 付録 D](../gedig_core_theory_unified.md) — 構造 ≡ 確率 の等価性
  - 本メモの「計量は cost 側、測度+位相は gain 側」は、等価原理下で何を意味するか open
  - 仮説: 等価原理が成立するなら、計量と (測度+位相) は Legendre 対となる ?

### 将来の検証
- 温度スケジューリング H_temp の迷路実験（本メモ §3.4）
