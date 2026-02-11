# HotPotQA v2 — Betti 拡張ゲージによる Multi-hop QA 実験

**作成日**: 2026-02-10
**Status**: 仕様策定中
**前提**: `experiments/_archive_before_20260201_refactor/hotpotqa-benchmark/`（v1）

---

## 0. 動機

### v1 の結果と残された問い

v1 実験（2026-01）では、geDIG の AG/DG ゲーティングを BM25 検索 + GPT-4o-mini に適用し、HotPotQA dev 全 7,405 問で評価した。

| 手法 | EM | F1 | SF-F1 |
|------|---:|---:|------:|
| Closed-book | 22.0% | 35.1% | 0.0% |
| BM25 | 36.6% | 52.3% | 35.0% |
| Static GraphRAG | **38.9%** | **55.9%** | 35.0% |
| geDIG v1 | 37.5% | 53.8% | 30.4% |

geDIG は BM25 を上回ったが、Static GraphRAG に僅差で負けた。特に **bridge 型質問**（2 つの独立文書を橋渡しする推論）で Win/Loss が拮抗。

### 仮説: β₀ の欠落が bridge 型の弱さの原因

v1 のゲージは ΔSP（最短路短縮）のみ。トポロジー的に見ると：

```
検索結果が 2 つの島に分かれている場合:

  [Q]---[Doc A]    [Doc B]---[Doc C]     β₀ = 2, β₁ = 0
                                          ΔSP = undefined（非連結）
                                          → v1 では構造変化を検出できない

橋渡し検索で島が統合された場合:

  [Q]---[Doc A]---[Doc B]---[Doc C]     β₀ = 1, β₁ = 0
                                          Δβ₀ = -1 ← 構造的に重要！
                                          Δβ₁ = 0  ← ループはまだない
```

**β₀（連結成分数）の減少は、断片知識の統合を意味する。** v1 のゲージはこれを見逃していた。

---

## 1. 拡張ゲージ v5

### 1.1 式

```
F = ΔEPC_norm − λ ( ΔH_norm + γ₁·Δβ₁ − γ₀·Δβ₀ )
```

| 項 | 数学的構造 | 計測対象 | 符号の意味 |
|----|-----------|---------|-----------|
| ΔEPC_norm | メトリック（距離） | 再構成コスト | + = コスト増 |
| ΔH_norm | 測度（確率） | エントロピー変化 | - = 秩序化 |
| Δβ₁ | トポロジー（ループ） | 独立閉路の変化 | + = 新ループ（良い） |
| Δβ₀ | トポロジー（島） | 連結成分の変化 | - = 島統合（良い） |

### 1.2 符号の整合性

F < 0（コミットすべき）になる条件：

| イベント | Δβ₀ | Δβ₁ | 寄与 (−γ₀Δβ₀ + γ₁Δβ₁) | F への効果 |
|---------|---:|---:|:----------------------:|:---------:|
| 島の統合 | -1 | 0 | +γ₀ | F 減少 |
| ループ生成 | 0 | +1 | +γ₁ | F 減少 |
| 孤立島の追加 | +1 | 0 | -γ₀ | F 増加 |
| ループ破壊 | 0 | -1 | -γ₁ | F 増加 |

すべて直感と一致する。

### 1.3 後方互換性

- 連結グラフ（迷路など）: 常に β₀ = 1 → Δβ₀ = 0 → γ₀ 項消滅 → v4 と同等
- γ₀ = 0 設定時: v4 互換モード
- γ₁ = 0, γ₀ = 0 設定時: 旧式 F = ΔEPC − λ·ΔH に退化

### 1.4 二段ゲート（AG/DG）の拡張

```
0-hop:     g₀   = ΔEPC_norm − λ · ΔH_norm
multi-hop: g_min = min_h { ΔEPC_norm − λ(ΔH_norm + γ₁·Δβ₁^(h) − γ₀·Δβ₀^(h)) }
```

AG は 0-hop（トポロジー項なし）のまま。DG でのみ β₀/β₁ が効く。

> **設計判断**: AG は「驚き」の検出であり、トポロジーは「構造的統合の質」の評価。
> 役割が異なるので、トポロジー項は DG にのみ含める。

---

## 2. 前提: メインコード β₀ 昇格

本実験はメインコードの β₀/β₁ 対応が前提。
設計詳細は **[`docs/design/betti1_threelayer_implementation.md` §11](../../docs/design/betti1_threelayer_implementation.md)** を参照。

要点:
- `compute_betti_numbers(g) -> (β₀, β₁)` をメインコードに追加
- `GeDIGConfig` に `structural_mode = "sp" | "betti" | "betti_full"`, `gamma_0`, `gamma_1` を追加
- `gedig_core.py` の F 計算を `structural_mode` で分岐
- デフォルトは `structural_mode="sp"` で後方互換

---

## 3. HotPotQA v2 実験設計

### 3.1 変更点（v1 → v2）

| 項目 | v1 | v2 |
|------|----|----|
| 構造項 | ΔSP（最短路短縮） | Δβ₁ − Δβ₀（ベッチ数） |
| ゲージ式 | F = ΔEPC − λ(ΔH + γ·ΔSP) | F = ΔEPC − λ(ΔH + γ₁Δβ₁ − γ₀Δβ₀) |
| geDIG 実装 | archive 内のローカル GeDIGCore | **メインコード**の GeDIGCore |
| Python 環境 | 独立 | プロジェクト共通 `.venv` |
| ベースライン | 6 種 | BM25 + Static GraphRAG（2 種に絞る） |

### 3.2 実験条件

| 条件名 | structural_mode | γ₀ | γ₁ | 目的 |
|--------|----------------|---:|---:|------|
| A: SP baseline | `sp` | - | - | v1 再現（ΔSP のみ） |
| B: β₁ only | `betti` | 0 | 1.0 | ループ項のみ |
| C: β₀ only | `betti_full` | 1.0 | 0 | 島統合項のみ |
| D: β₀ + β₁ | `betti_full` | 1.0 | 1.0 | フル Betti |
| E: β₀ + β₁ (tuned) | `betti_full` | tune | tune | dev-500 で最適化 |

### 3.3 評価指標

**主指標:**
- EM (Exact Match)
- F1 (Token-level)
- SF-F1 (Supporting Facts F1)

**分析指標:**
- **Question type 別** (bridge / comparison) — bridge での改善が仮説
- **AG/DG 発火率** — β₀ が DG の判断をどう変えるか
- **Δβ₀ 分布** — 島統合イベントの頻度と F 値の関係
- **Win/Loss 分析** — v1 の Loss ケースが v2 で Win に転じるか

### 3.4 データセット

| セット | サイズ | 用途 |
|--------|-------|------|
| dev-100 | 100 問 | 開発・デバッグ（mock LLM 可） |
| dev-500 | 500 問 | 閾値チューニング |
| dev-full | 7,405 問 | 最終評価 |

### 3.5 コスト見積もり

| フェーズ | API 呼び出し | 推定コスト |
|---------|------------|----------|
| 条件 A-D (dev-500) | 500 × 4 = 2,000 | ~$5-10 |
| 条件 E チューニング | ~1,000 | ~$3 |
| 最終評価 (dev-full) | 7,405 × 2 | ~$30-50 |
| **合計** | | **~$40-65** |

---

## 4. ディレクトリ構成

```
experiments/hotpotqa_v2/
├── SPEC.md              ← 本ドキュメント
├── configs/
│   ├── condition_a_sp.yaml
│   ├── condition_b_beta1.yaml
│   ├── condition_c_beta0.yaml
│   ├── condition_d_betti_full.yaml
│   └── condition_e_tuned.yaml
├── src/
│   ├── adapter.py       ← メインコードの GeDIGCore を使う adapter
│   ├── retriever.py     ← BM25 検索
│   ├── graph_builder.py ← 検索結果 → ナレッジグラフ
│   ├── answerer.py      ← LLM 回答生成
│   └── evaluator.py     ← EM/F1/SF-F1 計算
├── scripts/
│   ├── run_experiment.py
│   ├── run_baseline.py
│   └── analyze_results.py
├── data/                ← .gitignore, ダウンロードスクリプトで取得
└── results/             ← .gitignore
```

### v1 からの再利用

| v1 コンポーネント | v2 での扱い |
|------------------|-----------|
| `GeDIGCore` | ❌ archive 版は使わない。メインコードを使う |
| BM25 retriever | ✅ ロジック再利用（メインコードに依存しない） |
| Graph builder | ⚠️ β₀ を意味あるものにするためグラフ構築を調整 |
| LLM answerer | ✅ ほぼそのまま（GPT-4o-mini） |
| Evaluator | ✅ そのまま再利用 |
| 閾値チューニング | ⚠️ γ₀, γ₁ も探索空間に追加 |

---

## 5. 実装フェーズ

### 前提: メインコード β₀ 昇格

→ `docs/design/betti1_threelayer_implementation.md` §11 / Phase 3 で実施。
本実験は `structural_mode="betti_full"` が使える状態を前提とする。

### Phase 1: HotPotQA v2 実装（2-3 日）

1. adapter.py: メインコード GeDIGCore のラッパー
2. retriever.py + graph_builder.py: v1 から移植・調整
3. answerer.py: GPT-4o-mini 呼び出し
4. dev-100 で mock LLM デバッグ
5. 条件 A（SP baseline）で v1 結果を再現

### Phase 2: 実験実行（1-2 日）

1. 条件 B-D を dev-500 で実行
2. β₀ の効果を bridge/comparison 別に分析
3. 条件 E: γ₀, γ₁ チューニング
4. dev-full で最終評価

---

## 6. 成功基準

| 指標 | 基準 | 根拠 |
|------|------|------|
| Bridge 型 F1 | v1 比 +2pt 以上 | β₀ が島統合を検出し、bridge 推論を改善 |
| 全体 F1 | Static GraphRAG 以上 | v1 での -0.5pt ギャップを解消 |
| Comparison 型 | v1 と同等以上 | β₀ 追加で悪化しないこと |

### 失敗した場合の学び

- β₀ が効かない → RAG のグラフが常に連結（島が稀）→ グラフ構築の見直しが必要
- β₁ が効かない → 小グラフでは離散的すぎる → 正規化手法の検討
- 両方効かない → RAG タスクではトポロジーより情報量（ΔH）が支配的 → 理論的示唆

---

## 7. 関連ドキュメント

| ドキュメント | 内容 |
|------------|------|
| [`docs/design/betti1_threelayer_implementation.md`](../../docs/design/betti1_threelayer_implementation.md) §11 | β₀ 昇格の実装設計（本実験の前提） |
| [`docs/design/betti1_threelayer_implementation.md`](../../docs/design/betti1_threelayer_implementation.md) §11.6 | gedig_spec.md v5 更新案 |
| [`docs/gedig_spec.md`](../../docs/gedig_spec.md) | 現行ゲージ仕様（v4） |
| [`experiments/_archive_before_20260201_refactor/hotpotqa-benchmark/`](../_archive_before_20260201_refactor/hotpotqa-benchmark/) | v1 実験コード |

---

## 更新履歴

- 2026-02-10: 初版作成。動機、拡張ゲージ v5、実験設計。メインコード変更は設計書に統合。
