# L1 三種 Attention 設計書 — 原案（採用）

**作成日**: 2026-02-10
**Status**: Phase B 完了 / Phase C 実装中
**参照元**: `threelayer_search_implementation_plan.md` §19 までの議論 + 2026-02-10 設計セッション
**対象**: `experiments/maze/qhlib/graph_walker.py`, `attention.py`, `search_engine.py`

---

## 1. 背景と動機

### 1.1 現状の問題

Day 3 実験（§14-§19）で三層検索 C が legacy B に敗北（33% vs 100%）。
原因分析の結果、以下が判明：

1. **L1 のスコアリングが geDIG と無関係** — `attention^α × cosine_sim × σ(propagated/τ)` は ad-hoc
2. **L1 → L2 フォールバックが `if len(cands) >= N` の二値判定** — ゲーティングではない
3. **L1 の attention が geDIG 評価結果を反映していない** — decay/boost だけで F の値と無関係

### 1.2 設計原則

- **if 文でのハードコードは避ける** — 迷路以外のドメインに使えなくなる
- **AG/DG と同じゲーティング設計** — 連続値のゲート関数で制御
- **L0-L2 は同じ原理（geDIG）の精度/計算量トレードオフ**
  - L0: geDIG 結果のハッシュ引き（完全キャッシュ）
  - L1: geDIG 結果のグラフインデックス走査（ローカル近似）
  - L2: geDIG のフル計算

### 1.3 人間の迷路行動との対応

```
分岐点に立つ
  ↓
「右は前に行き止まりだった」（記憶の想起 = L1 index scan）
「左はまだ行ったことない」
  ↓
想像で判断できる → 即断（L1 採用）
想像で足りない → じっくり考える（L2 フル評価）
```

L1 は「行く先を想像する」。attention は「想起のしやすさ」であり、
その中身は **L2（geDIG）の過去の評価結果の蓄積**。

---

## 2. 三種の Attention 定義

### 2.1 一覧

| # | 名称 | 所属 | 記録タイミング | 何を表すか |
|---|------|------|--------------|-----------|
| 1 | `ag_attention` | エッジ | L2 走査時に新規接続 | **関連度** — この接続がどれくらい近かったか（類似度） |
| 2 | `dg_attention` | エッジ | DG コミット時 | **構造的価値** — geDIG がこの接続をどれくらい評価したか |
| 3 | `reward_attention` | ノード→エッジ伝播 | Sleep 相 | **方向の期待値** — この先に何があるか（正例/負例の伝播） |

### 2.2 AG Attention — 関連度

```
記録タイミング: L2 走査で候補エッジが生成された時
記録値:         TwoThresholdSelector の類似度スコア
                c-link 通過: selector が返した distance/similarity 値
                c-cand 通過: 同上
格納場所:       graph[u][v]["ag_attention"]
```

**意味**: このエッジが作られた時点で、クエリとこの候補がどれくらい近かったか。
初見時の「関連度のスナップショット」。

**更新**: 基本的に書き込みは1回（エッジ生成時）。
同一エッジが再評価された場合は max(既存値, 新値) で更新。

### 2.3 DG Attention — 構造的価値

```
記録タイミング: DG が発火してエッジがコミットされた時
記録値:         geDIG の評価スコア（g_min の符号反転 or 正規化値）
                DG 未発火のエッジ: dg_attention = 0.0（中立）
格納場所:       graph[u][v]["dg_attention"]
```

**意味**: geDIG がこの接続を「構造的に価値がある」と判断した度合い。
F が大きく負（= 大きな改善）なら dg_attention が高い。

**更新**: DG 再発火時に上書き。

### 2.4 Reward Attention — 方向の期待値

```
記録タイミング: Sleep 相で報酬伝播後
記録値:         ノードの propagated 値（Q-learning 式伝播済み）
格納場所:       graph.nodes[node]["propagated"]（既存）
エッジへの写像: エッジ (u, v) の reward_attention = propagated(target_node)
                ※ L1 走査時に動的に読む（エッジ属性にコピーしない）
```

**意味**: この方向に進んだ先の累積期待報酬。
正例/負例はヒューリスティック（Phase 1）:
- goal=+1.0, novel=+0.2, revisit=-0.4, deadend=-1.0, blocked=-1.0

**更新**: 毎 Sleep サイクルで全ノード再伝播。

---

## 3. Transformer QKV 対応

```
Transformer:
  Attention(Q, K, V) = softmax(Q·K^T / √d) · V

三層検索 L1:
  Q = current_query_vector   （今何を探してるか）
  K = ag_attention            （この接続の関連度）
  V = reward_attention        （この方向の価値）
  Gate = σ(dg_attention / τ)  （構造的確信度）
```

| Transformer | L1 三種 attention | 役割 |
|------------|-------------------|------|
| Q·K^T | sim(query, ag_attention) | 関連度マッチング |
| softmax(·/√d) | σ(dg_attention / τ) | 確信度ゲーティング |
| × V | × reward_attention | 価値による出力制御 |

**差異**: Transformer では V は出力を変調するだけで selection を制御しない。
L1 では reward_attention が selection 自体を制御する（行き止まり方向を抑制）。

---

## 4. L1 スコアリング

### 4.1 候補スコア

```
L1_score(edge) = relevance × confidence × value

  relevance  = ag_attention                     ← エッジ属性から直接読む
  confidence = σ(dg_attention / τ)              ← ゲート関数
  value      = σ(reward_attention / τ_reward)   ← ゲート関数（[0,1] 正規化）
```

3つの乗算: 一つでもゼロに近ければスコアが落ちる。

| 状況 | relevance | confidence | value | L1_score |
|------|-----------|-----------|-------|----------|
| 関連度高・DG確信・ゴール方向 | 高 | ≈1 | ≈1 | **高 → L1 即採用** |
| 関連度高・DG未評価・未知方向 | 高 | ≈0.5 | ≈0.5 | **中 → L2 に確認委任** |
| 関連度低 | 低 | - | - | **低 → 候補外** |
| DG確信・行き止まり方向 | - | ≈1 | ≈0 | **低 → 抑制** |

### 4.2 スケール正規化

各 attention を [0, 1] に収める:

- `ag_attention`: TwoThresholdSelector の類似度は既に [0, 1] 程度
- `dg_attention`: σ(·/τ) でゲート化 → [0, 1]
- `reward_attention`: σ(·/τ_reward) でゲート化 → [0, 1]

乗算の各項が [0, 1] なら、L1_score も [0, 1] に収まる。重みバランス問題は発生しない。

---

## 5. フォールバックゲート

### 5.1 設計

L1 → L2 の切り替えは if 文ではなくゲート:

```
L1_confidence = max(L1_score) − mean(L1_score)

→ 大きい: 一つの候補が明確に良い → L1 で即決
→ 小さい: 候補間に差がない → L1 に判断材料がない → L2 へ
```

これは attention 分布の「尖り具合」。Transformer の attention entropy と同じ発想。

### 5.2 ゲート関数

```
L1_gate = σ((max_score − mean_score − bias) / τ_fallback)

→ 1.0 に近い: L1 採用
→ 0.0 に近い: L2 フォールバック
→ 中間値: L1 と L2 の混合（将来拡張）
```

Phase 1 では L1_gate > 0.5 なら L1 採用、以下なら L2（二値近似）。
ただし閾値はゲート関数のパラメータ（τ_fallback, bias）で制御し、ハードコードしない。

---

## 6. 伝播ルール

| attention | 伝播 | 減衰 | 理由 |
|-----------|------|------|------|
| ag_attention | なし（エッジ固有） | 時間減衰（× decay_ag） | 文脈が変わると関連度が薄れる |
| dg_attention | なし（エッジ固有） | なし（geDIG 判断は永続） | 再評価時に上書きで更新 |
| reward_attention | Sleep で Q-learning 式伝播 | γ=0.95 で距離減衰 | 既存の propagated と同一 |

**AG → DG → Reward の記録タイミング順序**:

```
ステップ N:
  1. L2 走査 → 新規エッジに ag_attention 記録
  2. geDIG 評価 → DG 発火 → エッジに dg_attention 記録
  3. ステップ終了 → reward はノード属性に記録

Sleep 相:
  4. reward 伝播 → propagated をノードに書き込み

ステップ N+M（再訪時）:
  5. L1 が ag_attention, dg_attention, reward_attention を読んで判断
```

---

## 7. 未確定事項（実験で検証）

### Q1: ag_attention の具体値

c-cand の類似度か c-link の類似度か。閾値が異なるのでスケールが変わる。

**実験案**: 両方記録して、L1 スコアとの相関を見る。

### Q2: dg_attention の具体値

g_min をそのまま使うか、符号反転するか、正規化するか。
g_min は負が「良い」なので σ(-g_min/τ) が自然。

**実験案**: g_min を生値でエッジに記録し、σ 変換は L1 スコアリング時に行う。

### Q3: reward_attention のエッジへの写像

ノード属性の propagated をどうエッジに対応させるか。
- `propagated(target_node)` — 移動先の価値
- `max(propagated(u), propagated(v))` — 両端の最良値

**実験案**: `propagated(target_node)` をまず採用（方向性が明確）。

### Q4: 3項のスケールバランス

乗算で合成する場合、一項の分散が大きいと支配的になる。
各項を [0, 1] に σ で正規化する設計だが、τ の値で分布が変わる。

**実験案**: 各項の分布を 1 seed 分のログから確認し、τ を調整。

---

## 8. 段階的実験計画

### 設計方針

**一度に全部変えない。一つずつ足して、各 attention の効果を個別に確認する。**

### Phase A: 計測のみ（コード変更最小）

**目的**: 3 種の attention 値を記録するだけ。L1 のスコアリングは変えない。

```
[A-1] ag_attention をエッジに記録する（edges.py / commit.py）
[A-2] dg_attention をエッジに記録する（commit.py）
[A-3] ステップログに 3 値を出力する
[A-4] 1 seed × 500 steps で値の分布を確認
```

**期待出力**:
- ag_attention のヒストグラム（c-link/c-cand 起源の分布差）
- dg_attention のヒストグラム（DG 発火/非発火の分布）
- reward_attention のヒストグラム（propagated の分布）
- 3 値間の相関行列

**判断基準**: 3 値に十分な分散があり、互いに独立な情報を持つこと。

**Phase A 結果（2026-02-10 実施済み）**: → 詳細は計画書 §20A 参照

| attention | range | stdev | 判定 |
|-----------|-------|-------|------|
| ag_attention (similarity) | [0.95, 1.00] | 0.01 | 迷路では実質定数（二値）。RAG で効く |
| dg_attention (g0) | [-0.50, +0.50] | 0.06 | **十分な分散** |
| reward_attention (propagated) | [0.00, 1.00] | 0.01 | **WSW 後のみ有効** |

**結論**: 迷路では `L1_score ≈ const × σ(dg/τ) × σ(rw/τ)` の 2 変数式。
ag は迷路特性上定数だが、設計として正しい（汎用ドメインで分散が出る）。

### Phase B 結果 (並行スコアリング)

- **3att dynamic range**: 0.085–0.783 (legacy: 0.230–0.619) → 2.4× wider
- **相関**: r=0.52 → 部分的重複、相補的信号
- **ag 飽和**: 迷路では実質 2-channel (dg × reward)
- **Warmup 相転移**: Step 200 で reward 100% 浸透
- **Transformer × GNN 融合**: GNN トポロジー上の情報理論的 cached attention

### Phase B: L1 スコアリング変更

**目的**: L1 のスコアを三種 attention ベースに変更し、legacy と比較。

```
[B-1] graph_walker.py の effective_score を 3-attention 式に変更
[B-2] 旧スコアとの並行出力（両方ログに記録、判断は旧スコアで行う）
[B-3] 3 seeds × 500 steps で新旧スコアの乖離を確認
[B-4] 新スコアで判断を切り替えて B vs C 比較（共通ベースライン）
```

**期待出力**:
- 新旧スコアの相関（r² > 0.8 なら既存と大差なし、<0.5 なら異なる情報）
- B vs C 成功率比較

### Phase C: フォールバックゲート

**目的**: L1 → L2 のゲート関数を導入し、フォールバック精度を検証。

```
[C-1] search_engine.py に L1_confidence ゲートを実装
[C-2] ゲートなし（現状）vs ゲートあり の比較
[C-3] τ_fallback の感度分析（0.1, 0.3, 0.5, 1.0）
[C-4] 5 seeds × 500 steps で B vs C_gated 比較
```

**期待出力**:
- L1 採用率 vs ゲートパラメータの関係
- ゲートあり C がゲートなし C を上回ること
- 理想: ゲートあり C ≥ B（legacy）

### Phase D: 統合実験

**目的**: Phase A-C の最良設定で 30 seeds 実験。

```
[D-1] 25x25 × 30 seeds × 500 steps
[D-2] legacy vs threelayer_3att 比較
[D-3] 成功率、ステップ数、L1 ヒット率、β₁ 推移
```

---

## 9. 既存コードへの変更マップ

| ファイル | Phase A | Phase B | Phase C |
|----------|---------|---------|---------|
| `qhlib/attention.py` | ag/dg 記録メソッド追加 | — | — |
| `qhlib/graph_walker.py` | — | effective_score 変更 | — |
| `qhlib/search_engine.py` | — | — | フォールバックゲート |
| `qhlib/commit.py` | dg_attention 書き込み | — | — |
| `qhlib/edges.py` | ag_attention 書き込み | — | — |
| `qhlib/models.py` | StepRecord にログ追加 | — | — |
| `run_experiment_query.py` | ログ出力追加 | スコアリング切替 | ゲート有効化 |

---

## 10. 関連文書

- **実験経過**: `threelayer_search_implementation_plan.md` §14-§19
- **L1 失敗分析**: `discussion_l1_fallback_analysis.md`
- **geDIG 正準定義**: `docs/gedig_spec.md`
- **Graph-Persistent DG 仕様**: `experiments/maze/graph_persistent_dg/SPEC.md`
- **三層検索根拠仕様**: `docs/research/thinking/memory_search_implementation_20260208.md`

---

**End of Document**
