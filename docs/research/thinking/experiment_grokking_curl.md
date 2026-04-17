# 実験プロトコル: H_grokking-curl — Grokking 相転移での β₁ + curl 観測

**日付**: 2026-04-17  
**ステータス**: ☀ **実行可能な実験プロトコル**（experiment 接頭辞、insight ではなく action 候補）  
**優先度**: **最優先**（H_ising-bkt より上位、Part 1 §8.6 negative result への具体的対処）  
**関連**: [insight_transformer_phase_transition_landscape.md](insight_transformer_phase_transition_landscape.md) / [insight_beta1_dimension_free.md](insight_beta1_dimension_free.md) / [../gedig_cognitive_architecture.md §3](../gedig_cognitive_architecture.md) / [gedig_cognitive_foundation.md §8](gedig_cognitive_foundation.md)

---

## 0. 位置付けと背景

### 0.1 起点

ユーザーの洞察:
> モデルが急に賢くなるタイミング（grokking）と attention の渦度（curl）の相関を測る

この着想は、以下の既存要素の**自然な統合**:

1. **Grokking**: 機械学習最大の謎の一つ（Power et al. 2022, Nanda 2023）
2. **geDIG の β₁**: 位相不変量、相転移の signal（Part 1 §5）
3. **curl 検出**: attention flow の渦度（Part 2 §3.3）
4. **既存 TODO の実装機会**: [gedig_cognitive_foundation.md §8](gedig_cognitive_foundation.md) で既に記録された
   - 「Attention行列からの渦度計算の定式化」
   - 「Transformerのattention flowに渦構造があるか可視化」
   - 「curl-based attention の実装と効果測定」
   - 「geDIG AG/DGへのcurl項の統合」

**本メモは、これらの既存 TODO を grokking という具体的現象で検証する実験設計**。

### 0.2 本メモの独自貢献

- **Grokking 観測との結合**: 既存 curl 議論は「理論的」「認知的」側面が中心だった。本メモは**工学的検証プロトコル**として具体化
- **先行研究との位置取り**: β₁ × grokking は既に研究されている（TAG-DS 2025）が、**curl × grokking は未研究**
- **Part 1 §8.6 negative result への対処**: Transformer `delta_r2_struct` 全モデル負の問題に、β₁ + curl の組み合わせで対処可能か検証

---

## 1. 仮説 H_grokking-curl

> **H_grokking-curl**: Grokking の相転移時点（test loss 急降下）において、
> (1) attention graph の `β₁` が離散ジャンプし、
> (2) attention の `curl`（Part 2 §3.3 階層 3）が急増し、
> (3) geDIG のスカラー F が符号変化する。
> これらは従来の weight norm / gradient norm より**早期に** grokking を予兆する。

### 1.1 サブ仮説

- **H_β1**: `β₁` ジャンプは grokking 点で観測される
- **H_curl**: `curl(attention)` は grokking 点で急増する
- **H_F**: スカラー F は grokking 点で符号変化する
- **H_early**: β₁ / curl は weight norm より 10 epoch 以上早く signal を出す
- **H_integration**: β₁ + curl + ΔH の統合（F）は単独指標より高精度

---

## 2. 既存 curl 議論の参照

本実験は、既存メモで積み重ねてきた curl 議論の**実装・検証フェーズ**。

### 2.1 curl の階層的定義（[Part 2 §3](../gedig_cognitive_architecture.md)）

```
階層 1: 連続空間の数学的定義 (∇ × v, curl 極大 かつ div ≠ 0)
階層 2: FEP の認知段階 (予測フェーズ = curl 検出)
階層 3: 実装上の attention flow (attention 行列を vector field として curl 計算)
```

**本実験は階層 3 の初の工学的検証**。

### 2.2 既存の実装 TODO（[cognitive_foundation §8](gedig_cognitive_foundation.md)）

既に記録されているが未実装:
- [ ] 離散グラフ上の curl の厳密な定義
- [ ] graph curl = circulation / cycle flow としての定式化
- [ ] Attention行列からの渦度計算の定式化
- [ ] Transformerのattention flowに渦構造があるか可視化
- [ ] curl-based attention の実装と効果測定
- [ ] geDIG AG/DGへのcurl項の統合

**本実験はこれらを grokking 観測で一気に検証する**。

### 2.3 離散 curl の計算方法（[prediction_curl §4.2](gedig_prediction_curl.md)）

Attention 行列 `A ∈ R^{T×T}` から curl を計算する案:
1. A を vector field として解釈: ノード i の「流出ベクトル」 `v_i = Σ_j A[i,j] · (x_j - x_i)`
2. 反対称成分から渦度計算: `curl_i = Σ_{j<k} (A[i,j]·A[j,k] - A[i,k]·A[k,j])`
3. Hodge 分解による離散 curl（より厳密）: Gradient + Curl + Harmonic 成分に分解

**本実験では方法 2（簡易実装）から始め、必要なら方法 3 へ進む**。

---

## 3. 先行研究マップ

Grokking × 相転移 × 位相の最新研究（[insight_transformer_phase_transition_landscape.md §1](insight_transformer_phase_transition_landscape.md) より抜粋）:

| 研究 | 発見 | curl への言及 |
|---|---|---|
| Nanda et al. 2023 (arXiv 2309.02390) | modular addition で Fourier circuit 発見 | なし |
| TAG-DS 2025 (Betti-Fiedler proxy) | β₁ が grokking の proxy | **なし** ← geDIG 独自性 |
| Liu et al. 2023 Omnigrok | 様々な設定で grokking 一般化 | なし |
| Humayun et al. 2024 | 決定境界の時系列変化 | なし |
| arXiv 2604.04655 (2026) | Grokking = 次元相転移 (D<1 → D>1) | なし |
| arXiv 2603.05228 | Architectural topology で grokking 制御 | なし（位相言及のみ） |
| OpenReview 3ROGsTX3IR | Grokking = 1 次相転移 | なし |
| arXiv 2603.13331 | Norm-Separation Delay Law | なし |

**curl(attention) を grokking で測定した研究は私の検索では見つからない** → **geDIG 独自の貢献機会**。

---

## 4. geDIG 独自性（対比表）

| 指標 | 既存研究 | 本実験（H_grokking-curl） |
|---|---|---|
| β₁（位相数） | TAG-DS 2025 が proxy として使用 | **自由エネルギー項として統合** |
| curl（渦度） | **未研究** | **初の計測** |
| ΔH（entropy） | Özönder 2025 で相転移検出 | β₁ + curl と統合 |
| 次元 d_eff | arXiv 2604.04655 で次元相転移 | `d_eff = β₁/V + 1` で定義 |
| 統合指標 F | なし | **F = ΔEPC - λ(ΔH + γΔβ₁)** |

**独自性の核心**:
- 既存: 単一指標で grokking を観測
- geDIG: **4 指標を F として統合 + curl を新規追加**
- **curl は Part 2 §3.3 の初の工学検証**

---

## 5. データソース

公開済み、即入手可能:

### 5.1 Nanda et al. 2023 Modular Addition (最有力)
- GitHub: [neelnanda-io/Easy-Transformer](https://github.com/neelnanda-io/Easy-Transformer) または同等レポ
- データ: `mod p` addition、`p = 113` など
- モデル: 1-layer Transformer、最もクリアな grokking
- **最初に再現すべき**

### 5.2 TAG-DS 2025 β₁ 研究
- 論文: Betti-Fiedler partition for grokking detection
- コード公開の可能性（要確認）
- 本実験の baseline として有用

### 5.3 Humayun et al. 2024
- 決定境界の時系列データ
- 本実験では補助データ

### 5.4 自前データ（必要時）
- Omnigrok 的なセットアップを複数作成
- 多様なタスク（modular arithmetic、parity、小規模言語モデル）

---

## 6. 実験プロトコル

### Phase A: Grokking 再現（1-2 週間）

1. Nanda の modular addition grokking を再現
2. Train / test loss の時系列を保存
3. **Grokking 点（test loss 急降下）を明確に同定**
4. 標準指標（weight norm, gradient norm）との相関を確認

**成功確認**: test loss が epoch N で train loss に追いつく現象を再現

### Phase B: β₁ / curl の時系列測定（2-3 週間）

5. 各 epoch で attention weight を取得
6. **閾値化 or top-k でグラフ化**（§6.1 で詳述）
7. 層別に以下を測定:
   - `β₁ = E - V + C` (networkx で O(V+E))
   - `curl` (§2.3 方法 2 で計算)
   - `ΔH` (attention entropy)
   - `ΔEPC` (連続エッジ追加 / 削除コスト)
8. 時系列でプロット（epoch vs 各指標）

**注意**: attention は T×T の密な行列。閾値化の選択が結果に影響 → sensitivity analysis 実施。

### Phase C: 先行性の検証（1 週間）

9. β₁ / curl の急変時点を epoch で特定
10. weight norm / gradient norm / train loss との時間差を計算
11. **H_early: β₁ / curl が weight norm より 10+ epoch 早いか** を検証

### Phase D: F の統合（1 週間）

12. F = ΔEPC - λ(ΔH + γΔβ₁) を時系列計算
13. λ, γ の sensitivity analysis（0.1, 0.5, 1.0, 2.0）
14. F の符号変化点と grokking 時点の対応を検証
15. AG/DG 発火統計（F > θ_AG, F < θ_DG）との照合

### Phase E（拡張、任意）: 複数ドメインでの再現（2 週間）

16. Modular addition 以外（parity, 小規模言語モデル）で Phase A-D を繰り返す
17. β₁ / curl のパターンが**ドメインに依存する / しない**を判定
18. **スケール不変性**の裏付け

---

## 6.1 実装詳細

### attention のグラフ化

```python
def attention_to_graph(A: torch.Tensor, threshold_method='topk', k=5, th=None):
    """
    Attention 行列 A (T×T) をグラフ G (V=T, E={(i,j) | A[i,j] > threshold}) に変換
    """
    if threshold_method == 'topk':
        # 各トークンについて上位 k 個を edge に
        _, indices = torch.topk(A, k=k, dim=1)
        edges = [(i, int(j)) for i in range(A.shape[0]) for j in indices[i]]
    elif threshold_method == 'percentile':
        th = torch.quantile(A.flatten(), 1 - k/A.shape[0])
        edges = [(i, j) for i in range(A.shape[0]) for j in range(A.shape[1]) if A[i,j] > th]
    # ... (networkx で Graph 構築)
    return G
```

### β₁ 計算

```python
def compute_beta1(G):
    import networkx as nx
    V = G.number_of_nodes()
    E = G.number_of_edges()
    C = nx.number_connected_components(G)
    return E - V + C  # β₁ = E - V + C
```

### curl 計算（簡易版、§2.3 方法 2）

```python
def compute_attention_curl(A: torch.Tensor):
    """
    Attention 行列の反対称成分から渦度を計算
    curl_i = Σ_{j<k} (A[i,j]·A[j,k] - A[i,k]·A[k,j])
    """
    T = A.shape[0]
    curl = torch.zeros(T)
    for i in range(T):
        for j in range(T):
            for k in range(j+1, T):
                curl[i] += A[i,j]*A[j,k] - A[i,k]*A[k,j]
    return curl  # 各トークンの渦度
```

より厳密な版は離散 Hodge 分解（将来実装候補）。

### scalar F 計算

```python
def compute_F(EPC_prev, EPC_curr, H_prev, H_curr, b1_prev, b1_curr, lam=1.0, gam=1.0):
    dEPC = EPC_curr - EPC_prev
    dH = H_curr - H_prev
    db1 = b1_curr - b1_prev
    return dEPC - lam * (dH + gam * db1)
```

---

## 7. 成功条件 / 棄却条件

### 成功条件（geDIG の有効性を支持）

- **H_β1 ✓**: β₁ が grokking 点で 3σ 以上の離散ジャンプ
- **H_curl ✓**: curl が grokking 点で 3σ 以上の急増
- **H_F ✓**: F の符号変化が grokking 時点と ±5 epoch 以内
- **H_early ✓**: β₁ / curl が weight norm より 10 epoch 以上早く signal
- **H_integration ✓**: F が単独指標より高精度で grokking を予測（AUC > 0.85）

**最低条件**: 上記のうち 3 つ以上が成立

### 棄却条件（geDIG の Transformer 適用を棚上げ）

- β₁ / curl が grokking 点で無反応
- 結果が既存 β₁ proxy 研究（TAG-DS 2025）と同じで curl の追加価値なし
- F が noise 的で相関が弱い（|r| < 0.3）
- λ, γ の choice に過度に敏感（結果が不安定）

**棄却時の対処**: Part 4 Transformer 統合は別ドメイン（RAG / 迷路）に再配向

### 曖昧な結果（追加実験が必要）

- 一部仮説のみ成立 → ドメイン依存性の検証（Phase E）
- β₁ のみ成立、curl 不成立 → 既存研究と同等、独自性弱い → Part 4 の主張調整

---

## 8. 実施コスト・スケジュール

### 8.1 コスト

- **計算**: 小規模（modular addition は 1-layer Transformer）
- **GPU**: 1 枚で十分、RTX 3090 / 4090 級
- **期間**: Phase A-D で 5-7 週間、Phase E で +2-3 週間
- **人員**: 1 名（ML + networkx + numpy / torch）

### 8.2 スケジュール例

```
Week 1-2: Phase A (Nanda 再現)
Week 3-5: Phase B (β₁ / curl 測定)
Week 6:   Phase C (先行性検証)
Week 7:   Phase D (F 統合)
Week 8-9: (任意) Phase E 拡張
Week 10:  論文ドラフト化 or Part 4 への統合
```

### 8.3 マイルストーン

- **2 週間**: Phase A 完了（grokking 再現）
- **5 週間**: Phase B 完了（β₁ / curl 時系列取得）
- **7 週間**: 最初の主要結果（H_β1, H_curl の検証）
- **10 週間**: Part 4 への統合または論文化判断

---

## 9. Part 4 への統合方針

本実験の結果を [Part 4 Transformer 統合ノート](../gedig_transformer_architecture.md) に反映:

### 9.1 §5.X 新設: 「関連研究との位置取り」
- landscape メモの対比表
- 本実験の独自性（curl の初観測）

### 9.2 §7 実験計画: H_grokking-curl を最優先に
- 現行 §7 の優先度 1 (β₁ 指標切替) と並列または統合
- H_ising-bkt は優先度 3 以下に降格

### 9.3 §8 棄却条件: H_grokking-curl の結果条件を追加
- β₁ / curl が grokking で無反応 → Transformer 統合棚上げ
- これが Part 4 全体の load-bearing な検証

### 9.4 §3.1 案① (Hallucination Detector) への適用
- curl が grokking で機能するなら、hallucination 検知にも使える可能性
- AG ゲート強化の具体実装

---

## 10. 既存研究・メモの前提確認（crank 防止）

### 10.1 本実験が依存する既存知見

- [Part 1 §5](../gedig_core_theory_unified.md): β₁ の採用根拠（5 理由）
- [Part 2 §3](../gedig_cognitive_architecture.md): curl の階層的定義
- [Part 2 §3.3](../gedig_cognitive_architecture.md): 階層 3 = attention flow
- [gedig_cognitive_foundation.md §2-3](gedig_cognitive_foundation.md): curl の数学的基盤
- [gedig_prediction_curl.md §4](gedig_prediction_curl.md): 離散 curl の実装案
- [insight_beta1_dimension_free.md](insight_beta1_dimension_free.md): β₁ が Transformer で機能する理論的根拠

### 10.2 本実験が検証する仮説

- **新規**: curl × grokking の相関（どの既存研究にもない）
- **既存拡張**: β₁ × grokking（TAG-DS 2025 を F 統合版に拡張）
- **geDIG 固有**: 4 指標統合 F の優位性

### 10.3 作者の留保

- 実験前なので結果は未知
- 既存 β₁ proxy と同じ結果になれば独自性は curl のみに限定
- curl も無反応なら Part 4 Transformer 統合は棚上げ（§7 棄却条件）
- **結果に関わらず、実験自体が既存 curl TODO ([cognitive_foundation §8](gedig_cognitive_foundation.md)) の解消に貢献**

---

## 11. 関連リンク

### 参照元（本実験の前提）
- [insight_transformer_phase_transition_landscape.md](insight_transformer_phase_transition_landscape.md) — 先行研究ランドスケープ
- [insight_beta1_dimension_free.md](insight_beta1_dimension_free.md) — β₁ の次元フリー性
- [../gedig_core_theory_unified.md §5, §8.6, §9.5](../gedig_core_theory_unified.md) — β₁ 採用、negative result、解釈候補
- [../gedig_cognitive_architecture.md §3](../gedig_cognitive_architecture.md) — curl 階層的定義
- [../gedig_transformer_architecture.md](../gedig_transformer_architecture.md) — Part 4 全体

### 既存 curl 議論（本実験の実装基盤）
- [gedig_cognitive_foundation.md §2-3, §8](gedig_cognitive_foundation.md) — curl 数学的基盤、実装 TODO
- [gedig_prediction_curl.md §2, §4](gedig_prediction_curl.md) — curl = 予測、離散グラフでの実装
- [gedig_autonomous_discovery_machine.md §4](gedig_autonomous_discovery_machine.md) — curl + LLM 統合
- [gedig_as_discrete_fep_schrodinger_analogy_20260227.md §6](gedig_as_discrete_fep_schrodinger_analogy_20260227.md) — Berry phase との接続（speculative）

### 先行研究（2026-04-17 web 検索）
- [Nanda et al. 2023: Explaining grokking through circuit efficiency](https://arxiv.org/abs/2309.02390)
- [TAG-DS 2025: Betti-Fiedler partition as grokking proxy](https://www.emergentmind.com/topics/grokking-phase-transition)
- [Grokking as Dimensional Phase Transition (arXiv 2604.04655)](https://arxiv.org/abs/2604.04655)
- [Grokking as First Order Phase Transition (OpenReview)](https://openreview.net/forum?id=3ROGsTX3IR)
- [The Geometric Inductive Bias of Grokking (arXiv 2603.05228)](https://arxiv.org/abs/2603.05228)

---

## 12. 次のアクション

1. **Nanda のコード取得と再現**（Week 1）
2. **landscape メモの §4.5 に本メモへの cross-ref 追加**（今セッション内）
3. **INDEX.md Part 4 テーブルに本メモ参照追加**（今セッション内）
4. **Phase A 開始**（別セッション、GPU 利用時）
5. **Week 5 時点で中間結果を本メモに追記**（実験の進捗ログ）
