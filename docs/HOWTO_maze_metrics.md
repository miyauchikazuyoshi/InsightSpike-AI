# 迷路実験 やり方メモ（等資源 + 運用ELBO）

このメモは Phase 1（EPCベース）の迷路PoCを、等資源（equal-resources）条件と指標一式で再現するための最小手順です。

## 等資源（equal-resources）の固定

代表設定（論文の表 tab:maze_equal_resources と整合）
- 迷路サイズ: `15x15, 25x25, 50x50`
- エピソード数 N: `100 / 60 / 40`（サイズ別）
- 最大ステップ: `size_factor=4.0`（ステップ打ち切り）
- 候補幅 Top-k: `k ∈ [8, 64]`（固定 or 掃引）
- multi-hop: `H ∈ [1, 3]`
- ASPLサンプル対数: `M ∈ [64, 128]`
- SP評価: 固定ペア（before側の連結集合）で `ΔSP_rel`（符号付き）
- 閾値: burn-in分位で自動較正（例 `p_AG∈[0.8,0.9], p_DG∈[0.05,0.2]`）

Makefileヘルパ
- `make reproduce-maze`（25x25, seeds=32 のプリセット）
- `make maze-suite PRESET=25x25 SIZE=25 SEEDS=32 FACTOR=4.0`
- 15/25/50 など個別は `maze-run` / `maze-calibrate` / `maze-stats` を組み合わせ

## 指標（ログと集計）

出力してほしい列（CSV/JSONいずれでも）
- 成功・効率系
  - `success_flag`（per-episode）, `success_rate(%)`
  - `steps_total`, `steps_p50`
  - `edges_final`, `edges_link`, `compression = 1 - |E_final|/|E_link|`
- 受容・品質系（PSZ）
  - `accept_rate(%)`, `fmr(%)`（誤統合率）
  - `aspl_before`, `aspl_after`, `sp_rel = (L_before - L_after)/max(L_before, ε)`
- 運用ELBO（ELBO-op）系
  - フレームごと: `g0, gmin, b=min(g0,gmin), accepted(bool), negF = -b`
  - 集計: `f_sum = Σ negF`, `f_per_accept = mean(negF|accepted)`, `f_per_step = f_sum/steps`
- メタ
  - `size, N, max_steps_factor, k, H, M, p_AG, p_DG`

定義メモ
- Fは「負側が改善」。`b(t) = min{ g0, gmin }` をイベント時の運用ゲージとし、`negF = -b(t)`。
- 運用ELBO（ELBO-op）は厳密なELBOではなく、`Σ accepted( -F )` の代理指標（操作的近似）。論文では「運用上のELBO」として明示。

## 最短の実行例

仮想環境が整っている前提:

```bash
# 25x25, seeds=32 の一括ルート（preset→calibrate→stats）
make reproduce-maze

# 個別: 15x15 小規模チューニング（速い）
make maze-tune-15 SEEDS=12 FAST=1

# 統一ランナー（比較含む）
make maze-unified-15 SEEDS=20 FAST=1
```

結果の場所（デフォルト）
- `experiments/maze-navigation-enhanced/` 配下の `results/` または `outputs/`（スクリプト実装に依存）

## 論文への反映

- 表: `tab:maze_baseline_comparison` に「ランダム/貪欲/DFS風/geDIG」の各列を流し込み
- 表: `tab:maze_equal_resources` は上の固定値に合わせる
- 小節: 「成功判定と指標」「アブレーションと検証」にログ列名を準拠（`f_sum, f_per_accept, sp_rel` など）

## よくある質問（メモ）
- Q: 運用ELBOは必須？
  - A: 必須ではないが、FEP/MDL語彙に橋をかけつつ、受容の“質”を時系列で見せる補助指標として有用。
- Q: multi-hop無しでもOK？
  - A: OK。`gmin` が無い場合は `b=g0` とし、条件を表に明記して比較する。

---
補助: 詳細なパラメータやスクリプトは `experiments/maze-navigation-enhanced/src` と `Makefile` の `maze-*` ターゲットを参照。
