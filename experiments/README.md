# InsightSpike-AI Experiments

geDIG フレームワークを検証する実験群。**リポジトリ全体の歩き方は [docs/MAP.md](../docs/MAP.md)**、
確証実験の事前登録 SOP は [docs/prereg/README.md](../docs/prereg/README.md) を参照。

## アクティブな実験ライン

| ライン | 場所 | 内容と状態(2026-07 時点) |
|---|---|---|
| **迷路(主力)** | [`maze/`](maze/) | stage-1 PoC(創発制御+~98% 圧縮、実証済み)。stage-2 sleep は事前登録 v1–v6 完了(2026-07-06): v1 敗北 → v2 成立(replay 伝播 −39% 歩数)→ v3 新シード再現(−51%)→ v4 敗北 → v5 採用見送り → v6 棄却(cycles=1 既定維持)。台帳は [docs/prereg/README.md](../docs/prereg/README.md)。実行定型・CLI は [maze/README.md](maze/README.md) |
| **RAG / ルーティング** | [`hotpotqa_v2/`](hotpotqa_v2/) | BRIGHT(nDCG 0.439 biology 単一ドメイン)、HotpotQA dual-process(非有意)、MuSiQue v10/v11。F-routing は Stage A 敗北記録済み、Stage B は DECISION 待ちで凍結 |
| **Transformer** | [`transformer/`](transformer/) | F-trajectory 観測(8 モデル)。F-regularization は予備的・符号未確定([f_sign 監査](../docs/audits/f_sign_audit.md)) |
| **maze β₁** | [`maze_b1/`](maze_b1/) | β₁ ベース評価の実験(v7 Phase 0 関連) |

## 保存物(通常は読まない)

- `refactor_maze/`, `refactor_hotpotqa_v2/`, `refactor_transformer/` — 2026-02 リファクタ時の旧実装
- `_archive_before_20260201_refactor/` — 35GB のローカルアーカイブ(**git 外**)

## 規約

- [EXPERIMENT_GUIDELINES.md](EXPERIMENT_GUIDELINES.md) — 再現性・データ管理・記録標準
- [OUTPUT_CONVENTION.md](OUTPUT_CONVENTION.md) — 出力の置き場所規約
- 確証実験は **必ず事前登録してから**([docs/prereg/](../docs/prereg/))。探索的ランは
  `results/**/_exploratory_*/` に隔離し、NOTES.md を残す
- 結果は `results/`(gitignore 済み・ローカルのみ)

---

*旧版のこのファイル(2025-07、v3 論文時代の 3 実験の記載)は現状と乖離していたため全面更新した(2026-07-03)。*
