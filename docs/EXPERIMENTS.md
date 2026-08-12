# EXPERIMENTS — 論文準拠の実験入口

論文 v6.1(現行: `docs/paper/v6.1/`)に対応する実験は、以下の場所に集約しています。
一次インデックス(正)は [experiments/README.md](../experiments/README.md) です。

- 迷路(主力): `experiments/maze/`
  - stage-1 PoC と stage-2 sleep アブレーション(事前登録台帳は `docs/prereg/README.md`)
  - 補足ドキュメント: `docs/MAZE_NAV_SPEC.md`, `docs/HOWTO_maze_metrics.md`
- RAG / ルーティング: `experiments/hotpotqa_v2/`
  - BRIGHT、HotpotQA dual-process、MuSiQue v10/v11
- Transformer: `experiments/transformer/`
  - F-trajectory 観測、F-regularization(予備的)
- 迷路 β₁: `experiments/maze_b1/`
  - β₁ ベース評価(v7 Phase 0 関連)

保存物(2026-02 リファクタ時の旧実装、凍結オラクル・通常は読まない):
`experiments/refactor_maze/`, `experiments/refactor_hotpotqa_v2/`, `experiments/refactor_transformer/`

このファイル自体には手順を書かず、「どこを見ればよいか」のインデックスとしてだけ維持します。
