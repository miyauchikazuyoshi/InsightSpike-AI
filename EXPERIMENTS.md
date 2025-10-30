# EXPERIMENTS — 再現手順のメモ

ここでは最小限の再現パスのみ記載します。詳細は各 experiment ディレクトリの README と論文を参照してください。

## 迷路（Maze Navigation）

最小実行例（小さいサイズで高速に）:

```bash
python examples/maze50_experiment.py --size 15 --max-steps 1000 --verbosity 1
```

ポイント:
- ΔEPC/ΔIG に基づくイベント駆動制御（AG/DG）で探索効率を改善
- `--verbosity 1` 以上で進捗ログ

## RAG（Dynamic Knowledge Graph for QA）

準備中（Phase 1 は設計/コードあり、再現用のスクリプトを短縮化中）:
- 当面は `examples/` と `experiments/` ディレクトリのスクリプトを参照
- 近日中に最短実行パスをここに統合

## 論文と図表

- 論文 v3（EPC基準）: docs/paper/geDIG_onegauge_improved_v3.tex
- 図（概念・結果）: docs/paper/figures/

