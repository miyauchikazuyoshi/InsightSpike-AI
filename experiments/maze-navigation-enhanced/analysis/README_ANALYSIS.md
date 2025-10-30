# Maze Analysis – Data Exports for Paper Figures (M‑ROC, Latency, Steps‑CDF, M‑Causal)

This folder contains small, focused exporters to derive the CSVs used by the paper’s figure scripts. Each exporter is self‑contained and writes to the repo‑level `data/` tree so the paper scripts can “just work”.

Wherever possible, exporters can either (a) parse existing run folders under `experiments/maze-navigation-enhanced/results/**`, or (b) accept an explicit input (CSV/JSON) via flags. When raw artifacts are missing, exporters explain what to log and emit a schematic CSV (headers + guidance rows) to unblock downstream plumbing.

## Targets and CSV Schemas

- M‑ROC (Bridge Detection ROC)
  - Output: `data/maze_eval/bridge_scores.csv`
  - Columns: `run_id, step, aggregator, score, y_true`
    - `aggregator ∈ {min, softmin:τ, sum}`
    - `score`: lower is better (more “bridge‑like”; negative spike preferred)
    - `y_true`: 1 if “true shortcut” (e.g., ΔSPL_rel ≤ −0.02), else 0

- Latency table (Maze)
  - Output: `data/latency/maze_latency.csv`
  - Columns: `run_id, H, k, latency_ms`
    - One row per evaluation; used to compute P50/P95/P99 per (H,k)

- Steps CDF (with effect size d)
  - Output: `data/maze_eval/steps_distribution.csv`
  - Columns: `run_id, method, steps, success`
    - `method ∈ {Random, DFS, GED_only, IG_only, geDIG}` (自由拡張可)
    - `success` is 0/1

- Event alignment around NA (M‑Causal)
  - Output: `data/maze_eval/event_alignment.csv`
  - Option A (long): `run_id, t_from_NA, event, value`
  - Option B (wide): `run_id, t_from_NA, BT, accept, evict` (0/1)

## Exporters

- `export_bridge_scores.py`: Aggregate per‑candidate scores and GT=ΔSPL判定で `bridge_scores.csv` を生成。
- `export_latency.py`: 実測の評価時間を集計して `maze_latency.csv` を生成。
- `export_steps_distribution.py`: 走行ログから `steps_distribution.csv` を生成。
- `export_event_alignment.py`: NAトリガ整列のイベント集計を `event_alignment.csv` に出力。

各スクリプトは `--results` で走行結果フォルダを指定可能（未指定時は `experiments/maze-navigation-enhanced/results/**` をスキャン）。

## 何をログすれば良いか（最小）

- M‑ROC: 各ステップの候補スコア（min/softmin/sum）と、ショートカットGTに必要な `SPL_before` / `SPL_after` を記録。GTは ΔSPL_rel=(after−before)/before ≤ −0.02 などで定義。
- Latency: 1回の geDIG 評価にかかった測定時間（ms）と (H,k)。
- Steps CDF: 最終ステップ数、成功/失敗、比較する `method` 名。
- M‑Causal: NA整列の `t_from_NA` とイベント（BT/accept/evict）の 0/1 値（長形式でも可）。

ログの実装例や簡易パーサは各Exporter内のDocstringを参照してください。

