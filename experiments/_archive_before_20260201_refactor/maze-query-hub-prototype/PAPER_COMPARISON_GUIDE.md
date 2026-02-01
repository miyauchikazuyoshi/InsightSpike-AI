# 論文用 比較実験ラン・ガイド（Maze / Query‑Hub）

このガイドは、論文 v4（第5章）の比較実験を再現・更新するための最短手順です。
- 比較ラン（評価器 vs L3）
- 指標エクスポート（JSON/CSV）
- A/B差分の作成
- 超軽量モード＋DS再構成による長尺対応

## 前提（クラウド/ローカル共通）
- Python 3.x
- 依存は標準ライブラリのみ（エクスポータ/比較ツール）。実験本体は本リポジトリのコードを使用。
- 推奨ENV（任意）: `PYTHONPATH=src INSIGHTSPIKE_LOG_DIR=results/logs MPLCONFIGDIR=results/mpl`

## 1) 迷路スナップショット（任意・再現性向上）
実行開始時にレイアウトをJSON保存します。

```
--maze-snapshot-out docs/paper/data/maze_51x51.json
```

## 2) 比較ラン（評価器 vs L3）
ワンコマンドで A/B をまとめて実行できるランスクリプトを用意しています。

```
PYTHONPATH=src INSIGHTSPIKE_LOG_DIR=results/logs MPLCONFIGDIR=results/mpl \
python experiments/maze-query-hub-prototype/tools/run_paper_comparison.py \
  --maze-size 51 --max-steps 200 \
  --out-root experiments/maze-query-hub-prototype/results \
  --namespace paper_51x51_s200 \
  --maze-snapshot-out docs/paper/data/maze_51x51.json
```

- L3側 per-hop 記録を見たい場合（評価器フォールバック）:
  - 追加: `--union`
  - AGゲートでskipされるのを避ける: `--force-per-hop`（L3側のみ一時的にAG=-1.0）
- 超軽量（長尺向け）:
  - 追加: `--ultra-light`（steps.jsonから重い配列を生成/出力せず、post診断もOFF）

出力（既定の保存先）例:
- Eval: `_51x51_s200_eval_summary.json`, `_51x51_s200_eval_steps.json`
- L3  : `_51x51_s200_l3_summary.json`,   `_51x51_s200_l3_steps.json`

## 3) 論文用エクスポート（JSON/CSV）
上のランスクリプトは、完了後に自動で docs/paper/data/ に出力します。個別に実行する場合は:

```
python experiments/maze-query-hub-prototype/tools/export_paper_maze.py \
  --summary experiments/maze-query-hub-prototype/results/_51x51_s200_l3_summary.json \
  --steps   experiments/maze-query-hub-prototype/results/_51x51_s200_l3_steps.json \
  --out-json docs/paper/data/maze_51x51_l3_s200.json \
  --out-csv  docs/paper/data/maze_51x51_l3_s200.csv \
  --compression-base mem
```

- `--compression-base` は `mem|link` を選べます（冗長エッジ圧縮率の分母定義）。

## 4) ベースライン比較（A/B差分）

```
python experiments/maze-query-hub-prototype/baselines/compare_runs.py \
  --base-summary experiments/maze-query-hub-prototype/results/_51x51_s200_eval_summary.json \
  --base-steps   experiments/maze-query-hub-prototype/results/_51x51_s200_eval_steps.json \
  --var-summary  experiments/maze-query-hub-prototype/results/_51x51_s200_l3_summary.json \
  --var-steps    experiments/maze-query-hub-prototype/results/_51x51_s200_l3_steps.json \
  --out docs/paper/data/ab_eval_vs_l3_51x51_s200.json
```

## 5) HTML（軽量ステップ＋DS再構成）
超軽量で長尺を回した場合でも、DS（SQLite）からの再構成でHTMLを生成できます。

- 実行（超軽量＋DS）例:
```
... run_experiment_query.py ... \
  --steps-ultra-light --no-post-sp-diagnostics \
  --persist-graph-sqlite experiments/maze-query-hub-prototype/results/mq_51x51.sqlite \
  --persist-namespace ul51x51
```
- HTML生成:
```
python experiments/maze-query-hub-prototype/build_reports.py \
  --summary experiments/maze-query-hub-prototype/results/_51x51_s200_l3_ultra_summary.json \
  --steps   experiments/maze-query-hub-prototype/results/_51x51_s200_l3_ultra_steps.json \
  --sqlite  experiments/maze-query-hub-prototype/results/mq_51x51.sqlite \
  --namespace ul51x51 \
  --light-steps --present-mode strict \
  --out experiments/maze-query-hub-prototype/results/_51x51_s200_l3_ultra_interactive.html
```

## Tips
- 迷路ベースがHTMLに出ない場合は、`maze_data` を summary に含めるか、`--maze-snapshot-out` で作成したスナップショットをビルド前に取り込みます。
- 1000step など長尺はローカルでタイムアウトなし実行を推奨。`--checkpoint-interval 50` を併用すれば、途中経過のsteps.jsonから暫定HTMLを作成可能です。
 - ALL-PAIRS-EXACT のAPSP初期化コストを抑える最適化（任意）:
   - `--sp-exact-stable-nodes` を付けると、評価サブグラフ（SA）のノード集合をステップ間で単調増加に保ち、APSP行列の再利用ヒット率を上げられます（既定OFF; 数値整合性が重要な比較ではOFFのまま推奨）。

### Step Cap ランナー
25×25/51×51 を 500/1500 step で再評価する際は、手動入力ミスを避けるために以下のラッパーを利用できます。

````bash
python experiments/maze-query-hub-prototype/tools/run_adjusted_step_caps.py \
  --targets 25:600:60,51:2000:40 \
  --safety-caps 25:500,51:1500 \
  --out-root experiments/maze-query-hub-prototype/results/adjusted_steps \
  --namespace-prefix adj_v4 --ultra-light
````

- 既定で seed ごとに独立実行・出力するため、HTML や steps を seed 単位で検証可能です（旧挙動に戻す場合は `--aggregate-output`）。
- `--sp-cache-mode cached_incr`、SP pairset DS (`sp_pairsets.sqlite`)、**広域ノルム (`--link-radius 1 --cand-radius 1`)**、**無閾値 (`--theta-link 0 --theta-cand 0`)**、**SP 貪欲無制限 (`--sp-cand-topk 0`)** を既定オンにしており、遠方メモリや forced fallback も常に候補に載る構成です。
- `--build-html` を付けると run ごとに Strict DS HTML を生成し、g0/gmin/ΔSP/時間を即確認できます。
- `--safety-caps` の上限を超える指示は自動的に丸められ、長時間ランの暴走を防げます（超えたい場合のみ `--force-over-cap` を付与）。
- paper プリセット + L3-only multi-hop を内部で使用するため、論文本体の構成と整合します。
- `run_paper_l3_only.py`, `run_paper_comparison.py`, `run_paper_grid.py` などの論文ランナーはデフォルトで **軽量ログ（`--steps-ultra-light` + snapshot minimal）** を使います。詳細スナップショットが必要な場合のみ `--rich-logs` を付与してください。
- **短縮モード実行時の注意**: バッチ処理を短時間で終わらせたい場合は `--extra-args --sp-cand-topk 32 --candidate-cap 24 --top-m 24 --link-radius 0.7 --cand-radius 0.7` のようにパラメータを上書きすると 1 seed あたりの計算が軽くなります。ただし遠距離ノードや low-frequency の洞察が候補から漏れる恐れがあるため、最終報告値はフル設定（本節の既定値）で再走することを推奨します。

### 他手法ベースライン（オラクル最短路）

フルマップ既知の最短路（BFS/Dijkstra/A*）ベースラインをJSONで出力できます（論文比較用）。

```
PYTHONPATH=src python experiments/maze-query-hub-prototype/baselines/run_oracle_shortest.py \
  --maze-size 51 --seeds 40 --seed-start 0 \
  --method bfs \
  --out-json docs/paper/data/oracle_51x51_s250_bfs.json

PYTHONPATH=src python experiments/maze-query-hub-prototype/baselines/run_oracle_shortest.py \
  --maze-size 51 --seeds 40 --seed-start 0 \
  --method dijkstra \
  --out-json docs/paper/data/oracle_51x51_s250_dijkstra.json

PYTHONPATH=src python experiments/maze-query-hub-prototype/baselines/run_oracle_shortest.py \
  --maze-size 51 --seeds 40 --seed-start 0 \
  --method astar \
  --out-json docs/paper/data/oracle_51x51_s250_astar.json
```

出力には `baseline_success_rate` と `baseline_avg_steps` と `method` を含みます。DFS迷路（完全迷路）では常に経路が存在する想定です。

---

## 実行スクリプトと実行順（再現プロトコル）

以下の順で回すと、論文 v4（第5章）の全データが揃います。環境変数は一度設定しておくと便利です。

1) 共通ENV
```
export PYTHONPATH=src
export INSIGHTSPIKE_LOG_DIR=results/logs
export MPLCONFIGDIR=results/mpl
export INSIGHTSPIKE_PRESET=paper
```

2) L3-only（最終結果）を推奨
```
python experiments/maze-query-hub-prototype/tools/run_paper_l3_only.py \
  --maze-size 15 --max-steps 250 --seeds 100 \
  --out-root experiments/maze-query-hub-prototype/results/l3_only \
  --namespace l3only_15x15_s250 --ultra-light \
  --maze-snapshot-out docs/paper/data/maze_15x15.json

python experiments/maze-query-hub-prototype/tools/run_paper_l3_only.py \
  --maze-size 25 --max-steps 250 --seeds 60 \
  --out-root experiments/maze-query-hub-prototype/results/l3_only \
  --namespace l3only_25x25_s250 --ultra-light \
  --maze-snapshot-out docs/paper/data/maze_25x25.json

python experiments/maze-query-hub-prototype/tools/run_paper_l3_only.py \
  --maze-size 51 --max-steps 250 --seeds 40 \
  --out-root experiments/maze-query-hub-prototype/results/l3_only \
  --namespace l3only_51x51_s250 --ultra-light \
  --maze-snapshot-out docs/paper/data/maze_51x51.json
```

（A/B 比較は内部整合確認に留め、最終報告は L3-only を用いる。）

3) （任意）15×15 / 25×25 / 51×51（s=200）を比較ラン → 自動エクスポート
```
python experiments/maze-query-hub-prototype/tools/run_paper_comparison.py \
  --maze-size 15 --max-steps 250 --seeds 100 \
  --out-root experiments/maze-query-hub-prototype/results/paper_grid \
  --namespace paper_v4_15x15_s250 --ultra-light \
  --maze-snapshot-out docs/paper/data/maze_15x15.json

python experiments/maze-query-hub-prototype/tools/run_paper_comparison.py \
  --maze-size 25 --max-steps 250 --seeds 60 \
  --out-root experiments/maze-query-hub-prototype/results/paper_grid \
  --namespace paper_v4_25x25_s250 --ultra-light \
  --maze-snapshot-out docs/paper/data/maze_25x25.json

python experiments/maze-query-hub-prototype/tools/run_paper_comparison.py \
  --maze-size 51 --max-steps 200 --seeds 200 \
  --out-root experiments/maze-query-hub-prototype/results/paper_grid \
  --namespace paper_v4_51x51_s200 --ultra-light \
  --maze-snapshot-out docs/paper/data/maze_51x51.json
```

4) 51×51 / 250 / seeds=40 の本番（長時間）
```
bash experiments/maze-query-hub-prototype/tools/run_51x51_s250_seeds40.sh
```

5) オラクル最短路（BFS/Dijkstra/A*）の併記
```
python experiments/maze-query-hub-prototype/baselines/run_oracle_shortest.py \
  --maze-size 15 --seeds 100 --method bfs \
  --out-json docs/paper/data/oracle_15x15_s250_bfs.json

python experiments/maze-query-hub-prototype/baselines/run_oracle_shortest.py \
  --maze-size 25 --seeds 60 --method bfs \
  --out-json docs/paper/data/oracle_25x25_s250_bfs.json

python experiments/maze-query-hub-prototype/baselines/run_oracle_shortest.py \
  --maze-size 51 --seeds 40 --method bfs \
  --out-json docs/paper/data/oracle_51x51_s250_bfs.json
```

6) HTML 生成（DS再構成・strict）
```
python experiments/maze-query-hub-prototype/build_reports.py \
  --summary experiments/maze-query-hub-prototype/results/paper_grid/_25x25_s250_l3_summary.json \
  --steps   experiments/maze-query-hub-prototype/results/paper_grid/_25x25_s250_l3_steps.json \
  --sqlite  experiments/maze-query-hub-prototype/results/paper_grid/mq_25x25.sqlite \
  --namespace paper_v4_25x25_s250_l3 --light-steps --present-mode strict \
  --out experiments/maze-query-hub-prototype/results/paper_grid/_25x25_s250_l3_interactive.html
```

---

## 5.10 質的分析（提案）

25×25 の HTML/steps.json から、次の3パターンを抽出して図を作成します。

- 通常通路（corridor; 分岐なし）: `num_paths=2` かつ `is_dead_end=False` かつ `is_junction=False`
- T字路（T‑junction）: `is_junction=True`
- 行き止まり（dead-end）: `is_dead_end=True`

作図方針（最小実装）
- 迷路レイアウト（layout）を背景に、
  - 現在位置（青点）
  - 選択行動（太矢印）
  - 可能行動（薄矢印）
  を重ねた静止画（PNG）を作る。
- steps.json の `position`, `action`, `possible_moves` を用い、`summary.maze_data`（なければ `--maze-snapshot-out` のJSON）から `layout/start_pos/goal_pos/size` を読み込む。

スクリプト（生成）
```
PYTHONPATH=src MPLCONFIGDIR=results/mpl \
python experiments/maze-query-hub-prototype/tools/export_qualitative_panels.py \
  --summary experiments/maze-query-hub-prototype/results/paper_grid/_25x25_s250_l3_summary.json \
  --steps   experiments/maze-query-hub-prototype/results/paper_grid/_25x25_s250_l3_steps.json \
  --out-dir docs/paper/figures/maze_25x25_panels \
  --maze-json docs/paper/data/maze_25x25.json
```

出力（例）
- `corridor_panel.png`, `t_junction_panel.png`, `dead_end_panel.png`
- 図番号候補: Fig.~5.10a（通常通路）, 5.10b（T字路）, 5.10c（行き止まり）

分析観点（本文チップ）
- 通常通路: 観測候補の順位と候補台IGの緩やかな増加（$\sim\log K$）。hop0配線は Top‑L で安定。
- T字路: S\_link が空の際の強制1本配線（forced Top‑L）→ hop0 の $\Delta\mathrm{SP}$ 上昇と AG/DG の発火関係。
- 行き止まり: `--skip-mh-on-deadend=OFF` の挙動。未探索分岐に戻るための評価（S\_link空→forcedベースを採用）と DG 多発域の動作。

- まとめて再テスト（gridラン）
  - 15×15=100試行、25×25=32試行、51×51=10試行など、サイズごとの試行数で一括実行:
```
PYTHONPATH=src INSIGHTSPIKE_LOG_DIR=results/logs MPLCONFIGDIR=results/mpl \
python experiments/maze-query-hub-prototype/tools/run_paper_grid.py \
  --sizes 15 25 51 \
  --steps-per-size 15:250 25:250 51:250 \
  --seeds-per-size 15:100 25:32 51:10 \
  --out-root experiments/maze-query-hub-prototype/results/paper_grid \
  --namespace-prefix paper_v4 \
  --ultra-light --maze-snapshot-out docs/paper/data/maze_51x51.json
```
  - per-hop 記録が必要なら `--union` を追加（必要に応じて `--force-per-hop`）。
  - 上記は論文で用いる代表的な試行数の例です。実際の本番値に合わせて `--seeds-per-size` を調整してください。

---

## 6) 本稿の推奨条件（L3‑only + AG時のみ per‑hop）

論文の最終結果は L3‑only を採用し、AG発火時のみ per‑hop を記録して可視化します。高速化のため SP エンジンは `cached_incr` を使い、heavy配列は `--steps-ultra-light` で生成しません。

- 51×51 1000step（1 seed例）
```
PYTHONPATH=src INSIGHTSPIKE_PRESET=paper INSIGHTSPIKE_LOG_DIR=results/logs MPLCONFIGDIR=results/mpl \
python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --maze-size 51 --max-steps 1000 --seeds 1 --seed-start 0 \
  --linkset-mode --norm-base link --theta-ag 0.4 --top-link 1 \
  --use-main-l3 --sp-cache-mode cached_incr --eval-per-hop-on-ag --max-hops 10 \
  --steps-ultra-light --no-post-sp-diagnostics \
  --output experiments/maze-query-hub-prototype/results/l3_fast/_51x51_s1000_l3_agperhop_summary.json \
  --step-log experiments/maze-query-hub-prototype/results/l3_fast/_51x51_s1000_l3_agperhop_steps.json \
  --checkpoint-interval 100

python experiments/maze-query-hub-prototype/build_reports.py \
  --summary experiments/maze-query-hub-prototype/results/l3_fast/_51x51_s1000_l3_agperhop_summary.json \
  --steps   experiments/maze-query-hub-prototype/results/l3_fast/_51x51_s1000_l3_agperhop_steps.json \
  --out     experiments/maze-query-hub-prototype/results/l3_fast/_51x51_s1000_l3_agperhop_interactive.html \
  --light-steps
```

### Seed選定（AG多発のシードを抽出）
まず短尺プローブで AG 活動が多い seed を抽出します。

```
# プローブ（例: 51×51 60step, 20 seeds）
PYTHONPATH=src INSIGHTSPIKE_PRESET=paper INSIGHTSPIKE_LOG_DIR=results/logs MPLCONFIGDIR=results/mpl \
python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --maze-size 51 --max-steps 60 --seeds 20 --seed-start 0 \
  --linkset-mode --norm-base link --theta-ag 0.4 --top-link 1 \
  --use-main-l3 --sp-cache-mode cached_incr --eval-per-hop-on-ag --max-hops 10 \
  --steps-ultra-light --no-post-sp-diagnostics \
  --output experiments/maze-query-hub-prototype/results/l3_fast/_51x51_s60_l3_agperhop_probe_summary.json \
  --step-log experiments/maze-query-hub-prototype/results/l3_fast/_51x51_s60_l3_agperhop_probe_steps.json

# AGの多いseedを上位抽出
PYTHONPATH=src python experiments/maze-query-hub-prototype/tools/select_seeds_by_ag.py \
  --steps experiments/maze-query-hub-prototype/results/l3_fast/_51x51_s60_l3_agperhop_probe_steps.json \
  --top 8 \
  --out experiments/maze-query-hub-prototype/results/l3_fast/_51x51_s60_l3_agperhop_probe_topseeds.json
```

`_probe_topseeds.json` の上位から、1–3本を 1000step で L3‑only（上記コマンド）にて本番実行し、HTML を作成してください。

---

## 7) 論文の赤字（数値）更新の流れ

1. 比較ラン完了後、docs/paper/data/ 以下に出力された `maze_*` の JSON/CSV を確認。
2. 主要表・本文の赤字 `\experimNote{...}` 箇所を、上記の JSON 値で置換。
   - 例: `success_rate`, `avg_steps`, `edge_compression`, `ag_rate`, `dg_rate`, `avg_time_ms_eval`, `p95_time_ms_eval`。
3. 51×51（s=250, seeds=40）の本番ラン完了後、該当箇所（§5の表、注記）を差し替え。

補足: 本ガイドのコマンドは paper v4 の “L3‑only + cached_incr + AG時のみ per‑hop” を前提に記しています。A/B を伴う集計は `tools/run_paper_grid.py` を、L3‑only での長尺HTMLは上記 6) をそれぞれ利用してください。
## 付録: SP評価モード（評価サブグラフ内）

- fixed-before（既定）
  - 前サブグラフでサンプルした固定ペア集合に対し、afterの最短路を厳密評価して相対改善（ΔSP）を算出。
- ALL-PAIRS（診断）
  - 前後で到達可能なペアをすべて対象に平均最短路を比較（`--sp-allpairs`）。新規到達の短絡も可視化できる。
- ALL-PAIRS-EXACT（高速・厳密）
  - 評価サブグラフに限定し、ALL-PAIRS の数値を保ったまま、各hopで採用する1本のエッジに対して2回のBFSと O(n^2) の更新則（min(d, du+1+vt, dv+1+ut)）で平均最短路を厳密に増分更新。
  - 有効化: `--sp-allpairs-exact`
  - 例（25x25/100step）: step18/72 の hop2 で SP≈0.4167 を再現しつつ、全ペア再計算を避け高速化。

### 最新スナップショット（25x25 / 500step, ALL‑PAIRS‑EXACT）

実行:

```
python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --preset paper --maze-size 25 --max-steps 500 --layer1-prefilter \
  --eval-all-hops --sp-allpairs-exact \
  --output experiments/maze-query-hub-prototype/results/paper_25x25_s500_allpairs_exact_summary.json \
  --step-log experiments/maze-query-hub-prototype/results/paper_25x25_s500_allpairs_exact_steps.json
```

要約（実測・サマリ抜粋）:

- success_rate: 1.00
- avg_steps: 276.0
- avg_delta_sp_min: 0.12157
- best_hop_mean: 0.413（内訳: hop0=219, hop2=57）
- avg_time_ms_eval: 1.56 ms / step（p95: 3.73 ms）
- 実行時間（壁時計）: 約 23分14秒（real=23m14s, user=22m52s, sys=0m10s）

出力ファイル:

- summary: `experiments/maze-query-hub-prototype/results/paper_25x25_s500_allpairs_exact_summary.json`
- steps:   `experiments/maze-query-hub-prototype/results/paper_25x25_s500_allpairs_exact_steps.json`
