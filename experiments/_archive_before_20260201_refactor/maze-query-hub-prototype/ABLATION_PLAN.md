# Ablation Plan: geDIG (Query-Hub Prototype)

目的
- 混乱点（GED/IG/SPの定義、multi-hopの評価レンジ、正規化分母、表示シークエンス）を切り分ける。
- 1つずつ切替可能なアブレーションを用意し、短文レポート＋成果物（JSON/HTML）にリンクする。
- 初期は λ/hops/decay を固定: `λ=0.5 / max_hops=15 / decay=0.7`。

評価共通
- 迷路: 25×25、dfs、steps=60、seeds=3（回転を早めるため短め）
- 表示: HTML の Temporal Metrics は hop0 を除外した min（multi-hop-only）を描画
- Edges 表示: Committed only（保存済みエッジ）を基本に観察

観測項目
- g0, g_min(mh), ΔGED(min, mh), ΔIG(min, mh), ΔSP(min, mh)
- SP Debug 表（pair_count, Lb, top δSP）
- Per-hop Metrics（hop>=1 のみ）

切替軸（最初の2設定）

| ID | GED policy | IG policy | SP policy | Anchor/Scope | Eval flow | 期待 | JSON | HTML |
|----|------------|-----------|-----------|--------------|----------|------|------|------|
| A1 | Cmax=1+\|S_link\|、hop>0はGED(0)固定（ged_hop0_const=true） | linkset, all hops | fixed-before-pairs, union, hop_expand=3 | dual anchors（core+TopL）, union | H=15連続（eval-all-hops） | multi-hop差はIG/SPのみで出る。g_min(mh)が動くか確認 | results/ablations/A1/summary.json | results/ablations/A1/interactive.html |
| A2 | Cmax=1+\|S_link\|固定、hop>0はGED再計算（ged_hop0_const=false） | linkset, all hops | fixed-before-pairs, union, hop_expand=3 | dual anchors, union | H=15連続 | ΔGED(min,mh)に差が出るかの比較 | results/ablations/A2/summary.json | results/ablations/A2/interactive.html |

シークエンス軸（優先実施）

| ID | Hop0配線 | Fallback配線 | g0/g(h) 評価対象 | DG実コミット | 備考 | JSON | HTML |
|----|-----------|---------------|------------------|--------------|------|------|------|
| S0 | あり（Top‑L） | あり（S_link空→forced Top‑L） | prev vs after(h) | なし（commit_budget=0） | 現行ベースラインの近傍 | results/ablations/S0/summary.json | results/ablations/S0/interactive.html |
| S1 | なし（--top-link=0） | なし（--top-link=0で無効） | prev vs after(h)（仮想のみ） | なし | 「配線前に評価」派生。hop0のΔGEDは0に近いはず | results/ablations/S1_radius_before/summary.json | results/ablations/S1_radius_before/interactive.html |
| S2 | あり（Top‑L） | なし（S_link空時はEcandへ） | prev vs after(h) | なし | Fallbackは候補にのみ投入（配線せず） | results/ablations/S2/summary.json | results/ablations/S2/interactive.html |

実行例（共通の推奨フラグ）
// IG 符号は after_before 固定（切替ノブなし）
- SP境界: `MAZE_GEDIG_SP_BOUNDARY=trim`
- λ=0.5, H=15, union, eval-all-hops, linksetモード、k抑制（sp-cand-topk）を統一

例コマンド

S0（ベースライン）
```
python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --maze-size 25 --max-steps 60 --seeds 3 \
  --lambda-weight 0.5 --max-hops 15 --sp-scope union --sp-hop-expand 3 \
  --eval-all-hops --linkset-mode --theta-ag 0.2 --theta-dg 10 \
  --top-link 1 --commit-budget 0 --norm-base link \
  --sp-cache --sp-cache-mode core --sp-cand-topk 24 --sp-pair-samples 512 \
  --output experiments/maze-query-hub-prototype/results/ablations/S0/summary.json \
  --step-log experiments/maze-query-hub-prototype/results/ablations/S0/steps.json
python experiments/maze-query-hub-prototype/build_reports.py \
  --summary experiments/maze-query-hub-prototype/results/ablations/S0/summary.json \
  --steps   experiments/maze-query-hub-prototype/results/ablations/S0/steps.json \
  --out     experiments/maze-query-hub-prototype/results/ablations/S0/interactive.html
```

S1（配線前評価）
```
python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --maze-size 25 --max-steps 60 --seeds 3 \
  --lambda-weight 0.5 --max-hops 15 --sp-scope union --sp-hop-expand 3 \
  --eval-all-hops --linkset-mode --theta-ag 0.2 --theta-dg 10 \
  --top-link 0 --commit-budget 0 --norm-base link \
  --sp-cache --sp-cache-mode core --sp-cand-topk 24 --sp-pair-samples 512 \
  --output experiments/maze-query-hub-prototype/results/ablations/S1/summary.json \
  --step-log experiments/maze-query-hub-prototype/results/ablations/S1/steps.json
python experiments/maze-query-hub-prototype/build_reports.py \
  --summary experiments/maze-query-hub-prototype/results/ablations/S1/summary.json \
  --steps   experiments/maze-query-hub-prototype/results/ablations/S1/steps.json \
  --out     experiments/maze-query-hub-prototype/results/ablations/S1/interactive.html
```

S2（Fallbackは候補のみ）
- 現状実装では `--top-link 0` で S_link 空時の forced fallback 配線も抑止されます（Ecandへのみ回す）。
```
python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --maze-size 25 --max-steps 60 --seeds 3 \
  --lambda-weight 0.5 --max-hops 15 --sp-scope union --sp-hop-expand 3 \
  --eval-all-hops --linkset-mode --theta-ag 0.2 --theta-dg 10 \
  --top-link 1 --commit-budget 0 --norm-base link \
  --sp-cache --sp-cache-mode core --sp-cand-topk 24 --sp-pair-samples 512 \
  --output experiments/maze-query-hub-prototype/results/ablations/S2/summary.json \
  --step-log experiments/maze-query-hub-prototype/results/ablations/S2/steps.json
```

期待される観測
- S1: hop0 の ΔGED は 0 近傍になり、g0 は主に IG で決まり、multi-hop でのみ ΔSP が出るケースが増える。
- S0/S2 比較: seed0 step32 のような「hop0=0 だが hop>0=1.0」挙動がどちらで再現するかで、候補配線タイミング or fallback 配線の影響を切り分けられる。

新シークエンス（S0a / S2a）: 次Q/時系列エッジをgraphへ追加しない

- 変更点（抜本）
  - env.step() 直後に生成していた次Qノードの graph への追加を停止（表示のみ）。
  - 実行履歴エッジ（Q_prev→dir→Q_next, Q_prev↔Q_next）は graph に追加せず、StepRecord.timeline_edges のみへ記録（HTMLで描画）。
  - これにより、評価（SP/GED/IG）のサブグラフには次Q/時系列エッジが一切混入しない。
  - 切戻し用フラグ: `--timeline-to-graph` と `--add-next-q` を追加（既定はOFF）。

| ID  | 変更                     | 既定 | CLIで元に戻す |
|-----|--------------------------|------|---------------|
| S0a | timeline_to_graph=false, add_next_q=false | OFF  | `--timeline-to-graph --add-next-q` |
| S2a | 同上 + `--commit-from link`               | OFF  | `--timeline-to-graph --add-next-q` |

例コマンド

S0a
```
python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --maze-size 25 --max-steps 60 --seeds 1 \
  --lambda-weight 1.0 --max-hops 15 --sp-scope union --sp-hop-expand 3 \
  --eval-all-hops --linkset-mode --theta-ag 0.2 --theta-dg 10 \
  --top-link 1 --commit-budget 0 --norm-base link \
  --gh-mode radius --snapshot-mode after_select \
  --sp-cache --sp-cache-mode core --sp-cand-topk 24 --sp-pair-samples 512 \
  --sp-beta 1.0 --action-policy softmax --action-temp 0.1 \
  --output experiments/maze-query-hub-prototype/results/ablations/S0a/summary.json \
  --step-log experiments/maze-query-hub-prototype/results/ablations/S0a/steps.json
python experiments/maze-query-hub-prototype/build_reports.py \
  --summary experiments/maze-query-hub-prototype/results/ablations/S0a/summary.json \
  --steps   experiments/maze-query-hub-prototype/results/ablations/S0a/steps.json \
  --out     experiments/maze-query-hub-prototype/results/ablations/S0a/interactive.html
```

S0b / S2b（差分メタ付ログ + 任意のSQLite永続化）

- 追加点: steps.json に `committed_only_edges_meta` / `committed_only_nodes_meta` を含める。必要に応じて `--persist-graph-sqlite` でSQLiteに差分を永続化。

S0b
```
python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --maze-size 25 --max-steps 60 --seeds 1 \
  --lambda-weight 1.0 --max-hops 15 --sp-scope union --sp-hop-expand 3 \
  --eval-all-hops --linkset-mode --theta-ag 0.2 --theta-dg 10 \
  --top-link 1 --commit-budget 0 --norm-base link \
  --gh-mode radius --snapshot-mode after_select \
  --sp-cache --sp-cache-mode core --sp-cand-topk 24 --sp-pair-samples 512 \
  --sp-beta 1.0 --action-policy softmax --action-temp 0.1 \
  --persist-graph-sqlite experiments/maze-query-hub-prototype/results/ablations/S0b/graph.sqlite \
  --persist-namespace maze_qh \
  --output experiments/maze-query-hub-prototype/results/ablations/S0b/summary.json \
  --step-log experiments/maze-query-hub-prototype/results/ablations/S0b/steps.json
python experiments/maze-query-hub-prototype/build_reports.py \
  --summary experiments/maze-query-hub-prototype/results/ablations/S0b/summary.json \
  --steps   experiments/maze-query-hub-prototype/results/ablations/S0b/steps.json \
  --out     experiments/maze-query-hub-prototype/results/ablations/S0b/interactive.html
```

S0bl / S2bl（forced候補リンクも永続化）

- 追加点: `--persist-forced-candidates` を有効化し、forced候補の Q↔dir も `graph_edges.edge_type='forced_candidate'` でSQLiteへ保存（steps.json には `forced_edges(_meta)` で出力）。
- 評価用graphには入れない（可視化の selected_links で表示）。

S0bl
```
python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --maze-size 25 --max-steps 200 --seeds 1 \
  --lambda-weight 1.0 --max-hops 15 --sp-scope union --sp-hop-expand 3 \
  --eval-all-hops --linkset-mode --theta-ag 0.2 --theta-dg 10 \
  --top-link 1 --commit-budget 0 --norm-base link \
  --gh-mode radius --snapshot-mode after_select \
  --sp-cache --sp-cache-mode core --sp-cand-topk 24 --sp-pair-samples 512 \
  --sp-beta 1.0 --action-policy softmax --action-temp 0.1 \
  --persist-graph-sqlite experiments/maze-query-hub-prototype/results/ablations/S0bl/graph.sqlite \
  --persist-namespace maze_qh --persist-forced-candidates \
  --output experiments/maze-query-hub-prototype/results/ablations/S0bl/summary.json \
  --step-log experiments/maze-query-hub-prototype/results/ablations/S0bl/steps.json
python experiments/maze-query-hub-prototype/build_reports.py \
  --summary experiments/maze-query-hub-prototype/results/ablations/S0bl/summary.json \
  --steps   experiments/maze-query-hub-prototype/results/ablations/S0bl/steps.json \
  --out     experiments/maze-query-hub-prototype/results/ablations/S0bl/interactive.html
```

S2bl（`--commit-from link`）
```

S0blf / S2blf（forced候補を linkset の台に採用 + 永続化）

- 追加点: `--link-forced-as-base`（実装は内部フラグ; コマンド例に注記）を有効化し、S_link空時は forced を台にした評価（before: forced, after: forced+query）で ΔH/ΔGED を安定化。永続化は S0bl と同じく有効。

S0blf
```
python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --maze-size 25 --max-steps 200 --seeds 1 \
  --lambda-weight 1.0 --max-hops 15 --sp-scope union --sp-hop-expand 3 \
  --eval-all-hops --linkset-mode --theta-ag 0.2 --theta-dg 10 \
  --top-link 1 --commit-budget 0 --norm-base link \
  --gh-mode radius --snapshot-mode after_select \
  --sp-cache --sp-cache-mode core --sp-cand-topk 24 --sp-pair-samples 512 \
  --sp-beta 1.0 --action-policy softmax --action-temp 0.1 \
  --persist-graph-sqlite experiments/maze-query-hub-prototype/results/ablations/S0blf/graph.sqlite \
  --persist-namespace maze_qh --persist-forced-candidates \
  --output experiments/maze-query-hub-prototype/results/ablations/S0blf/summary.json \
  --step-log experiments/maze-query-hub-prototype/results/ablations/S0blf/steps.json
```

S2blf（`--commit-from link`）
```
python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --maze-size 25 --max-steps 200 --seeds 1 \
  --lambda-weight 1.0 --max-hops 15 --sp-scope union --sp-hop-expand 3 \
  --eval-all-hops --linkset-mode --theta-ag 0.2 --theta-dg 10 \
  --top-link 1 --commit-budget 0 --norm-base link \
  --commit-from link \
  --gh-mode radius --snapshot-mode after_select \
  --sp-cache --sp-cache-mode core --sp-cand-topk 24 --sp-pair-samples 512 \
  --sp-beta 1.0 --action-policy softmax --action-temp 0.1 \
  --persist-graph-sqlite experiments/maze-query-hub-prototype/results/ablations/S2blf/graph.sqlite \
  --persist-namespace maze_qh --persist-forced-candidates \
  --output experiments/maze-query-hub-prototype/results/ablations/S2blf/summary.json \
  --step-log experiments/maze-query-hub-prototype/results/ablations/S2blf/steps.json
```
python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --maze-size 25 --max-steps 200 --seeds 1 \
  --lambda-weight 1.0 --max-hops 15 --sp-scope union --sp-hop-expand 3 \
  --eval-all-hops --linkset-mode --theta-ag 0.2 --theta-dg 10 \
  --top-link 1 --commit-budget 0 --norm-base link \
  --commit-from link \
  --gh-mode radius --snapshot-mode after_select \
  --sp-cache --sp-cache-mode core --sp-cand-topk 24 --sp-pair-samples 512 \
  --sp-beta 1.0 --action-policy softmax --action-temp 0.1 \
  --persist-graph-sqlite experiments/maze-query-hub-prototype/results/ablations/S2bl/graph.sqlite \
  --persist-namespace maze_qh --persist-forced-candidates \
  --output experiments/maze-query-hub-prototype/results/ablations/S2bl/summary.json \
  --step-log experiments/maze-query-hub-prototype/results/ablations/S2bl/steps.json
python experiments/maze-query-hub-prototype/build_reports.py \
  --summary experiments/maze-query-hub-prototype/results/ablations/S2bl/summary.json \
  --steps   experiments/maze-query-hub-prototype/results/ablations/S2bl/steps.json \
  --out     experiments/maze-query-hub-prototype/results/ablations/S2bl/interactive.html
```

## 追加：Link Auto‑wire（T字路安定化）

目的
- T字路で S_link に2本入っているにも関わらず、0‑hopのTop‑L=1で片側のみ配線→SPが立ちづらい問題を解消。
- 0‑hopベース配線を「S_link全本」に切替（Top‑Lを無視）して before に両腕を含め、SP即時発火を観測しやすくする。

設定
- `--link-autowire-all` をON（既定OFF）。その他は従来通り。
- 正規化は `--norm-base link` のまま（Cmax=1+|S_link|、ステップ内は固定）。
- 候補の半径やSPの評価半径は変更しない（r_link/r_cand, sp_hop_expand は従来値）。

期待と観測
- 0‑hopで両腕がbeforeに入り、hop>=1でΔSP>0が立ちやすくなる → gminがその場でg0より下がるケースが増える。
- Cmaxは|S_link|に依存するため、2本→正規化ΔGEDが小さめに出る（分子は同じでも分母で割るため）。

実行例（短縮・診断）
```
python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --maze-size 25 --max-steps 40 --seeds 1 \
  --max-hops 15 --norm-base link --linkset-mode \
  --theta-ag 0.2 --theta-dg 0.15 --top-link 1 --link-autowire-all \
  --snapshot-level standard \
  --output experiments/maze-query-hub-prototype/results/_diag_std_autowire/summary.json \
  --step-log experiments/maze-query-hub-prototype/results/_diag_std_autowire/steps.json \
  --persist-graph-sqlite experiments/maze-query-hub-prototype/results/_diag_std_autowire/graph.sqlite
python experiments/maze-query-hub-prototype/build_reports.py \
  --summary experiments/maze-query-hub-prototype/results/_diag_std_autowire/summary.json \
  --steps experiments/maze-query-hub-prototype/results/_diag_std_autowire/steps.json \
  --sqlite experiments/maze-query-hub-prototype/results/_diag_std_autowire/graph.sqlite \
  --present-mode none --strict \
  --out experiments/maze-query-hub-prototype/results/_diag_std_autowire/interactive.html
```

補足（Q2の棚上げメモ）
- ΔGEDの分子は「prev_graph→after_graph_h」の新規ノード/エッジの編集コスト合計（行動やタイムラインは含まない）。
- 正規化の分母は `--norm-base` に従う（link: Cmax=1+|S_link|、cand: Top‑kcandベース）。
- ステップ内では `ged_hop0_const=true` で ΔGED(h)を一定扱い。g0とgminの差はIG/SP由来。
- ステップ間のΔGEDの上下は分母変動（|S_link|変化）による“見かけの降下”があり得る。

# S2bl-2phase（Greedy + Two-phase commit）

目的
- geDIG評価と実コミットのタイミングを完全分離。Hop0（Top‑L）は評価専用グラフで配線し、geDIG計算後にHop0とDG（best_hopまで/予算上限）をまとめて「実コミット」する。
- 表示はDS（永続）最優先。steps.jsonには dg_committed_edges を追加し実コミット内容を可視化。

設定
- gh-mode=greedy、eval-all-hops、linkset、union、hop_expand=3、λ=1、γ=1、τ=0.1
- DG採用はポリシーで制御: --dg-commit-policy threshold|always|never（既定: threshold）
- コミット対象: Q↔dir のみ（Q↔Qは可視化のみ）

ログ/スナップショット
- graph_edges_eval: 評価専用（Hop0のみ配線）の eval_after
- graph_edges: 実コミット反映後の最終
- committed_only_edges: 当ステップで実際に追加されたエッジ（preselectとの差分）
- dg_committed_edges: DGで採用されたエッジ一覧（新規）

実行例（200 steps, 1 seed）
```
MAZE_GEDIG_SP_BOUNDARY=trim \
python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --maze-size 25 --max-steps 200 --seeds 1 \
  --lambda-weight 1.0 --max-hops 15 --decay-factor 0.7 --adaptive-hops \
  --sp-beta 1.0 --linkset-mode --sp-scope union --sp-hop-expand 3 \
  --theta-ag 0.2 --theta-dg 0.15 --top-link 1 --anchor-recent-q 32 \
  --norm-base link --action-policy softmax --action-temp 0.1 \
  --gh-mode greedy --eval-all-hops --dg-commit-policy threshold \
  --persist-graph-sqlite experiments/maze-query-hub-prototype/results/ablations/S2bl_2phase/graph.sqlite \
  --persist-namespace S2bl_2phase \
  --output experiments/maze-query-hub-prototype/results/ablations/S2bl_2phase/summary.json \
  --step-log experiments/maze-query-hub-prototype/results/ablations/S2bl_2phase/steps.json

python experiments/maze-query-hub-prototype/build_reports.py \
  --summary experiments/maze-query-hub-prototype/results/ablations/S2bl_2phase/summary.json \
  --steps   experiments/maze-query-hub-prototype/results/ablations/S2bl_2phase/steps.json \
  --out     experiments/maze-query-hub-prototype/results/ablations/S2bl_2phase/interactive.html
```

新規: S2bl-greedy-fix（貪欲モード修正＋DS描画既定）

- 目的: gh-mode=greedy のインデント不整合で発生していた UnboundLocalError（h, eff_h_diag）と候補スコア評価の外側実行問題を修正し、hopごとのδSP探索とg(h)評価を正しくループ内で行う。
- 表示: HTMLはDSグラフ（persisted）を既定にし、evalオーバーレイ無効化。タイムラインのQ→dir→QのエッジはDSへも累積（edge_type=timeline）し、抽象グラフ上も“実際に保存されたもののみ”を描画。

実行例（1seed/120steps, H=15, λ=1, γ=1, τ=0.1, linkset, union, hop_expand=3）
```
MAZE_GEDIG_SP_BOUNDARY=trim \
python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --maze-size 25 --max-steps 120 --seeds 1 \
  --lambda-weight 1.0 --max-hops 15 --decay-factor 0.7 --adaptive-hops \
  --sp-beta 1.0 --linkset-mode --sp-scope union --sp-hop-expand 3 \
  --theta-ag 0.2 --theta-dg 0.15 --top-link 1 --anchor-recent-q 32 \
  --norm-base link --action-policy softmax --action-temp 0.1 \
  --gh-mode greedy --eval-all-hops \
  --persist-graph-sqlite experiments/maze-query-hub-prototype/results/short_greedy/graph.sqlite \
  --persist-namespace short_greedy \
  --output experiments/maze-query-hub-prototype/results/short_greedy/summary.json \
  --step-log experiments/maze-query-hub-prototype/results/short_greedy/steps.json
python experiments/maze-query-hub-prototype/build_reports.py \
  --summary experiments/maze-query-hub-prototype/results/short_greedy/summary.json \
  --steps   experiments/maze-query-hub-prototype/results/short_greedy/steps.json \
  --out     experiments/maze-query-hub-prototype/results/short_greedy/interactive.html
```

S2b（`--commit-from link`）
```
python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --maze-size 25 --max-steps 60 --seeds 1 \
  --lambda-weight 1.0 --max-hops 15 --sp-scope union --sp-hop-expand 3 \
  --eval-all-hops --linkset-mode --theta-ag 0.2 --theta-dg 10 \
  --top-link 1 --commit-budget 0 --norm-base link \
  --commit-from link \
  --gh-mode radius --snapshot-mode after_select \
  --sp-cache --sp-cache-mode core --sp-cand-topk 24 --sp-pair-samples 512 \
  --sp-beta 1.0 --action-policy softmax --action-temp 0.1 \
  --persist-graph-sqlite experiments/maze-query-hub-prototype/results/ablations/S2b/graph.sqlite \
  --persist-namespace maze_qh \
  --output experiments/maze-query-hub-prototype/results/ablations/S2b/summary.json \
  --step-log experiments/maze-query-hub-prototype/results/ablations/S2b/steps.json
python experiments/maze-query-hub-prototype/build_reports.py \
  --summary experiments/maze-query-hub-prototype/results/ablations/S2b/summary.json \
  --steps   experiments/maze-query-hub-prototype/results/ablations/S2b/steps.json \
  --out     experiments/maze-query-hub-prototype/results/ablations/S2b/interactive.html
```

S2a
```

S-obs-no-guard（観測ガード無効化; 壁/不可能行動も候補許容）
- 変更点: obs 候補の抽出で `action in possible_moves` / `passable` のフィルタを無効化
- CLI: `--no-obs-guard`
- 期待: 行き止まりでの候補探索が増え、softmax τ が低い場合にランダム性が残る。壁方向選択のログ観察が可能。

S-link-forced-as-base（forcedフォールバックをlinksetの台に採用）
- 変更点: S_link が空のとき `forced_links` を linkset の `s_link` として扱い、正規化の台やlinkset IGのbefore/afterを `before: forced`, `after: forced + query` 相当にする。
- 影響: フォールバック時の安定化（ΔH/ΔGED の基準が0ではなくforced台になる）。
python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --maze-size 25 --max-steps 60 --seeds 1 \
  --lambda-weight 1.0 --max-hops 15 --sp-scope union --sp-hop-expand 3 \
  --eval-all-hops --linkset-mode --theta-ag 0.2 --theta-dg 10 \
  --top-link 1 --commit-budget 0 --norm-base link \
  --commit-from link \
  --gh-mode radius --snapshot-mode after_select \
  --sp-cache --sp-cache-mode core --sp-cand-topk 24 --sp-pair-samples 512 \
  --sp-beta 1.0 --action-policy softmax --action-temp 0.1 \
  --output experiments/maze-query-hub-prototype/results/ablations/S2a/summary.json \
  --step-log experiments/maze-query-hub-prototype/results/ablations/S2a/steps.json
python experiments/maze-query-hub-prototype/build_reports.py \
  --summary experiments/maze-query-hub-prototype/results/ablations/S2a/summary.json \
  --steps   experiments/maze-query-hub-prototype/results/ablations/S2a/steps.json \
  --out     experiments/maze-query-hub-prototype/results/ablations/S2a/interactive.html
```

補助トグル（組合せでシークエンスを完全分解）
- gh-mode: `--gh-mode greedy|radius`
  - greedy: 現行（hop>0 は候補を1本ずつ貪欲採用）
  - radius: 追加配線なし（afterはh=0固定）、評価半径だけ拡大＝「g0, g(h)同時取得」に相当
- snapshot-mode: `--snapshot-mode before_select|after_select`
  - before_select: 候補選択前に prev_graph を確保
  - after_select: 候補選択後（現行既定）に prev_graph を確保
- pre-eval: `--no-pre-eval` で hop0 前診断（IG/SPのみ）を無効化

例（同一S1で組合せ）
```
# S1 + greedy + after_select
python ... --top-link 0 --gh-mode greedy --snapshot-mode after_select --output .../S1_greedy_after.json

# S1 + radius + before_select （g0, g(h)同時取得／半径のみ拡大）
python ... --top-link 0 --gh-mode radius --snapshot-mode before_select --output .../S1_radius_before.json

# S1 + greedy + no-pre-eval（事前診断なし）
python ... --top-link 0 --gh-mode greedy --no-pre-eval --output .../S1_greedy_nopre.json
```

メモ
- A1は論文の式(5)の直感（構造台を固定し、情報項で差分）に最も近い。A2は比較用。
- 今後: SP=all-pairs（connected-only）や anchors/expand のスイープ、Ecandフォーカス（最近QのQ→過去Q優先）も追試。

短評テンプレ
- 観測: g_min(mh)の出方、SP Debugのpair_count/Lb/top δSP、Per-hop(g, ΔGED, IG, ΔSP)の傾向
- 所感: 表示と内部値の一致（debug g_min(json/calc)）、挙動上の違和感
- 次手: どの軸を詰めるか（anchors/expand、Ecand、SP定義、IGのhop適用 等）

---

観測メモ（今回のセット）

- 症状: GED(h) が 1.0 に張り付き（上限飽和）しやすい。g0 の GED と定義が食い違っている可能性が高い。
  - 影響: g_min(mh), ΔGED(min,mh) が実際よりも悪化寄りに固定され、multi-hop の差が情報項だけでは出にくく見える。
  - 仮説原因:
    - hop>0 側での GED 正規化（Cmax）の取り方が g0 とズレている、もしくは subgraph 抽出が hop0 と不整合。
    - ged_hop0_const=true のときに「表示は const」だが内部で再計算している箇所があり、1.0 へクリップされた可能性。

- 具体例: seed=0, step=32
  - 観測: hop0 の ΔGED=0（g0 側では構造差分なし）なのに、hop>0 系列の GED が高止まり（1.0）する。
  - 解釈: そのステップでは候補（Top‑L/強制/仮想採用）の扱いが before/after ともに一致しておらず、評価用 after_graph_h の構成が誤っているか、候補選択（commit_items/Ecand）に不整合がある可能性。

対応（アブレーション表への記述）

- A1: GED を hop0 固定（ged_hop0_const=true）にした設定では、表示上の ΔGED(min,mh) は 0 に近くなるはず。実データで 1.0 に張り付く個所があれば、定義不一致の疑いとして注記。
- A2: hop>0 で GED を再計算（ged_hop0_const=false）。ここで ΔGED(min,mh) が 1.0 に張り付く場合は subgraph/正規化の不整合を優先疑義として記載。

次手（検証TODO）

1) GED 一貫性チェック（g0 vs hop>0）
   - 分母: どちらも Cmax=1+|S_link| を明示指定（norm_override）して等価化する。
   - サブグラフ: before/after で同一アンカー集合・同一 hop 展開（union/trim 規則も一致）を強制。

2) step=32 のログ再検証（seed=0）
   - pre/after の anchor_nodes、commit_items、Ecand、stage_graph の構成をダンプし、hop0=0 かつ hop>0=1.0 の原因を特定。
   - 「既に edge が存在」しているケース（0ΔGED）を事前に検知し、候補除外または δGED=0 に固定して評価。

3) 可視化/HTML
   - Per-hop Metrics テーブルに ΔGED の分母（Cmax）と使用したアンカー数の簡易統計（nodes/edges）を追加（デバッグ用）。
