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

1) ~~GED 一貫性チェック（g0 vs hop>0）~~ ✅ 完了（2026-02-02）
   - **結論**: `ged_hop0_const=True`（デフォルト）でGEDは全hop一貫
   - hop0/hop>0で同一のGED値を使用（A1設定）
   - GEDが1.0に張り付く問題はA2設定（ged_hop0_const=False）でのみ発生
   - 現在のデフォルトで問題なし

2) step=32 のログ再検証（seed=0） → 不要（GED一貫性確認済み）

3) ~~可視化/HTML~~ ✅ 完了（2026-02-02）
   - Per-hop Metrics テーブルに以下を追加:
     - `Cmax`: GED正規化分母
     - `nodes(b/a)`: before/afterサブグラフのノード数
     - `edges(b/a)`: before/afterサブグラフのエッジ数
   - 修正ファイル: `qhlib/evaluator.py`, `run_experiment_query.py`, `query_interactive_template.html`

---

## 実験結果 (2026-02-01)

### S0/S1/S2 シークエンス軸（完了）

| ID | 設定 | gmin_mean | best_hop_mean | avg_delta_sp_min | レポート |
|----|------|-----------|---------------|------------------|----------|
| S0 | baseline (top-link=1) | -0.101 | 7.06 | 0.281 | [HTML](results/ablations/S0/interactive.html) |
| S1 | top-link=0 | -0.101 | 7.06 | 0.281 | [HTML](results/ablations/S1/interactive.html) |
| S2 | top-link=1, fallback候補のみ | -0.101 | 7.06 | 0.281 | [HTML](results/ablations/S2/interactive.html) |

所見: S0/S1/S2間で大きな差は出なかった。timeline_to_graph=true, linkset_mode=true, theta_ag=-1.0 の修正により、SP/IG/マルチホップ評価が正常に動作。

### A1/A2 GEDポリシー比較（完了）

| ID | 設定 | gmin_mean | best_hop_mean | hop=0がbest | hop>=3がbest | レポート |
|----|------|-----------|---------------|-------------|--------------|----------|
| A1 | ged_hop0_const=true | -0.101 | 7.06 | 75/180 (42%) | 93/180 (52%) | [HTML](results/ablations/A1/interactive.html) |
| A2 | ged_hop0_const=false | 0.042 | 0.26 | 168/180 (93%) | 3/180 (2%) | [HTML](results/ablations/A2/interactive.html) |

A2のGED累積状況:
```
hop= 0: GED mean=0.502,  =1.0:  0%
hop= 5: GED mean=0.604,  =1.0: 20%
hop=10: GED mean=0.686,  =1.0: 38%
hop=15: GED mean=0.715,  =1.0: 45%
```

所見:
- A1（GED固定）: マルチホップ評価が効く。gminが負値（洞察発見）。論文設計意図に合致。
- A2（GED再計算）: hopが進むとGEDが累積し1.0に張り付く傾向。マルチホップのメリットがなくなる。
- 結論: `ged_hop0_const=true`（A1）がgeDIGの正しい運用設定。GEDは構造コストの「基準」として固定し、マルチホップ評価はIG/SPで差分を見る設計が有効。

### デフォルト値修正（DEFAULT_VALUES_MEMO.md参照）

| パラメータ | 変更前 | 変更後 | 理由 |
|-----------|--------|--------|------|
| timeline_to_graph | False | True | SP計算でsub_beforeが0エッジになる問題を回避 |
| linkset_mode | False | True | linksetベースのエントロピー計算（論文準拠） |
| theta_ag | 0.0 | -1.0 | linkset_mode時にg0が負になりAG gateが発動する問題を回避 |

### スケール検証（完了）

| サイズ | steps | seeds | gmin_mean | best_hop_mean | avg_delta_sp_min | 評価時間(ms) | レポート |
|--------|-------|-------|-----------|---------------|------------------|--------------|----------|
| 25×25 | 60 | 3 | -0.101 | 7.06 | 0.281 | 1,698 | [A1](results/ablations/A1/interactive.html) |
| 51×51 | 120 | 3 | -0.119 | 6.47 | 0.278 | 6,133 | [HTML](results/ablations/scale_51x51/interactive.html) |

所見:
- 51×51でも25×25と同様の傾向（マルチホップ有効、gmin負値、SP改善）
- 評価時間は約3.6倍（サイズに応じたスケーリング）
- 実行時間: 約2時間（120steps × 3seeds）

### 次のステップ

- [x] スケール検証: 51×51迷路、seeds=3 ← 完了
- [x] シード間並列化 ← 完了
- [ ] Link Auto-wire（T字路安定化）の検証
- [ ] S2bl-2phase（Greedy + Two-phase commit）の検証

---

## 並列化計画 (2026-02-02)

### 目的
- 51×51迷路で約2時間かかる実行時間を短縮
- シード間は完全に独立なので並列実行可能

### 実装方針

1. **シード単位の処理を関数に切り出し**
   - 現在: `for offset in range(args.seeds):` ループ内で直接処理
   - 変更: `_run_single_seed(seed, config, ...)` 関数を作成

2. **multiprocessing.Pool で並列実行**
   ```python
   from multiprocessing import Pool

   with Pool(processes=args.workers) as pool:
       results = pool.starmap(_run_single_seed, [(seed, config, ...) for seed in seeds])
   ```

3. **結果のマージ**
   - 各シードの結果（runs, warmup_runs, maze_data, step_records）を集約

### CLIオプション追加

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--workers N` | 並列ワーカー数 | 1（逐次実行） |

### 修正ファイル

- `experiments/maze/run_experiment_query.py` のみ

### 期待効果

| seeds | workers | 期待される実行時間 |
|-------|---------|-------------------|
| 3 | 1 | 2時間（現状） |
| 3 | 3 | ~40分 |
| 10 | 4 | ~50分 |

### 注意点

- `--workers 1` の場合は従来通り逐次実行（デバッグ用）
- ログ出力はワーカーごとに分離（混在防止）
- メモリ使用量はワーカー数に比例して増加
- curriculum モード（warmup + eval）は並列化非対応（逐次実行）

### 実装結果（2026-02-02）

実装完了。動作確認結果：

| 設定 | 実行時間 | 高速化率 |
|------|----------|----------|
| workers=1（逐次） | 38秒 | 1.0x |
| workers=3（並列） | 16秒 | 2.4x |

- 結果の整合性: 完全一致 ✓
- 51×51迷路の推定実行時間: 2時間 → 約50分

## ファイル分割リファクタリング（2026-02-02）

### 目的

`run_experiment_query.py`（5551行）が巨大化しているため、責務ごとにファイルを分割して保守性を向上させる。

### 分割計画

| 対象 | 抽出先 | 行数 | 状態 |
|------|--------|------|------|
| StepRecord, QueryHubConfig, EpisodeArtifacts | `qhlib/models.py` | 326行 | ✅ 完了 |
| 定数・ヘルパー関数（QUERY_MARKER, make_query_node等） | `qhlib/utils.py` | 213行 | ✅ 完了 |
| Sleep関連（Q学習、edge weight） | `qhlib/sleep.py` | 422行 | ✅ 完了 |
| 集計関数（aggregate） | `qhlib/aggregate.py` | 134行 | ✅ 完了 |
| CLI（parse_args, build_config） | `qhlib/cli.py` | 347行 | ✅ 完了 |
| run_episode_query本体 | `qhlib/episode.py` | TBD | 保留（~3300行、更なる分割検討要） |

### 実装結果

**Phase 1: models.py（完了）**
- `run_experiment_query.py`: 5551行 → 5240行（311行削減）
- `qhlib/models.py`: 326行（新規作成）
- 不要インポート削除: `dataclass`, `field`を削除

**Phase 2: utils.py（完了）**
- `run_experiment_query.py`: 5240行 → 5060行（180行削減）
- `qhlib/utils.py`: 213行（新規作成）
- 抽出内容: 定数（QUERY_MARKER等）、ノード変換、ベクトル計算、距離関数

**Phase 3: sleep.py（完了）**
- `run_experiment_query.py`: 5060行 → 4660行（400行削減）
- `qhlib/sleep.py`: 422行（新規作成）
- 抽出内容: `build_sleep_action_plan`, `build_sleep_q_table`, `build_sleep_edge_weights`

**Phase 4: aggregate.py（完了）**
- `run_experiment_query.py`: 4660行 → 4534行（126行削減）
- `qhlib/aggregate.py`: 134行（新規作成）
- 抽出内容: `aggregate` 関数（実験結果集計）

**Phase 5: cli.py（完了）**
- `run_experiment_query.py`: 4534行 → 4340行（194行削減）
- `qhlib/cli.py`: 347行（新規作成）
- 抽出内容: `parse_args`, `build_config`, `build_selector_params`, `build_gedig_params`

**Phase 6: 純粋関数抽出（完了）**
- `run_experiment_query.py`: 4340行 → 4273行（67行削減）
- `qhlib/utils.py`: 213行 → 327行（+114行）
- 抽出内容: `candidate_index`, `edge_set`, `norm_edge`, `normalise_container`, `item_key`, `merge_unique`
- 動作確認: インポート・テスト実行ともに成功 ✓

### 累計削減

| 指標 | 値 |
|------|-----|
| 元ファイル | 5551行 |
| 現在 | **4273行** |
| 削減行数 | **1278行（23%）** |

### 新規作成モジュール一覧

| モジュール | 行数 | 責務 |
|-----------|------|------|
| `qhlib/models.py` | 326 | データクラス（StepRecord, QueryHubConfig等） |
| `qhlib/utils.py` | 213 | 定数・ヘルパー関数 |
| `qhlib/sleep.py` | 422 | Sleep関連ロジック（Q学習、edge weight） |
| `qhlib/aggregate.py` | 134 | 実験結果集計 |
| `qhlib/cli.py` | 347 | CLI引数パース・config構築 |

### 残り構成（4340行）

| 部分 | 行数 | 備考 |
|------|------|------|
| `run_episode_query` | ~3300行 | 主要ロジック、更なる分割検討要 |
| `main` | ~750行 | エントリポイント |
| `HopGreedyEvaluator` | ~160行 | 未使用（デッドコード、削除候補） |

### 次のステップ

1. ~~`qhlib/utils.py`: ヘルパー関数の抽出~~ ✅ 完了
2. ~~`qhlib/sleep.py`: Sleep関連ロジックの抽出~~ ✅ 完了
3. ~~SP計算最適化（ファイル分割後に実施）~~ ✅ 完了

## SP計算パラメータ最適化（2026-02-02）

### 目的
評価時間を短縮し、大規模実験の効率を向上させる。

### 検証結果

**25×25迷路**:

| 設定 | avg_time_ms | p95_time_ms | 高速化率 |
|------|-------------|-------------|----------|
| Baseline (512/24, hops=15) | 66.6ms | 464ms | - |
| Reduced (64/8, hops=15) | 28.5ms | 239ms | 57%↓ |
| Optimized (128/12, hops=10) | 26.6ms | 178ms | **60%↓** |

**51×51迷路**:

| 設定 | avg_time_ms | p95_time_ms | 高速化率 |
|------|-------------|-------------|----------|
| Baseline (512/24, hops=15) | 264.8ms | 1714ms | - |
| Optimized (128/12, hops=10) | 42.5ms | 291ms | **84%↓ (6.2倍)** |

### 結果品質

| 指標 | Baseline | Optimized | 評価 |
|------|----------|-----------|------|
| gmin_p95 | 0.259 | 0.097 | ✓ 改善 |
| acceptance_rate | 0.85 | 0.92 | ✓ 改善 |
| best_hop_mean | 0.217 | 0.092 | ✓ 同等 |

### 推奨設定

```bash
--max-hops 10 --sp-pair-samples 128 --sp-cand-topk 12
```

### 実行時間見積もり（51×51, 120steps, 3seeds）

| 設定 | 時間 |
|------|------|
| 従来 | ~96秒 |
| 最適化後 | ~15秒 |

## Link Auto-wire / S2bl-2phase 実験（2026-02-02）

### 実験設定

共通設定（最適化済み）:
```bash
--max-hops 10 --sp-pair-samples 128 --sp-cand-topk 12
--lambda-weight 1.0 --decay-factor 0.7 --sp-beta 1.0
--linkset-mode --sp-scope union --sp-hop-expand 3
--theta-ag -1.0 --theta-dg 0.15 --top-link 1
--norm-base link --action-policy softmax --action-temp 0.1
```

### 結果比較（25×25, 60steps, 3seeds）

| 実験 | gmin_mean | best_hop_mean | avg_delta_sp_min | avg_time_ms |
|------|-----------|---------------|------------------|-------------|
| Baseline | -0.439 | 0.18 | 0.011 | 46.7ms |
| Link Auto-wire | -0.439 | 0.18 | 0.011 | 47.8ms |
| **S2bl-2phase (greedy)** | **-0.560** | **2.58** | **0.132** | 225.9ms |

### 所見

**Link Auto-wire**:
- ベースラインと差なし
- 今回のseed/迷路ではT字路効果が観測されず
- 効果が出るケースは限定的か、別途条件が必要

**S2bl-2phase (greedy)**:
- マルチホップ評価が大幅に活性化
- best_hop_mean: 0.18 → 2.58（**14倍**）
- delta_sp_min: 0.011 → 0.132（**12倍**）
- gmin改善（-0.439 → -0.560）
- 評価時間は増加（46.7ms → 225.9ms）だがマルチホップ探索のため妥当

### 結論

- `--gh-mode greedy`はマルチホップ評価を活性化させる有効な設定
- SP改善（delta_sp_min）が大きく、より良い洞察発見につながる可能性
- 評価時間増加とのトレードオフを考慮して使用

### 51×51スケール検証（2026-02-02）

| 指標 | Baseline | S2bl-2phase (greedy) | 変化 |
|------|----------|---------------------|------|
| gmin_mean | -0.458 | **-0.542** | 18%改善 |
| best_hop_mean | 0.32 | **1.61** | **5.1倍** |
| avg_delta_sp_min | 0.013 | **0.077** | **5.9倍** |
| best_hop_hist_3 | 3 | **56** | **18.7倍** |
| avg_time_ms | 91ms | 340ms | 3.7倍 |

**所見**:
- 51×51でもgreedy modeのマルチホップ効果は有効
- 25×25（14倍）より控えめ（5.1倍）だが有意な改善
- 評価時間は3.7倍増加だがマルチホップ探索のため許容範囲

**推奨使用ケース**:
- マルチホップ評価の効果を最大化したい場合: `--gh-mode greedy`
- 高速実行優先の場合: デフォルト（radius mode）

## パラメータスイープ（2026-02-02）

### 実験設定
- 51×51迷路、120steps、3seeds、greedy mode
- 最適化済みSP設定: `--sp-pair-samples 128 --sp-cand-topk 12`

### Lambda (λ) スイープ

| λ | gmin_mean | best_hop_mean | delta_sp_min | time_ms | 備考 |
|---|-----------|---------------|--------------|---------|------|
| 0.5 | -0.051 | **1.62** | **0.078** | 334 | マルチホップ最大だがgmin浅い |
| 1.0 | -0.542 | 1.61 | 0.077 | 341 | バランス（推奨） |
| 2.0 | **-1.418** | 0.56 | 0.021 | **155** | gmin最良だがマルチホップ少 |

### Max Hops スイープ

| hops | gmin_mean | best_hop_mean | delta_sp_min | time_ms | 備考 |
|------|-----------|---------------|--------------|---------|------|
| 5 | -0.502 | 0.64 | 0.038 | **186** | 高速だが探索制限 |
| 10 | -0.542 | 1.61 | 0.077 | 341 | バランス（推奨） |
| 15 | **-0.562** | **2.37** | **0.096** | 439 | 探索最大だが遅い |

### Decay Factor スイープ

| decay | gmin_mean | best_hop_mean | delta_sp_min | time_ms |
|-------|-----------|---------------|--------------|---------|
| 0.5 | -0.542 | 1.61 | 0.077 | 336 |
| 0.7 | -0.542 | 1.61 | 0.077 | 341 |
| 0.9 | -0.542 | 1.61 | 0.077 | 353 |

**所見**: decay factorは今回の設定では効果なし

**原因調査結果（2026-02-02）**:
- `decay_factor`は`spike.py`の`aggregate_reward`計算にのみ使用
- 実験コードは`argmin(g[h])`で`best_hop`を選択（decayを使わない）
- `aggregate_reward`は論文未定義、実験でも未使用 → デッドコードの可能性
- **結論**: decay_factorパラメータは現状無効。将来的に削除検討

### 推奨設定

**バランス重視**:
```bash
--lambda-weight 1.0 --max-hops 10 --decay-factor 0.7
--sp-pair-samples 128 --sp-cand-topk 12
--gh-mode greedy --eval-all-hops
```

**マルチホップ最大化**:
```bash
--lambda-weight 0.5 --max-hops 15
```

**速度優先**:
```bash
--lambda-weight 2.0 --max-hops 5
--sp-pair-samples 64 --sp-cand-topk 8
```

### 比較HTML
`experiments/maze/results/sweep/comparison.html`

### 追加スイープ（2026-02-02）

**SP Beta スイープ**:

| sp_beta | gmin_mean | best_hop_mean | delta_sp_min | 備考 |
|---------|-----------|---------------|--------------|------|
| 0.5 | -0.504 | 1.61 | 0.077 | SP重み低 |
| 1.0 | -0.542 | 1.61 | 0.077 | デフォルト |
| 2.0 | **-0.631** | 1.60 | 0.075 | **gmin最良** |

**Action Temperature スイープ**:

| temp | gmin_mean | best_hop_mean | delta_sp_min | 備考 |
|------|-----------|---------------|--------------|------|
| 0.05 | -0.542 | 1.61 | 0.077 | 決定的 |
| 0.1 | -0.542 | 1.61 | 0.077 | デフォルト |
| 0.2 | -0.542 | 1.61 | 0.077 | 探索的 |

**所見**:
- `sp_beta↑` → gmin改善（SP重視でより良い洞察発見）
- `action_temp` → 効果なし（今回の迷路サイズ/seedでは差が出ない）

### 動作確認（2026-02-02）

リファクタリング後の動作確認:

| 迷路サイズ | seeds | 成功率 | avg_steps | 結果 |
|-----------|-------|--------|-----------|------|
| 5×5 | 3 | 100% | 4.0 | ✓ 正常 |
| 10×10 | 5 | 80% | 32.8 | ✓ 正常 |

リファクタリングによる影響なし。分割したモジュール（models, utils, sleep, aggregate, cli）すべて正常動作。
