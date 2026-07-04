# 気づきメモ: エッジ流れ場ナビゲーション(node記憶 → sleep伝播 → エッジ嗜好 → QKV)

**日付**: 2026-07-04
**ステータス**: 🔄 仮線・実装前(既存フラグで半分は即検証可能 — §5)
**関連**: sleep ライン v1-v3([[sleep-line-2026-07]])、[gedig_prediction_curl.md](gedig_prediction_curl.md)、[gedig_triangular_contrastive_learning.md](gedig_triangular_contrastive_learning.md)、SPEC.md §2.2/§4
**発端**: 2026-07-04 のユーザーとの議論。「交差点で記憶を元にゴール方向が選べる伝播がしてほしい」「正例負例の記憶は node に宿るでよいが、sleep でそれがエッジを辿って伝播し、行き止まりを避けゴールに向かうようエッジ重みに嗜好が伝播すべき」「それが Transformer の QKV に近いアイデアになっていくのでは」

---

## 0. 前提(ユーザーの選択で確定)

- **消去法派**: エージェントはゴール座標を**知らない**前提。潜在ポテンシャル shaping(φ = −dist(s,goal))は「記憶を元に」ではなく「ヒントを元に」なので主軸にしない(対照条件として置くのはあり)。
- 「行けない場所を全部覚えれば、残りが道」— 記憶構造そのものからゴール方向が浮かび上がるのが目標。これは構造の消去 = Δβ₁ 的でもあり、geDIG の思想と整合。

## 1. 交差点での選択に効く3つの力

| チャネル | 中身 | 現状(v2/v3, abs mode) |
|---|---|---|
| **Pull** | goal 正報酬が γ 減衰で軌跡を遡り、交差点で goal 側 Q が高い | 実現・実証済み(成功層 −51%)。**到達記憶がある時のみ** |
| **Push** | 経験済み(負〜0) vs 未探索(0)の相対差が未知へ押す | 実現済みだが**等方的**(全未探索が一律 0、ゴール寄りの未知を優先できない) |
| **Eliminate** | 負例で枝を潰し、消去法で残りが道 | **半分だけ稼働**(Q は possible_moves を 0 初期化 → 未経験行動が消去法で選ばれる)。ただし deadend/blocked ペナルティが既定 0.0 で「枝の死」が浅い |

未達エピソードでは Pull は原理的に不在(ゴール位置が記憶に無い)。よって消去法ナビの本体は
**Eliminate の彫り込み + Push の非等方化**。

## 2. 中核アイデア: エッジ重み = 独立チャネルではなく「値の差 = 流れ場」

sleep 後、各通路 {u,v} に対し離散勾配場:
```
flow(u→v) = V(v) − V(u)      (V(s) = max_a Q(s,a)、既に query node に転写済み)
```
- goal 到達エピソード → 流れがゴールへ収束(Pull が流れの言葉になる)
- 袋小路枝(deadend 負例が Q に入っていれば)→ 枝内の流れが**入口向きの逆流** = 「行くな」ではなく「**出ろ**」の向き付き信号。**Eliminate と Pull が同一の流れ場に統一される**
- 交差点 readout: `flow(現在 → 候補)` の比較 = 「記憶を元にゴール方向を選ぶ」がエッジ量比較として実現

**設計上の美点**: 新しい学習パラメータがゼロ(既存 Q の差を読むだけ)。ユーザーの自律化原則
「ノブをメタ制御せず局所学習状態に溶かす」(Wake2 で 4β→1α に畳んだ前例、SPEC §4.1)に完全一致。
独立エッジ重みチャネルを作ると学習則・減衰・初期値の 3 ノブが生えるが、流れ場は生えない。

## 3. QKV との対応(ユーザーの予感の展開)

現在の行動選択は既に構造的に attention:
- query vector(観測 + dim8/9 のターゲット値)× direction node の abs_vector の重み付きマッチング + softmax = **q·k attention**
- weight_vector = スケーリング、**dim9 = キーに焼き込まれた価値(value)**
- **sleep(dim9 書き込み)= value-modulated attention の学習**(v2 で実証済み)

対応表:
| geDIG maze | Transformer |
|---|---|
| node に記憶(reward, propagated) | K, V がトークンに宿る |
| query × direction のマッチング | q·k attention |
| sleep が dim9 を価値変調 | value 学習 |
| **flow = V(v)−V(u)(未踏)** | **attention の反対称成分 = 渦度 / curl** |

flow は curl 系メモ([gedig_prediction_curl](gedig_prediction_curl.md))の数学と同一。迷路で flow readout が
機能したら、**同じ演算子を Part 4 の attention flow 解析へ持ち込める** — 迷路→Transformer の橋が
比喩ではなく演算子の同一性で架かる。

## 4. 保存場の留保(重要な限界)

V 差の流れ場は**保存場(ポテンシャル勾配)しか表現できない**。「A→B は好むが B→A は嫌」までは書けるが
「ぐるぐる回りたい」回転選好は書けない。
- 迷路では健全な制約(回転=ループ=非効率)。むしろ **curl≠0 の検出 = 値地形の矛盾の診断信号**として
  消去法の検算器に使える。
- RAG では文脈依存の通行選好(同じエッジでもクエリ次第で向きが変わる)が要る可能性 → そこで初めて
  独立エッジチャネル(= フル QKV、非保存場)が正当化される。**段階論**: 迷路は保存場で足り、
  非保存場は RAG の必要が示してから。

## 5. 実装の現状(2026-07-04 コード確認)

**驚き: flow readout は既に実装済み**。`--propagated-mode gradient`
(`run_experiment_query.py::_propagated_for_action`)が `prop_next − prop_here` を返す = §2 の flow。
v2/v3 は `abs`(既定、direction node の生 Q)で走っていた。**議論の半分は既存フラグ切替で即検証可能**。

未実装/未整合:
1. **deadend/blocked ペナルティが既定 0.0**(`build_sleep_q_table`)。枝の死が Q に十分入っていない
   → dim8(wake 記録 deadend −1.0)と dim9(Q deadend 0.0)の価値体系が乖離
2. flow が **query node の V(=max Q) 差**であることの確認と、交差点での readout 比較の明示化
3. `sleep_replay_optimize` が書く `edges[u][v]["propagation_weight"]` は現状**読み手ゼロ(死にコード)**
   — flow を陽に持たせるならここを活かせる(あるいは on-the-fly 計算のまま)

### 実装レビューの発見(2026-07-04、探索ラン起動後の照合)

- ✅ **gradient = flow の同一性を確認**: `_propagated_at_pos` = その位置の direction node の
  propagated の max = V(s)。gradient 分岐は正確に V(next) − V(here)
- ✅ **未探索セルの扱いは失敗層で正しく機能する構造**(設計時の見落とし、コードが賢かった):
  グラフ外セルは V=0 扱い → 失敗層(V(here)≤0)では未探索方向の flow ≥ 0 で相対浮上(Push 維持)、
  全行動が負に沈んだ場所では flow > 0 = 脱出方向として浮く。消去法と流れ場は干渉しない
- ⚠ **gradient 切替は α バイアス経路のみ**: dim9 類似度経路は abs 値(tanh Q)のまま残る。
  つまり「gradient モード」は flow + abs 類似度のハイブリッドであり、純粋 flow readout ではない。
  prereg の機構分解の論点(dim9 ゼロ化条件を作るか、weight_vector で切るか)
- ⚠ **blocked_penalty はほぼ飾り**: blocked 遷移は s2==s の自己ループで
  Q(s,壁) ← r−1.01 + 0.99·V(s) とペナルティが自己 V で薄まり、さらに観測ガードで壁方向は
  候補から除外されるため readout に乗らない。彫り込みの本命は deadend のみ
- ⚠ **操作チェック(枝内 flow 向き)の道具が未実装**: Q テーブルは JSON に保存されない。
  prereg 前に per-seed メタへの flow 要約追加 or warmup 軌跡からの Q 再計算スクリプトが必要
- 探索ラダーは 2×2 に修正(abs/grad × nopen/deadend の 4 条件、seed=52)

## 6. (a)再設計 prereg の骨子(消去法ナビ)

- **操作**: 2 ノブの独立切替 —
  (i) deadend/blocked ペナルティ有効化(dim8 と価値統一。**調整ではなく整合性原則**で登録可)
  (ii) readout を flow(gradient mode)に切替
- **難度層**: off が自力ナビで解けない迷路に限定(v3 の反省 — off でも 5/12 解けたので成否検定が死んだ)。
  選定は事前に「off eval FAIL のシード」で層別(決定的)
- **主検定**: v3 の教訓から**質を主に**(eval 歩数 or 袋小路)。成否は副次
- **操作チェック(この設計の肝)**: 袋小路枝**内**の flow が入口向き(負)か、を伝播値分布で機械検証。
  「Eliminate が向きとして効いている」ことの装置内証明
- **対照(上限)**: potential shaping(座標ヒントあり)を「記憶だけでどこまで行けるか」の天井として併置

## 7. 次アクション

1. 本メモをコミット(議論の保存)← 完了
2. **予備実験(探索、非登録)**: 難度シード数個で `--propagated-mode gradient` × deadend ペナルティ ON を
   `results/**/_exploratory_*/` で試し、flow が枝内で逆流するかを可視化 ← 完了(§8)
3. 感触が得られたら (a)prereg を §6 の骨子で FROZEN 化 → 未使用シードで確証 ← **§8 により 1 ノブに簡約して v4 へ**

## 8. 探索結果(2026-07-04、seed=52 の 2×2)と readout 等価性の発見

探索記録: `experiments/maze/results/graph_persistent_dg/_exploratory_flow/`(probe.log + NOTES.md)。
seed=52 は v3 で replay(abs) 256 歩が off 108 歩に負けた「伝播が害」の診断ケース。

| | ペナルティなし | deadend/blocked −1.0 |
|---|---|---|
| abs | 256 歩・袋小路 6 | **104 歩・袋小路 2** |
| gradient(flow) | 256 歩・袋小路 6 | **104 歩・袋小路 2** |

**発見 1 — ヒーローは彫り込み**: deadend −1.0 だけで 256→104(−59%)、**off(108)を初めて下回った**。
「負例の伝播で行き止まりを避ける」の原直感が、軌跡 backup + ペナルティで本来の力を出した。

**発見 2 — abs と flow は候補比較において情報等価(数学)**: 同一交差点での候補比較では
`Q(here,a) = r + γ·V(next_a)` と `flow_a = V(next_a) − V(here)` は、r と V(here) が候補間で
共通の定数なので**どちらも V(next_a) の単調関数 → argmax が一致**する。4 条件で歩数まで完全同一
だったのは偶然ではなく必然。**flow readout は行動バイアスとしては不要**。
flow の独自価値は反対称性を使う用途 — 往路/復路の区別、curl 診断(値地形の矛盾検出)— に限られる。

**§3(QKV 対応)の更新**: 「エッジ嗜好」の実用は **V 地形を正しく彫ること(value shaping)が本体**で、
readout 形式(abs/flow)は等価類。Transformer への翻訳も「value 学習が本体、attention の読み出し形式は
等価な族」という一段精密な読みになる。

**帰結**: (a)prereg は **deadend/blocked ペナルティの dim8 価値統一という 1 ノブ**に簡約
→ [docs/prereg/maze_sleep_v4_deadend_carving.md](../../prereg/maze_sleep_v4_deadend_carving.md)(操作 1 変数の理想形)。
seed=52 は探索で消費したため確証(シード 60–89)から除外。

Related: [[sleep-line-2026-07]], [[prereg-protocol]], [[sleep-rag-design-2026-06]]
