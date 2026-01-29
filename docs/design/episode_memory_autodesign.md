# エピソード記憶の自己設計：AG/DG × Wake/Sleep × 自己生成負例

**Version**: 0.2 (Draft)  
**Date**: 2026-01-29  
**Author**: Kazuyoshi Miyauchi  
**Status**: Proposal (updated with Maze PoC status)

---

## 1. 背景：人間の解き方 = 負例生成のループ

人間の認知は、ざっくり言うと次の流れになっている。

1. 正解（デモ）を見る
2. 操作方法・UI（表現/操作の語彙）を把握する
3. ルールを抽象的に把握する（仮説を立てる）
4. 問題から回答を生成する

このとき本質的なのは、**「外れた仮説を捨てる」過程が、次の学習信号（負例）になっている**点である。

ARCの文脈では、trainの入出力例が「正解（デモ）」に相当し、テストの正解は見えない。したがって、負例生成は **train例への整合性**（および内部の整合性・簡潔性）から作られる必要がある。

---

## 2. 目的 / 非目的

### 2.1 目的
- **負例（Hard Negative）を自己生成**し、探索・記憶・表現学習に再利用する。
- **エピソード記憶の単位（chunking）**と**索引（index/embedding）**を、Wake/Sleep で自己改善する。
- 迷路PoCとARC（DSL探索）の両方に通る、実装可能な最小定義を置く。

### 2.2 非目的（最初はやらない）
- テスト正解の利用（ARC系では禁止/不可能）
- 大規模モデルによる end-to-end 生成（別トラック扱い）

---

## 3. 用語（最小）

- **Hypothesis（仮説）**：ルール、変換、部分プログラム、エピソード分割規則など「提案され、検証されるもの」。
- **Episode（エピソード）**：記憶に保存される経験単位（状態・行動・結果・コンテキスト）。
- **Positive（正例）**：DGで確定（commit）された仮説、または確定したエッジ/マクロ。
- **Negative（負例）**：AGで検討対象になったがDGで棄却、あるいは検証で破綻した仮説。
- **Hard Negative**：ほぼ正しいが、ある条件（例: 1つのtrain例）で破綻する仮説。

---

## 4. 全体ループ（geDIG + AG/DG）

この設計では、負例生成は「副産物」ではなく**第一級の成果物**として扱う。

### 4.1 Wake（オンライン）

1. **提案（Propose）**：候補仮説 `h`（候補エッジ/候補マクロ/候補プログラム/候補分割）を生成する
2. **軽量評価（0-hop）**：`g0(h)` を計算する（安価な近似）
3. **AG（Attention Gate）**：`g0 > θ_AG` なら探索を開く（候補増やす / 高コストDSL解放 / multi-hopへ）
4. **高価評価（multi-hop）**：`g_min(h)` を計算する（より厳密な検証）
5. **DG（Decision Gate）**：`min(g0, g_min) ≤ θ_DG` なら commit する
6. **ログ**：
   - commit された仮説 → **正例**
   - 棄却された仮説 → **負例（理由付き）**

> 注：AG/DGの形式は実装の `src/insightspike/algorithms/gating.py` と整合する。

### 4.2 Sleep（オフライン）

Sleep は「記憶や表現を作り直す時間」であり、負例を使って次を改善する。

- **負例マイニング**：近いが破綻した仮説（hard negatives）を優先抽出
- **表現の学習**：Episode/Task/Program の埋め込みを、対比学習で再配置（「醸成」）
- **マクロ圧縮**：DG確定の部分構造を共通化して DSL マクロ化（再利用率を上げる）
- **自己設計の更新（メタDG）**：
  - 例: 「どの粒度でエピソードを切るか」「どの特徴で索引するか」「負例の比率や温度」
  - 変更案を “仮説” として扱い、改善が一貫して出るときのみ commit する

### 4.3 DGを強くする：コミット（2-phase）とDG Ledger

DG感が強い状態とは、少なくとも次の3点が成立していること。

- **DGだけが永続状態を変える**（AGは探索を開くが、状態は“確定”しない）
- **commitは取引（transaction）**で、根拠（evidence）がログに残り、追試できる
- **commitが“たまに起きる”**（常時commitは情報が無いのと同義）

#### 4.3.1 commit対象（例）

- **迷路**：`dg_committed_edges` へのエッジ追加/削除、dead-endタグの確定、近道マクロ（n-step）の追加
- **ARC**：DSLマクロの追加/剪定、部分プログラムの確定（語彙化）
- **Sleep（学習成果の反映）**：
  - affordance prior（`P(passable|s,a)`）モデルのバージョン更新
  - 距離/温度など検索幾何の更新（Phase 0/2）
- **メタ（自己設計）**：エピソード分割規則、特徴集合、`k_cap`、探索温度、分位目標、`θ_AG/θ_DG` の校正規約

> 注：AG/DGの不等号が逆に見える問題は、`g0` と `gmin` を「同一スカラーの別表記」だと思うと起きる。実装では `g0` と `gmin` は別スコアで、AGは `g0 > θ_AG`、DGは `min(g0,gmin) ≤ θ_DG` で決まる（`src/insightspike/algorithms/gating.py`）。

#### 4.3.2 二相コミット（staging → evaluate → commit）

1. **staging**：候補（仮説）を一時領域へ置く（本体には反映しない）
2. **evaluate**：固定seed/短い検証で「最低限の改善」と「破綻しない」を見る
3. **commit**：基準を満たすときのみ、永続状態へ反映（以後は正例として扱う）
4. **rollback（revert）**：後続の観測で悪化が確認されたら、revert自体をDGイベントとして記録する

#### 4.3.3 採択基準（最小）

- **改善の最小セット**：Solved率・平均step・P95・invalid率（mazeならhit_wall率/ループ率も）
- **安定性**：複数seed/サイズで改善の符号が揃う（少なくとも“勝った/負けた”が再現する）
- **予算（commit budget）**：一定ステップ/一定エピソードあたりのcommit上限を置き、スパイク（場当たり統合）を防ぐ

#### 4.3.4 DG Ledger（監査ログ）最小スキーマ案

commit/revert を 1行1イベントとして JSONL に残す（「負例データセット」兼「再現の台帳」）。

```json
{
  "commit_id": "c_2026-01-29T12:34:56Z_001",
  "domain": "maze|arc|meta",
  "kind": "edge_add|macro_add|affordance_update|metric_update|chunking_update|revert",
  "hypothesis_id": "h_... (proposal hash)",
  "parent_commit_id": "c_... (optional)",
  "gate": { "g0": 0.12, "gmin": -0.03, "theta_ag": 0.5, "theta_dg": 0.0, "ag": true, "dg": true },
  "eval": {
    "seeds": [0,1,2],
    "metrics_before": { "solved": 0.60, "p95_steps": 210 },
    "metrics_after":  { "solved": 0.72, "p95_steps": 160 }
  },
  "decision": "commit|reject|revert",
  "reason": "p95_steps_improved_and_stable",
  "artifacts": ["path/to/trace.json", "path/to/model.pt"]
}
```

> 迷路PoCでは、`experiments/maze-query-hub-prototype/run_experiment_query.py` に `--dg-ledger-log <path>.jsonl` を付けると、`staged_edges`（提案）と `committed_edges`（実コミット）を分けたDGレジャーを出力できる。  
> また `--curriculum-warmup-steps N`（Wake→Sleep→Wake）で warmup→Sleep→eval を回せる。Sleep は warmup の経験遷移上で BFS により最短プランを作り、eval は `--sleep-guide override|prefer|off` で適用する。  
> 負例ラベルの最小実装として `--cortisol-mode log --cortisol-repeat-visits 2` を用意しており、2回目以降に踏んだマスへの移動を `cortisol_fire=true`（`cortisol_reason=revisit`）としてログ化できる。

---

## 5. 自己生成負例の種類（推奨カタログ）

負例は「単に失敗した」では弱い。**どこで破綻したか**が価値になる。

1. **即死負例（easy）**：明らかに矛盾（trainの多くで破綻）
2. **近接負例（near-miss）**：trainの大半は合うが、少数例で破綻
3. **不変量破壊（invariance-break）**：回転/平行移動など、想定不変量を崩すと破綻
4. **過剰自由度（overfit）**：train-fitはするが、記述が過度に複雑（ΔEPCが大きい）
5. **対立仮説（contradictory）**：同じデモを説明するが、他デモと両立しない

このうち 2) と 4) が、探索と汎化を強く鍛える（hard negative の主要供給源）。

---

## 6. データ設計（学習用ログの最小スキーマ案）

ログは「再現」だけでなく「学習」にも使う前提で、最初から構造化しておく。

### 6.1 Pair / Triplet（対比学習）

例：JSONL（1行1サンプル）

```json
{
  "domain": "maze|arc",
  "anchor": { "type": "state|task|partial_program", "repr": "..." },
  "positive": { "type": "episode|macro|program", "repr": "..." },
  "negatives": [
    { "repr": "...", "kind": "near_miss|overfit|invariance_break", "weight": 1.0 }
  ],
  "gate": { "g0": 0.12, "g_min": -0.03, "theta_ag": 0.5, "theta_dg": 0.0, "ag": true, "dg": false },
  "evidence": { "failed_example_ids": ["train:2"], "reason": "mismatch_cells=7" }
}
```

### 6.2 “負例生成”の禁止事項（ARC系）

- **test output は使わない**
- negative の判定根拠は **train例** と **内部コスト/整合性** からのみ作る

---

## 7. 迷路PoCへの写像（エピソード記憶）

迷路では、各ステップで「候補（観測/記憶からのリンク）→検証→行動」が起きる。

- **Anchor**：現在状態（位置、局所観測、直近の軌跡、近傍グラフ）
- **Hypothesis**：候補リンク、近道エッジ、エピソード分割（“ここから先は別チャンク”）など
- **Positive**：
  - DGで `dg_committed_edges` に入ったエッジ
  - あるいは（設計により）「その後の成功確率が上がった」エピソード分割
  - （迷路PoC）Sleepで得たプラン上の遷移（evalで `sleep_guided=true`）を正例として扱う（“短い到達”の教師）
- **Negative**：
  - AGで探索が開いたがDGで commit されなかった候補（hard negative の核）
  - `is_dead_end=true` に繋がりやすい候補（行動面の負例）
  - （迷路PoC）revisit：移動後のセルが2回目以降の訪問なら負例（`cortisol_fire=true`, `cortisol_reason=revisit`）

実験ログ（例：`experiments/maze-query-hub-prototype/results/*steps*.json`）には、`g0/gmin`、`theta_ag`、`ag_fire/dg_fire`、候補集合などが含まれるため、**負例データセットを外部教師なしで作れる**。

### 7.1 抽象定義 → 迷路エピソードの最小単位（導出）

論文側の抽象定義では、エピソードを「局所文脈（状態）」「操作（行為/クエリ）」「帰結（観測・成否・報酬近傍）」の最小まとまりとして扱う。
さらに表現レベルでは、次の最小分解（スキーマ）に対応づけられる：

- `context`（文脈／状態）
- `operation`（操作）
- `affordance`（制約・可用性）
- `salience`（頻度・重要度）
- `outcome`（帰結）
- `goal`（目標）

迷路では「スタートからゴールへ辿り着け」という命令があるが、部分観測（3×3）かつ逐次意思決定であるため、まずは **命令を“1ステップの意思決定問題”へ分割**するのが自然になる。
このとき、最小単位（原子エピソード）は次の形に落ちる：

- **Episode（迷路の原子）** = 「ある位置（文脈）で、ある方位へ進む（操作）ときの、制約/既知度/帰結/目標近傍を束ねたもの」
- **ノード粒度** = `(anchor_position, direction)`（実装上は direction node）
- **クエリ粒度** = “今この瞬間”の状態（実装上は query node）

これを、PoCの8次元ベクトルに対応づけると：

```
v = [x/W, y/H, dx, dy, wall, log(1+visits), success, goal]
w = [1, 1, 0, 0, 3, 2, 0, 0]  # 距離用の重み（例）
q = [x/W, y/H, 0, 0, 1, 0, 0, 0]  # クエリ（例）
```

- `x/W, y/H` → `context`（スケール不変な位置）
- `dx, dy` → `operation`（候補行動の方向）
- `wall` → `affordance`（可用性/制約）
- `log(1+visits)` → `salience`（既視/探索圧）
- `success` → `outcome`（成否のタグ）
- `goal` → `goal`（目標近傍のタグ）

> 注：重み `w` やクエリ `q` は「近傍候補の選別」に主に効かせ、`ΔH`（IG）はベクトル集合の類似度分布から推定する。よって「距離に使わない次元」も、IG側で意味を持ち得る。

### 7.2 「この単位が最小」と言うために DG log で見るべき観察

“最小”は主観ではなく、**削ると壊れる**（または **足すと得しない**）という観測で定義する。
迷路では DG log（ステップログ）から、次の観察が取れる。

#### A. ゲートの可観測性（AG/DGが意味を持つか）
- `g0` が「曖昧/新規っぽい局面」で上がる（AGが開く）
- `gmin` が「確証が取れた局面」で下がる（DGが確定する）
- `ag_fire` の発火率が分位校正どおりに安定（サイズ/seedを跨ぐ）
- `dg_fire` が“常時True/常時False”に潰れない（情報がある）

#### B. 候補分布の情報性（ΔHが死んでいない）
- `linkset_entropy_before/after` が極端に飽和しない（常に0や常に最大、にならない）
- `linkset_delta_h`（=IG成分）が、改善局面で一貫して正方向に出る

#### C. 統合の意味（DG確定が「近道」になっている）
- `dg_committed_edges` が、`delta_sp`（経路短縮）と整合して増える
- commit が増えても “誤統合の増加”に繋がらない（将来的にはFMR等の導入）

#### D. 反証可能性（負例が生成できている）
- `candidate_pool` / `ranked_candidates` に、near-miss（似ているが破綻）が一定割合で出る
- AGで開いた探索が、DGで棄却されるケースが生まれている（hard negativeが取れる）

### 7.3 抽象スキーマから「8次元」に収束させる探索（自己設計の手順案）

エピソードの粒度と特徴は、実装都合ではなく **“ゲートが働くための最小条件”** から逆算して決める。

#### Step 0: 粒度の候補を並べる（divide）
- **粗すぎ**：`context` のみ（位置だけ）  
  → 方向が失われ、近傍遷移が曖昧になりやすい（候補が同一視される）
- **現行（原子）**：`context + operation`（位置+方位）  
  → “次に何を試すか”という最小意思決定を表せる
- **細かすぎ**：局所視野の生配列まで含める（3×3など）  
  → ノイズで `g0/gmin` が振れ、分位校正が不安定になりやすい

#### Step 1: 特徴の候補を増減させる（conquer）
候補特徴を「削る/足す」たびに、DG log のA–Dを観察する。

- `wall` を落とす → “制約”が埋め込みに乗らず、負例（壁）と正例（通路）が近づきやすい  
  → `ranked_candidates` の上位に `meta_passable=false` が混ざる、などの兆候で検出
- `visits` を落とす → “顕著性”が消え、探索圧が表現に乗らない  
  → ループが増え、`delta_sp` 改善が頭打ち、`ag_fire` が過剰/不足に振れる、などで検出
- `dx,dy` を落とす → “操作”が消え、同一位置での4方向が同一視される  
  → `dg_committed_edges` が意味のない結線になりやすい（短絡にならない）
- `success/goal` を落とす → “帰結/目標”が消え、IG（ΔH）が行動結果と結びつきにくい  
  → `linkset_delta_h` が鈍化/不安定になる、などで検出

#### Step 2: 「足すと得する」をゲートで commit する（自律化）
上の ablation を、人手ではなくメタDGで回す：

- **仮説**：特徴追加/削除（または重み `w`・温度 `T`・ビニング規約）の変更案
- **コスト（ΔEPC）**：次元数増加、計算量、記憶量、候補爆発
- **利得（ΔIG）**：A–Dの改善（成功率、探索ノード、P95、ゲート安定性、負例の質）
- **DG（メタ）**：改善がseed/サイズで安定に再現する案のみ採択する

### 7.4（検討中の仮説）ルール獲得：affordance を自己生成負例で学習する

ARCに向けた一般化を重視するなら、「通れる/通れない」「この操作は許されない」といった\textbf{ルール（制約）}を、
あらかじめ手で埋め込むのではなく、\textbf{試行の失敗から自律的に獲得}していく必要がある。

ここではルールを二種類に分ける：

- **環境ルール（遷移制約）**：`(state, action) -> next_state` が成立する条件（迷路の壁/1マス移動/ワープ禁止など）
- **タスクルール（変換制約）**：`(input, transform) -> output` が成立する条件（ARCのプログラム探索での「この操作はこのタスクでは不適」など）

#### Maze（環境ルール）の獲得：壁衝突を “教師信号” にする

現在の迷路PoCでは安全のため「壁には実行しない」設計に寄せているが、仮説としては次を許容する：

- \textbf{わざと壁に行こうとする}（`action` を実行し、移動できない/位置不変を観測する）
- その結果を、\textbf{負例エピソード}として蓄積する（`outcome=fail`, `affordance=blocked`）

このとき、affordance の学習対象は次のように定義できる：

- `A(s, a) ∈ {passable, blocked}`（最小）
- あるいは確率 `P(passable | s, a)`（不確実性つき）

**DG log で見るべき観察（例）**：
- `hit_wall` / `position_unchanged` の発生（負例の発生源）
- `blocked` を学習した後に「無効試行率」が下がる（invalid action rate の低下）
- それでも探索効率（成功率、ステップ、P95）が改善/維持される（負例学習が“足を引っ張っていない”）

#### ARC（タスクルール）の獲得：失敗プログラムを “壁” とみなす

ARCはインタラクティブに試行できないため、迷路の「壁衝突」に相当する負例は次の形になる：

- \textbf{候補プログラム（操作列）が train 例で破綻する}（near-miss を含む）
- これを `blocked` と同型に扱い、`A(task, op, params)=blocked` を学習する

すなわち、迷路では `blocked = 物理的に通れない`、ARCでは `blocked = 記述として成立しない/汎化に不利` を指す。
どちらも「\textbf{次に試すべき候補集合を狭める}」という点で affordance として同型に扱える。

**DG log（ARC版）で見るべき観察（例）**：
- 失敗候補の理由が構造化されている（どの train 例で破綻/どの損失が支配的か）
- 学習後に「無効候補の提案率」が下がる（候補生成の精度が上がる）

#### 重要：ルール獲得は “探索を遅くする” リスクもある

負例を取りに行くほど短期性能が落ちる可能性があるため、導入は段階的にする。

- \textbf{Phase A（安全）}: 明示的な `is_passable`（壁情報）をラベルとして学習（試行しない）
- \textbf{Phase B（学習）}: 壁衝突の試行を少量だけ許容（探索の一定割合で exploration）
- \textbf{Phase C（統合）}: 学習した `A(s,a)` を候補選別とゲート（AG/DG）に戻し、改善が安定すればメタDGで採択

---

## 8. ARCへの写像（ルール/プログラム探索）

ARCでは、Hypothesis は「DSLプログラム（または部分プログラム/マクロ）」になる。

- **Positive**：
  - train の全例で完全一致するプログラム
  - かつ ΔEPC が小さい（短い/単純）ものを優先（同じ train-fit でも“良い”とする）
- **Negative**：
  - 1つのtrain例だけ外す near-miss プログラム（hard negative）
  - train-fit だが極端に冗長なプログラム（overfit）
  - 想定不変量を壊すプログラム（invariance-break）

ここでの「負例生成」は、**候補を作って照合するだけ**なので、テスト正解に依存しない。

---

## 9. `self_organizing_world_model.md` との接続（“意味空間の醸成”）

`docs/research/self_organizing_world_model.md` の要点を、この設計に落とすとこうなる。

- **対比学習**：DG確定（正例）と、DG棄却（負例）で埋め込み空間を彫る
- **グラフ構造パターン認知**：同型発見/近同型/モチーフ抽出で `(anchor, positive, hard negatives)` を作れる（設計メモ: `docs/design/graph_pattern_sleep_semantic_space.md`）
- **不確実性（VAE的）**：エピソードを点ではなく分布（μ, Σ）として持ち、曖昧性を扱う（v2以降）
- **階層（双曲空間）**：抽象ルール（上位）と具体例（下位）の関係を、自然に表現できる可能性（任意）

最初はユークリッド埋め込み + 対比学習だけで十分で、Sleep での改善が回り始めたら拡張する。

---

## 10. 実装に使うライブラリ候補

### 10.1 最小（推奨）
- `numpy`：グリッド/特徴量
- `scipy`：連結成分、距離、軽量最適化
- `networkx`：グラフ表現（設計検証と可視化）
- `pydantic`：ログ/スキーマ（壊れない記録）
- `hnswlib`（または `faiss-cpu`）：類似検索（エピソード/タスク近傍）

### 10.2 学習（任意）
- `torch`：埋め込み学習
- `pytorch-metric-learning`：InfoNCE/Triplet/Hard-mining を実装しやすい（導入するなら）

### 10.3 幾何（任意）
- `geoopt`：双曲空間などのRiemann最適化（必要になってから）

---

## 11. MVP（まず回す最小タスク）

1. 迷路ログから、`(anchor, positive, negatives)` を JSONL に書き出す
2. もっとも単純な埋め込み（例: MLP）を対比学習で学習する
3. Wakeで「近傍エピソード提案」に使う（探索の初期候補を改善）
4. “改善したか” を、Solved率/探索ノード数/P95 で比較する

---

## 12. メモ：古典アプローチから取り入れるべき要素（Episode Memory向け）

この設計は「候補生成→検証→統合」という意味で古典的だが、\textbf{古典の強い部品}を取り込むと一気に“回る”。

### 12.1 MDL/モデル選択（ΔEPC/ΔIG の設計）
- \textbf{EPCは“露骨に”効かせる}：チャンク粒度を細かくしすぎる／特徴を増やしすぎる／マクロを増やしすぎる、はコストとして明示的に罰する
- \textbf{IGは段階評価を許す}：完全一致/完全成功だけでなく、改善量（ループ減、無効試行率減、P95改善など）を利得に含める
- \textbf{ゲートは固定値より分位}：`θ_AG/θ_DG` は分位校正で安定化（seed/サイズ/局所分布の違いに追随）

### 12.2 PBE的：失敗（負例）を資産化する
- \textbf{near-miss（ほぼ良い）を残す}：最も学習価値が高いのは「似ているが破綻した」負例
- \textbf{失敗理由を構造化して保存}：どの条件で破綻したか（例：`hit_wall`, `position_unchanged`, `dead_end`, 特定train例）を教師信号に直結させる

### 12.3 DreamCoder的：Sleepで“語彙”を育てる
- \textbf{成功軌跡からチャンク/マクロを抽出}して探索空間を圧縮
- \textbf{過学習マクロを剪定}（再利用率が低い/コストが高い/汎化しない）
- \textbf{語彙追加もDGで確定}：追加が本当に得か（held-outで安定改善）で採否を決める

---

## 13. 実装案（具体）：迷路→ARCに同型で持ち込むための最小構成

「自律 divide & conquer（エピソード最小単位の自己設計）」を\textbf{手順化}すると、実装は3つの部品に分解できる。

### 13.1 部品A：DG log から学習用データを生成（exporter）

**入力**：
- 迷路：`experiments/maze-query-hub-prototype/results/*steps*.json`（1ステップ＝1レコード）
- ARC：`solve_one --dump-trace` のような探索ログ（将来）

**出力（例：JSONL）**：
- `affordance.jsonl`：`(state_repr, action_repr) -> blocked/passable`
- `triplets.jsonl`：`(anchor, positive, negatives[])`（対比学習）

**迷路のラベル（最小）**：
- `passable`：移動できた（位置が変わった）／または `meta_passable=true`（Phase A）
- `blocked`：`hit_wall=true` または `position_unchanged=true`（Phase B以降）
- `revisit`：移動後のセルが「2回目以降の訪問」（PoCでは `cortisol_reason=revisit`）。壁試行をしなくても “無駄足” の負例が作れる。

**hard negative（迷路）候補**：
- `P(passable|s,a)` が高いのに `blocked` だった（予測の裏切り）
- 似ている（距離が近い）のに、その後 `is_dead_end=true` に繋がりやすい（将来：短いn-stepで近似）
- `revisit` を強く踏む行動（ループ/バックトラック）で、しかも `g0` が高い（AGを開かせ続ける）もの

### 13.2 部品B：affordance model（提案器のprior）

目的は end-to-end ではなく、\textbf{候補生成の事前分布}を作ること。

**最小モデル**（まずこれで十分）：
- ロジスティック回帰 or 小さなMLP
- 入力：`φ_state(s) ⊕ φ_action(a)`（迷路なら `(x,y)` と `(dx,dy)`、局所観測を入れるなら後から）
- 出力：`P(passable|s,a)`（または `P(blocked|s,a)`)

**統合の仕方（安全）**：
- 候補スコアに prior を掛ける：`score' = score * P(passable|s,a)`
- しきい値でフィルタ：`P(passable|s,a) < τ_block` の候補は除外（ただし exploration で少量は残す）

**ログで監査する指標（最低限）**：
- invalid action rate（無効試行率）
- 成功率 / 平均ステップ / P95
- `ag_fire` / `dg_fire` の発火率（分位校正の破綻検知）

### 13.3 部品C：メタDG（分割/特徴/温度/モデル自由度の採否）

“自律化”の本体は、\textbf{設計変更を仮説として扱い、採択をDGで固定}すること。

**仮説の例**：
- 追加特徴：局所観測、n-step要約、成功タグ、goal近接タグ
- 変更：重み `w`、温度 `T`、候補上限 `k_cap`、exploration率
- 変更：チャンク粒度（原子→短いマクロ→長いマクロ）

**コスト（ΔEPC）の見積り**（例）：
- 次元数増加、モデルパラメータ数、候補数増、計算時間増、メモリ増

**利得（ΔIG）の見積り**（例）：
- held-outでの success↑、steps↓、P95↓、invalid rate↓、探索ノード↓

**採択ルール（例）**：
- 変更案を複数seed/サイズで評価し、改善が安定（方向が揃う）なら commit
- 改善が局所/一時的なら保留 or 棄却（hard negativeとして保存）

---

## 14.（検討メモ）Sleep とエピソードベクトル自律化：段階導入の順番・仕様・達成目標

パラメータ爆発を避けるため、Sleep とベクトル自律化は **「自由度の低い更新」から順に**解放する。
この順番は、迷路→ARCへ同型に持ち込むときの“事故率”を下げる目的で採用する。

### Phase 0（Sleep v0）: 校正（calibration）だけで回す

**狙い**: 学習を入れる前に、ゲートと近傍選別の安定性を作る（再現性の土台）。

**仕様（更新対象）**
- `θ_AG/θ_DG` の分位校正（発火率ターゲットで調整）
- 距離の重み `w` と温度 `T`（または `τ_block` など）を小さい探索グリッドで校正
- モデル学習は無し（重み更新無し）

**入力（観察/DG log）**
- `g0/gmin`, `ag_fire/dg_fire`, `theta_ag`（必要なら `theta_dg` 相当）
- `delta_sp`, `dg_committed_edges`
- `linkset_entropy_before/after`, `linkset_delta_h`（IGの飽和/死活監視）

**達成目標（Done）**
- 発火率が安定（例：AG 5–8% 付近、サイズ/seedを跨いで大崩れしない）
- Success率を維持したまま、平均ステップ or P95 が改善（または悪化が統計的に小さい）
- `ΔH`（IG）が飽和していない（常に0/常に最大、にならない）

### Phase 1（Sleep v1）: goal-directed prior（Q値/重み伝播）を学習

**狙い**: 1回目の試行（Wake）で得た「正例/負例」をエッジ重みに保存し、Sleepでゴール価値を伝播（Q-learning的）して、2回目（Wake）では softmax をバイアスして “近道を選びやすくする”。

**仕様（更新対象）**
- 主要な学習対象は `Q(s,a)`（または edge weight `w(s,a)`）：
  - 正例（短い到達/有効な統合）を強化
  - 負例（revisit/破綻/無駄足）を弱化
- Sleep は replay で `Q(s,a)` を更新（Fitted Q-learning / DP）し、「どこへ進むとゴールに近いか」を“重み”に焼き込む
- Wake は softmax の logits をバイアスして使う（ハードoverrideではなく “学習prior” として混ぜる）
  - 例：`logit(a) += β * Q(s,a)`（候補スコアにも同様に掛ける）
  - exploration は残す（完全遮断しない、βスケジュール/温度で調整）

**入力（教師信号）**
- 迷路:
  - ゴール到達（成功）を遅延報酬として使う
  - `revisit` を即時の負例として使う（安全に取れる負例）
  - （必要なら）`blocked`/`dead_end`/ステップ罰も混ぜる
- ARC:
  - train-fit の改善量（誤差減少）を報酬として定義し、破綻/冗長を負例として定義（教師ありを許容するなら supervised も併用可）

**達成目標（Done）**
- “2回目で短くなる” が、複数seed/サイズで安定に再現する（steps↓、revisit率↓、P95↓）
- AGが開きっぱなしにならない（探索が収束する）
- 学習priorを入れてもDGが壊れない（誤統合が増えない、監査可能）

> 実装メモ: まずは “テープ再生” である Sleep最短プラン（BFS）をベースラインとして置き、次に `Q(s,a)` の soft bias（`--sleep-guide prefer` 相当）へ置き換えるのが安全。
>
> - 迷路の `state`（例）: `(x,y)` + visits + （任意で局所観測）
> - `action`: 4近傍（N/S/E/W）
> - `reward`: ゴール到達の報酬に加えて、`revisit` の罰・ステップ罰、必要なら `-F`（=geDIGの“得”）を小さく混ぜる
> - Sleep: `*steps*.json` の遷移ログを replay して Q を更新
> - Wake: 類似度ランキング `score` に `exp(β·Q(s,a))` を掛けて提案器（prior）として使う（softmaxバイアス）

### Phase 2（Sleep v2）: ベクトル自律化（写像/メトリック）を“低自由度”で学習

**狙い**: ベクトル本体を書き換えず、検索幾何だけ育てて安定に改善する。

**仕様（更新対象）**
- `z = normalize(diag(s) * v)` のような “対角スケーリング” から開始（学習パラメータは次元数ぶん）
- 次に `z = normalize(W v)`（低ランク/小次元）へ拡張（必要になってから）
- 制約: L2正規化・ランキング保持率・局所摂動の滑らかさを簡易検定で監視（Φの要件）

**入力（正例/負例）**
- 正例: DGで効いた候補（短絡/成功に寄与したエッジやエピソード）
- 負例: near-miss（似ているが破綻）、blocked 予測の裏切り、dead-endに寄る候補

**達成目標（Done）**
- 近傍検索の質が上がる（Top-kの有効率、近傍からの改善率が増える）
- `g0/gmin` の分位校正が壊れない（発火率が暴れない）
- P95/探索ノード数が改善し、改善がseed/サイズで安定

### Phase 3（Sleep v3）: メタDGで“設計変更”を採択する（自己設計の本丸）

**狙い**: 「特徴を足す/削る」「粒度を変える」「学習自由度を上げる」を仮説として扱い、採否を自動化する。

**仕様（更新対象の例）**
- 特徴: 局所観測、n-step要約、成功タグ、goal近接タグ
- 運用: `k_cap`, exploration率、候補生成方針、分位目標
- 粒度: 原子→短いマクロ→長いマクロ（語彙増やす/剪定する）

**採択ルール（例）**
- “改善が一貫して出る”場合のみ commit（複数seed/サイズで符号が揃う）
- 一時的/局所的な改善は棄却（hard negativeとして保存）

**達成目標（Done）**
- 人手のチューニング量が減っても性能が落ちない（再現性の維持）
- 改善が「ログで説明できる」（どの仮説が、どの指標をどれだけ改善したか）

---

## 15.（検討メモ）FEP（ELBO / KL）をこの設計に入れるならどこか

FEPの見方（ELBO/KL）を入れること自体は有望だが、\textbf{入れすぎると} geDIG の強み（分離性・解釈性・運用の安定）が崩れやすい。
したがって、まずは **Sleep（オフライン更新）側の目的関数として限定導入**し、Wake（オンライン意思決定）の `F` には直接混ぜないのが安全。

### 15.1 何に対応づけるか（最小の対応）

- **Wake（意思決定）**: `F = ΔEPC − λ·ΔIG` と AG/DG（現状維持）
  - 0-hop / multi-hop は「探索を開く/統合する」の制御信号であり、\textbf{オンラインの可観測性}を優先する
- **Sleep（表現更新）**: 「埋め込み/affordance/提案器」を更新する目的関数として ELBO/KL を使う

### 15.2 入れどころ（おすすめ順）

1) **Sleep v1（affordance prior）を“FEP的”に解釈**  
   `P(passable|s,a)` の学習は、観測（成功/失敗）に対する NLL（= surprise）の最小化なので、
   KL/クロスエントロピーの枠で整合的に説明できる。実装はロジスティック回帰/小MLPで十分。

2) **Sleep v2（ベクトル自律化）に KL を“暴走止め”として導入**  
   いきなりVAEにせず、まずは更新の自由度を小さく保つ：
   - `z = normalize(diag(s) * v)`（低自由度）を学習しつつ、
   - `KL(q(z|e) || p(z))`（例: 標準正規）や、前回表現へのドリフト罰（`||z - z_prev||`）で安定化する

3) **VAE（μ,Σ）化は “後から”**  
   hard negative が十分に溜まり、表現更新が安定して効くことが確認できた段で
   `q(z|e)=N(μ,Σ)` を導入し、\textbf{曖昧性をσで表現}する（破綻しやすいエピソードほどσを大きくする、等）。

### 15.3 「入れすぎ」を避けるガード（重要）

- **二重計上の回避**：ELBO/KL を `F` に直接入れると、構造（EPC/SP）と表現（ELBO）の効果が混ざりやすい  
  → \textbf{まずはSleepの更新則に閉じ込める}
- **分位校正を壊さない**：表現更新で `g0/gmin` 分布が崩れると AG/DG が破綻する  
  → Phase 2（v2）では「発火率の安定」を必須KPIにする
- **メタDGで採択**：ELBO/KL 導入自体も仮説として扱い、held-outで改善が安定した場合のみ採択する

### 15.4（最小の実装スケッチ）Sleep目的関数の例

```
L_sleep = L_task_or_contrastive
       + β * KL(q(z|e) || p(z))
       + α * drift(z, z_prev)
```

- `L_task_or_contrastive`: 正例/負例（near-miss, blocked）から作る損失（InfoNCE/Triplet など）
- `KL`: 表現の正則化（暴走防止、一般化圧）
- `drift`: 直前の表現からの変化量を抑える（運用安定性）

> 方針: “入れる”のはOK。ただし「WakeのFに混ぜない」「低自由度から」「メタDGで採択」の3点で過剰統合を避ける。

---

## 16. 現在の実装状況（迷路PoC, 2026-01-29）

このドキュメント上の「仕様案」に対して、現時点で動いているもの／まだ入れていないものを明示する。

### 16.1 実装済み（Maze Query-Hub）

- 実装ファイル: `experiments/maze-query-hub-prototype/run_experiment_query.py`
- Wake→Sleep→Wake（2回で短くする）:
  - `--curriculum-warmup-steps N` で warmup→Sleep→eval を実行
  - Sleep は warmup の経験遷移だけから BFS で最短プラン（1-step plan）を作る（経験内では最短が保証される）
  - Sleep は warmup の遷移ログから replay で `Q(s,a)` を学習（価値伝播 / Q-learning）
	  - eval は `--sleep-guide override|prefer|off` で適用:
	    - `override`: BFSプランをそのまま実行（テープ再生）
	    - `prefer`: `Q(s,a)` を softmax/argmax の prior として“バイアス”に使う（ハードoverrideしない）
	      - 追加で `--sleep-plan-beta` を入れると、BFSプランの「次アクション」にも soft なボーナスを与えられる（still prefer）
	    - `off`: Sleep由来の誘導なし
	  - ログ: `episode_phase`, `sleep_plan_action`, `sleep_plan_beta`, `sleep_guided`, `sleep_q_applied`, `sleep_q_*`
- 負例（revisit）ラベル:
  - `--cortisol-mode log --cortisol-repeat-visits 2` で「2回目以降に踏んだマスへの移動」を負例としてログ化
  - ログ: `cortisol_fire`, `cortisol_reason=revisit`

### 16.2 動作確認の例（手元ログ）

- **override（BFSプランのテープ再生）**:
  - 15×15（seed=0）: warmup 36 step → Sleep plan 28 → eval 28 step（warmup revisit=4, eval revisit=0）
  - 25×25（seed=0）: warmup 276 step → Sleep plan 124 → eval 124 step（warmup revisit=76, eval revisit=0）
- **prefer（Sleep Qのsoft bias）**:
  - 25×25（seed=0, `--use-main-l3 --max-hops 3 --theta-ag 0.2 --theta-dg 0.15`）: warmup 276 step → eval 144 step（eval revisit=10, `dg_fire`=121/144, `sleep_guided`=124/144）
  - 25×25（seeds=10, `--max-steps 200 --curriculum-warmup-steps 800 --sleep-q-beta 8 --sleep-plan-beta 2 --max-hops 2`）: 成功率 1.0 / avg_steps 124.4（`results/maze-local/sleepq_prefer_25_n10_qb8_pb2_h2_w800.json`）
  - 成功率を上げるチューニング用: `experiments/maze-query-hub-prototype/tools/run_sleepq_prefer_grid.py`（`sleep_q_beta` / `sleep_plan_beta` のスイープ）

### 16.3 未実装（次の差分）

- `Q(s,a)` を（row,col）の表から、より一般の「状態表現（エピソードベクトル）」へ拡張（迷路以外の空間に移植）
- 負例生成の自律化（revisit以外の “near-miss/blocked/誤操作” をどう作るか）
- Sleep v2/v3（ベクトル自律化・メタDG採択）に向けた目的関数と安全柵（KL/ドリフト等）の具体設計

---

## References

- `docs/design/arc_prize_spec.md`
- `docs/research/self_organizing_world_model.md`
- `src/insightspike/algorithms/gating.py`
