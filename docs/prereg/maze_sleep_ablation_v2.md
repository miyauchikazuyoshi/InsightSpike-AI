# 事前登録: 迷路 sleep 再設計バリアント#1 — replay 伝播 × 自力ナビ（guide off）

> **日付**: 2026-07-02（実験実施前にコミット）
> **状態**: DRAFT — パイロット（seed=0）の結果を §6 に追記後、著者承認で FROZEN に変更し、以後は追記のみ
> **位置づけ**: [v1 事前登録](maze_sleep_ablation.md) §3.2 の**再設計試行予算（N=2）の 1 回目**。
> v1 の敗北記録（P1 FAIL: 現行の無向 max 伝播はデフォルト構成で行動への寄与ゼロ）と
> [設計監査](../audits/sleep_ablation_design_audit.md)（4 ギャップ R1–R4）を受けた再設計。
> **実験装置**: `--sleep-propagate replay` を 2026-07-02 に実装 — 伝播を**軌跡ベース Q backup の転写**に置換
> （`qhlib.sleep.build_sleep_q_table` の Q(s,a) を direction node の `propagated` に、V(s)=max Q を
> query node に書き込む。有向・有界・負例保存。転写契約は `test/test_sleep_propagate_semantics.py`
> の TestReplayVariant で固定済み）。

## 1. 主張（検証対象）

> 軌跡ベース Q backup による伝播値（replay）は、**行動選択が伝播 readout に依存する条件**
> （`--sleep-guide off`、辞書誘導なしの自力ナビゲーション）下で、warmup 成功シードの
> 第 2 エピソード（Wake2）探索を改善する。

### 1.1 設計監査ギャップ R1–R4 の充足

| # | ギャップ（v1 で欠けていたもの） | 本実験での充足 |
|---|---|---|
| R1 | 辞書誘導が行動を支配 | **両群 `--sleep-guide off`** — eval は類似度＋伝播 readout の自力ナビ |
| R2 | eval が既に理論最短（改善余地ゼロ） | R1 の帰結として自力ナビ化 → 最短一致が崩れ余地が生まれる（パイロットで確認） |
| R3 | warmup 失敗時に勾配源（goal 報酬）が不在 | **対象を warmup 成功シードに限定**（v1 実験で warmup 成功が確定している 23 シード。warmup は誘導・伝播と無関係な初回エピソードなので、この層別は決定的であり事後選択ではない） |
| R4 | 伝播値の飽和＋max 演算による負例消滅 | **replay = 有向 episodic backup**。Q は goal_reward=1.0 スケールで有界（tanh 非飽和）、負例（revisit/step ペナルティ）は軌跡に沿って保存される（単体テストで固定） |

### 1.2 スコープ限定（v1 §1.2 の Phase 論を継承）

- 本実験も「**F はまだ活きていない**」レベル — 環境定義報酬の（今度は正しく設計された）伝播値が
  行動に寄与するかの検証である。F 駆動 sleep は依然として範囲外（試行予算の残り 1 回の本命候補）。
- replay の Q パラメータは辞書版 sleep_q の既定値（gamma=0.99, alpha=0.4, iters=50,
  goal_reward=1.0, revisit_penalty=−0.2, step_penalty=−0.01）をそのまま使う。
  パラメータ探索は行わない（辞書版はこの値で eval を最短化できた実績があり、Q 値に十分な情報が
  あることは既知 — 本実験が問うのは「その情報を**ベクトル readout 経由で**行動に変換できるか」）。

## 2. 理論の事前予測

- **P1（Primary）**: Wake2（eval）ステップ数: replay < off（paired、23 シード、打ち切り失敗→500）
- **P2**: Wake2 の袋小路遭遇数（`dead_end_steps`）: replay < off — warmup で経験した袋小路を
  伝播値の負勾配で**踏む前に**回避する、という著者の設計意図の最直接指標
- **P3（操作チェック）**: replay 群は伝播ノード > 0 かつ off 群 = 0、全対象シードの warmup 成功、
  warmup paired 差 = 厳密に 0

## 3. 反証条件

| 条件 | 帰結 |
|------|------|
| P1 不成立 | **試行予算 1/2 を消費**。「正しく設計された伝播値でも、ベクトル readout（dim9 + α バイアス）経由では行動を改善できない」と記録。residual 容疑者は readout 経路そのもの（設計監査 R1 の残り: propagated_alpha の適用形が exp 乗算/加算混在）に移る。残り 1 回は F 駆動 sleep または readout 診断のどちらかに使う（着手前に事前登録） |
| P1 成立・P2 不成立 | 改善は袋小路回避以外の機構（経路プライア等）によると記録し、機構の解釈を保留 |
| 両群とも成功率が崩壊（23 シード中成功 12 未満が両群で発生） | 自力ナビが弱すぎて課題が変質したと記録（装置の限界）。ただし判定自体は打ち切り済み paired 比較で実施可能なため P1 判定は行う |
| P3 不成立 | 装置故障。結果無効、修理後再実行 |

v1 と同じ禁止事項を継承する: 事後の矛盾吸収をしない。予測が外れたら本ファイルに敗北記録を追記する。
off 群（生グラフ継承 + 自力ナビ）が replay 群に勝った場合も、その事実をそのまま記録する。

## 4. 実験設計

### 条件（操作は 1 変数のみ）

| | replay | off |
|---|---|---|
| `--sleep-propagate` | `replay` | `off` |
| `--sleep-guide` | `off`（共通） | `off`（共通） |
| それ以外 | v1 と同一（v6_perseed COMMON_ARGS + extended） | 同左 |

- 対象シード（23、v1 で warmup 成功が確定）: 0,1,2,3,5,6,7,8,10,11,12,13,14,15,17,18,19,20,21,24,25,26,29
- 実行: `bash run_sleep_ablation_v2.sh`（per-seed、replay→off の順で paired 実行）
- **v1 との差分は 2 点のみ**: 伝播アルゴリズム（max→replay）と誘導（override→off）。
  この 2 点は R1–R4 を埋めるために**同時に**必要（伝播だけ直しても readout の幕がなく、
  誘導だけ切っても飽和値では識別できない）。したがって本実験の対照は「replay あり/なし」であり、
  「v1 の on」との三つ巴比較は行わない（v1 の on は寄与ゼロが確定済みなので off と等価）。

### 統計（変更禁止）

v1 と同一の枠組み: P1 = Wilcoxon signed-rank（両側 α=0.05）+ paired bootstrap CI95
（シード 20260703）、打ち切り failure→500、感度分析 = 両成功ペアのみ。
P2 = 同じ検定を `dead_end_steps` に適用。主検定は P1 の 1 つ、P2 は副次と明記。
**解析コード**: `experiments/maze/analyze_sleep_ablation_v2.py` を実験実行前にコミット
（v1 の解析スクリプトは v1 凍結の一部なので不変のまま残す）。

## 5. 主要指標（変更禁止）

1. **Primary**: eval ステップ数の replay − off paired 差（23 ペア、打ち切りあり）
2. Secondary: eval `dead_end_steps` の paired 差（P2）、成功率（記述統計 — 自力ナビの床/天井の文脈）、
   warmup 指標の無差確認（操作チェック）

## 6. 既知のリスク

- **自力ナビの成功率低下**: guide off での eval は未知の挙動。両群とも失敗が多発すると打ち切りが支配し
  検出力が落ちる（§3 の崩壊条項）。パイロット（seed=0、replay/off 両方）で確認し、ここに追記する。
- **readout 経路の質**: 伝播値が正しくても、readout（dim9 類似度 + propagated_alpha の exp/加算混在）が
  雑音的なら効果は出ない。P1 不成立時はここが residual 容疑者になる（§3）。
- **層別の限定**: warmup 成功 23 シードのみを対象とするため、「warmup 未達シードの引き上げ」
  （著者の設計意図のもう半分）は本実験では検証しない。それには R3 の別解
  （warmup を到達直後打ち切りにする等の予算変更）が必要で、範囲外と明記する。

### パイロット実測（FROZEN 前に追記）

（パイロット seed=0 実行中 — 完了後に追記）

## 7. 凍結手続き

1. パイロット seed=0（replay/off、guide off）で実行時間・自力ナビ成功可否・JSON 健全性を確認 → §6 に追記
2. 状態を FROZEN に変更してコミット（**著者の承認事項**）
3. 実験実行: `bash run_sleep_ablation_v2.sh` → 解析: `analyze_sleep_ablation_v2.py` → 結果が何であれ §8 に追記

---

## 8. 結果（実行後に追記 — 実行前は空欄のこと）

（未実行）
