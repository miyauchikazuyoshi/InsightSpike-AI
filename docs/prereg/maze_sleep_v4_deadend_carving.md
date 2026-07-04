# 事前登録: 迷路 sleep v4 — deadend 彫り込み(dim8/dim9 価値統一)による消去法ナビゲーション

> **日付**: 2026-07-04(実験実施前にコミット)
> **状態**: DRAFT — 著者承認で FROZEN に変更し、以後は追記のみ
> **位置づけ**: [v3](maze_sleep_ablation_v3.md) §8 が指示した「引き上げ/失敗層の再設計 —
> (ii) 質を主検定に」の実行。設計討議は
> [edge_flow_field_navigation_20260704.md](../research/thinking/edge_flow_field_navigation_20260704.md)
> (消去法ナビ: Pull/Push/Eliminate の分解、readout 等価性の発見)。
> v1 §3.2 の試行予算とは独立の新ライン(彫り込み仮説)。v2/v3 で確立した成功層の効果は本実験の前提であり検証対象ではない。

## 0. 探索の開示

本 prereg に先立つ探索(2026-07-04、`results/graph_persistent_dg/_exploratory_flow/` NOTES.md):
- **seed=52**(v3 で replay 256 歩が off 108 歩に負けた診断ケース)で 2×2(abs/gradient × ペナルティ有無):
  - deadend/blocked −1.0 で **256→104 歩(off 108 を下回る)**、袋小路 6→2
  - readout 形式(abs/gradient)は歩数まで完全同一 — 候補比較では
    Q(here,a)=r+γV(next) と flow=V(next)−V(here) が V(next) の単調関数として argmax 一致(数学的必然)
- 帰結: **操作は 1 ノブに簡約**(gradient は登録しない)。**seed=52 と seed=4(v3 探索)は確証から除外**
  (60–89 は両探索と無縁の未使用シード)。
- v3 探索(seed=4)では deadend −1.0 の追加効果はマージナルだった(448→440)。seed=52 では決定的(256→104)。
  **効果の条件依存性(彫り込みが効く迷路と効かない迷路がある)こそが、n=1 の探索を確証に昇格させる理由**である。

## 1. 主張(検証対象)

> replay Q への deadend/blocked ペナルティの彫り込み(wake 記録 dim8 と同じ −1.0 に統一)は、
> warmup 失敗層における第 2 エピソード探索の質(歩数・袋小路遭遇)を改善する。
> 機構: 袋小路の負が γ=0.99 の軌跡 backup で枝の入口まで遡り、交差点で「死んだ枝」が
> 値として読める(Eliminate の彫り込み)。ゴール座標は使わない(消去法派、
> potential shaping は導入しない)。

### 1.1 スコープ限定

- 操作は**報酬値の統一のみ**(deadend/blocked: 0.0 → −1.0 = dim8 の wake 記録値)。
  これはパラメータ「調整」ではなく、[設計監査](../audits/sleep_ablation_design_audit.md)以来
  懸案だった **dim8/dim9 価値体系乖離の解消(整合性原則)**である。
- revisit の乖離(dim8 −0.4 vs Q −0.2)は**今回触らない**(未探索の残余として §6 に記録)。
- readout は現行の abs のまま(§0 の等価性により gradient は不要)。
- blocked ペナルティは自己ループ希釈+観測ガードにより実質不活性と判明済み
  (実効ノブは deadend)— 統一原則としてセットで変更するが、効果の帰属は deadend に置く。

## 2. 理論の事前予測

- **P1(Primary)**: warmup 失敗層の eval 歩数(打ち切り failure→500): carving < current(paired)
- **P2**: 同層の eval 袋小路遭遇数: carving < current
- **P3(操作チェック)**: carving 腕の `sleep_q` メタ q_min ≤ −0.9(彫り込みが Q に入った証拠)、
  current 腕の q_min > −0.9、両腕 warmup paired 一致、伝播ノード両腕 > 0
- **P4(退行チェック)**: warmup 成功層の eval 歩数で carving が悪化しない
  (paired 差 carving−current の CI95 上限 < +20 歩。悪化が確定した場合、v4 採用は見送り、
  「失敗層特化の彫り込み(条件付き有効化)」を次の設計課題として記録する)

## 3. 反証条件

| 条件 | 帰結 |
|------|------|
| P1 不成立(CI95 が 0 を含む or 逆方向) | **彫り込み仮説の敗北記録**。seed=52 の −59% はシード特異と記録し、消去法ナビ路線は「彫り込みでは不足 — 探索構造の変更(warmup 予算・複数エピソード)が必要」に更新 |
| P1 成立・P2 不成立 | 質改善は袋小路回避以外の経路と記録(機構解釈保留) |
| P4 逆方向が確定(成功層で有意悪化) | v4 の全面採用を見送り。「彫り込みは失敗層限定で有効」として条件付き設計へ |
| P3 不成立 | 装置故障。結果無効、修理後再実行 |

事後の矛盾吸収はしない。結果が何であれ §8 に追記する。

## 4. 実験設計

### 条件(操作は 1 変数)

| | carving | current |
|---|---|---|
| `--sleep-q-deadend-penalty` | **−1.0** | 0.0(既定) |
| `--sleep-q-blocked-penalty` | **−1.0** | 0.0(既定) |
| それ以外 | v2/v3 と同一(replay + guide off + v6_perseed COMMON_ARGS + extended、abs readout) | 同左 |

- 対象: **シード 60–89(全実験ライン未使用)を全数実行**し、warmup 成否で層別(決定的・両腕共通)
- 失敗層が主戦場(P1/P2)、成功層は P4(退行チェック)
- 標本拡張規則: 失敗層が 5 未満の場合のみシード 90–119 を 1 回だけ追加(v3 と同じ optional stopping 封じ)
- 実行: `bash run_sleep_v4_carving.sh`(per-seed、carving→current の順で paired)

### 統計(変更禁止)

1. **Primary**: 失敗層 eval 歩数の paired 差(打ち切り failure→500)。Wilcoxon(両側 α=0.05)+
   paired bootstrap CI95(シード 20260705)
2. P2: 同層 `dead_end_steps` に同手法。P4: 成功層 eval 歩数に同手法(判定は §2 の CI 基準)
3. 主検定は P1 の 1 つ。P2/P4 は副次と明記
4. **解析コード**: `experiments/maze/analyze_sleep_v4_carving.py` を実験実行前にコミット

## 5. 主要指標(変更禁止)

1. **Primary**: 失敗層 eval 歩数の carving − current paired 差(CI95・p 値)
2. Secondary: 失敗層袋小路差、成功層退行チェック、q_min 操作チェック、eval 成功率(記述)

## 6. 既知のリスク

- **効果の条件依存性**: seed=4 ではマージナル、seed=52 では決定的。失敗層の構成次第で平均効果が
  希釈される可能性(それでも paired 設計で方向は検出可能)
- **成功層への副作用**(P4 で監視): goal 到達エピソードにも deadend 経験は含まれ、彫り込みが
  正の勾配地形を歪める可能性。探索(seed=52 は失敗層)ではこの面は未検証
- **revisit 乖離は未解消のまま**(dim8 −0.4 vs Q −0.2)— 本実験の範囲外、§1.1
- **パイロット省略の宣言**: 実行時間(失敗層 ~27–35 分/本、成功層 ~2–10 分/本)と JSON 健全性は
  v3 本実験 60 ラン+探索 7 ランで既知のため、追加パイロットは行わない。
  60–89 の失敗層期待数は v3 実測(12/30)から 9–15。総実行時間の見積もりは **10–14 時間**

## 7. 凍結手続き

1. 状態を FROZEN に変更してコミット(**著者の承認事項**)
2. 実験実行: `bash run_sleep_v4_carving.sh` → 解析: `analyze_sleep_v4_carving.py` → 結果が何であれ §8 に追記

---

## 8. 結果(実行後に追記 — 実行前は空欄のこと)

(未実行)
