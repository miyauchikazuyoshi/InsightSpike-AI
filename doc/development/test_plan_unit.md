# ユニットテスト計画（L1/L2/L3 + ΔH候補マスク + 自動候補OFF）

目的: 仕様の中核（候補所有= L1/L2、ΔH=cand_mask、L3 自動候補 OFF、符号固定 after−before）をユニットテストで担保する。

---

## テスト対象と観点

1) L3 自動候補 OFF/ダウンシフト
- 条件: `sp_engine=cached_incr`, `candidate_edges=None`
- 期待: `metrics.sp_engine='cached'`, `metrics.input_contract_ok=False`, `metrics.autocand_used=False`
- 逆条件（研究用フラグON）: `graph.allow_autocand=True` のみ `autocand_used=True`

2) ΔH=cand_mask（候補マスク母集団）
- 小規模グラフで、A/B の ΔH と L3 の ΔH を比較
  - 重み=ON, norm='before', delta='after_before'
  - 候補外ノードの特徴=0化
  - 期待: sign 一致、比率が許容範囲（±ε）

3) 重みの適用確認（8D迷路）
- WEIGHT_VECTOR を適用しない場合との差が出ること（既定は適用）
- 期待: `metrics.debug.h_weighted=True`（あれば）/ ΔH 値の差

4) by-hop 候補優先
- `candidate_edges_by_hop={0:[...],1:[...]}` を渡した場合
  - 期待: hop ごとに異なる候補が消費され、ΔH/ΔSP が hop で差分
- 未指定時: 単一 `candidate_edges` が全 hop で共通使用

5) 旧benefit表記の不在
- `h_benefit`/`ig_benefit` が metrics に存在しない

6) ENV/Config による ΔH 符号切替の不在
- 以前の `MAZE_GEDIG_IG_DELTA` 系が無効であること
- 期待: 常に `after_before`（固定）

---

## ファイル案とサンプル

- `tests/unit/test_l3_autocand_off.py`
- `tests/unit/test_l3_delta_h_cand_mask.py`
- `tests/unit/test_l3_byhop_candidates.py`
- `tests/unit/test_no_benefit_fields.py`
- `tests/unit/test_no_delta_sign_knob.py`

各テストは最小合成グラフ（networkx）または軽量 Data スタブ（numpy 行列）で構築し、`L3GraphReasoner.analyze_documents(..., context=...)` を直接呼ぶ。

---

## 合格基準と閾値

- sign 一致は厳密一致
- 比率比較は |ΔH_L3 − ΔH_AB| / max(1e-6, |ΔH_AB|) ≤ 0.1（例）
- 自動候補 OFF/ダウンシフトは 100% 再現
- 不在確認はキー存在判定で厳格

