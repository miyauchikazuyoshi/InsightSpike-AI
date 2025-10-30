# パイプラインテスト計画（L1→L2→L3、AG拡張）

目的: L1 が指揮者として centers/Ecand（必要なら by-hop）/norm_spec を構築し、L3 が消費して ΔGED/ΔH/ΔSP/IG を一貫した仕様で出力することを E2E で確認する。AG 発火後の候補拡張フローも合わせて検証する。

---

## シナリオA: A/B vs L3 一致性（ΔH 指標）

- 準備: 小規模合成グラフ（迷路8D重みを適用）
- L1: `centers`, `candidate_edges` を生成（by-hop 未指定）
- L3: `sp_engine=cached_incr` で評価（ΔH は cand_mask、norm='before'）
- 比較: A/B の ΔH（entropy_ig after−before）と L3 の ΔH（cand_mask）が sign/傾きで一致

## シナリオB: union-of-k-hop（by-hop 候補）

- L1: `candidate_edges_by_hop={0:[...],1:[...],2:[...]}` を生成
- L3: union 評価で hop ごとに by-hop 候補を消費
- 期待: g(h) がフラットではなく、hop 間で差（best_h が 0 以外になり得る）

## シナリオC: 自動候補 OFF / ダウンシフト

- L1: `candidate_edges=None`
- L3: `sp_engine=cached_incr`
- 期待: `cached` にダウンシフト、`metrics.input_contract_ok=False`、`autocand_used=False`

## シナリオD: AG 発火 → 候補拡張（L1 主導）

- L3: AG を検出し、`ExpansionRequest{ centers, prev_Ecand, budget, hints }` を L1 に発行
- L1: 拡張 Ecand（必要なら by-hop）を構築して返却
- 次サイクルで L3 が拡張 Ecand を消費し、候補数/IG が増える（時系列で確認）

---

## 実装・検証メモ

- L1/L2 はテスト用スタブを用意（索引/リング検索はシンプル実装）
- L3 のメトリクスに `h_scope='cand_mask'`, `input_contract_ok`, `autocand_used` を記録し、アサート
- 再現性のため乱数種と `WEIGHT_VECTOR` を固定
- HTML レポート（任意）を補助出力とし、数値基準（JSON）で判定

---

## 合格基準

- シナリオA/B: ΔH の sign 一致、best_h が 0 以外のケースを最低1つ作る
- シナリオC: ダウンシフトが常に発動し、メタが記録される
- シナリオD: 拡張前→後で候補数/IG が単調増加（少なくとも1回）

