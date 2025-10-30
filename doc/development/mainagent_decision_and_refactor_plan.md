# MainAgent の判断責務とリファクタリング計画（Maze vs Main の比較）

目的: 迷路実験における意思決定（判断）フローを基準に、MainAgent が担うべき判断責務を明確化し、レイヤ内に残っている判断を吸い上げる。RAG 実験へ拡張しても迷路実験が破綻しないよう、段階的なリファクタリング計画を定義する。

---

## 現状の比較（Maze シークエンス vs MainAgent）

- 迷路実験（runner 側の挙動・判断例）
  - hop 系列の評価（union-of-k-hop）と best_h の採用
  - NA/DA ゲートに応じた拡張/縮退の分岐
  - AG 発火後の候補拡張（リング/近傍/Top‑K/θ の調整）
  - 予算・遅延（P95/P99）の管理・早期停止
  - HTML/ログの構造化（step/hop series/cand 等）

- MainAgent（メインコード）に不足/分散している判断
  - hop 系列の制御（use_main_l3=hop0 相当になりがち）
  - NA/DA の二段ゲート・エスカレーション（設計と実装の集約）
  - 候補拡張のオーケストレーション（AG→L1 拡張の司令）
  - 予算/PSZ/閾値（λ/γ 含む）を一箇所で統制・伝播
  - 入力契約の監査（candidate_edges 未提供時のダウンシフト/警告）
  - 再現性メタ（h_scope, sp_scope, input_contract_ok, autocand_used, best_h など）

---

## 判断責務（MainAgent に一元化する項目）

1) NA/DA 判定と hop エスカレーション
- NA 高→ hop を広げる（もしくは候補拡張）
- DA 高→ 縮退/停止/候補の絞り込み
- 判定・基準・cooldown を MainAgent 内で集約

2) hop 系列の採用と反映
- L3 の union 系列（g(h)）から best_h を採用
- 次サイクルの評価/候補生成に best_h を反映（L1 へのヒント）

3) AG 発火後の候補拡張
- MainAgent が L1 に ExpansionRequest を発行
- L1 は Ecand（必要に応じ by-hop）を再構築
- 次サイクルで新 Ecand を消費

4) 予算/PSZ/閾値の統制
- P95/P99・PSZ・λ/γ・prune/window/sources の標準プロファイル化
- MainAgent が設定→各レイヤへ伝播（ENV ではなく config/context 経由）

5) 契約監査とフォールバック
- `sp_engine=cached_incr` で `candidate_edges` 無し → cached へダウンシフト
- メトリクスに `input_contract_ok=false`, `autocand_used=false` を記録

6) ログ/再現性の責任点
- decisions（採否/拡張/停止/理由）を MainAgent で一元記録
- h_scope='cand_mask', sp_scope, best_h, hop_series（必要時）

---

## リファクタリング計画（段階的）

Phase 1（責務の固定化・不整合解消）
- L3：自動候補 OFF 既定、契約違反時は cached へダウンシフト
- ΔH：cand_mask + WEIGHT_VECTOR + norm='before' に統一（A/B と一致）
- metrics：`h_scope`, `input_contract_ok`, `autocand_used`, `best_h` を標準化

Phase 2（MainAgent の判断導入）
- DecisionController（MainAgent 内）を追加
  - 入力：L3 metrics
  - 出力：Action（maintain/expand_hop/expand_cands/stop）
  - Policy：NA/DA, hop エスカレーション, 予算/PSZ, cooldown
- L1 ExpansionRequest（AG 発火後）
  - `centers`, `prev_Ecand`, `budget`, `radius_hint`, `theta_link_hint` を渡し、Ecand（必要なら by-hop）を再構築

Phase 3（by-hop 候補と union 系列の一体化）
- L1：`candidate_edges_by_hop` を生成可能に
- L3：by-hop 候補を優先消費（無ければ単一 Ecand）
- MainAgent：best_h と by-hop 候補選定のフィードバックループを完成

Phase 4（RAG への展開とガードレール）
- 迷路/実験への影響を避けるため、研究用オプションは `experimental.*` に隔離
- ENV 切替は廃止・最小化し、config/context のみで制御
- 迷路/共通のリグレッションテストを CI に追加（ΔH/IG の sign/傾き、best_h の存在）

---

## インタフェース草案

- DecisionController（MainAgent 内）
```python
class DecisionController:
    def decide(self, metrics: dict, state: dict) -> dict:
        """Return action: {mode: 'maintain|expand_hop|expand_cands|stop', params:{...}}"""
```

- ExpansionRequest（MainAgent → L1）
```python
@dataclass
class ExpansionRequest:
    centers: list[int]
    prev_ecand: list[tuple[int,int,dict]] | None
    budget: int
    radius_hint: float | None
    theta_link_hint: float | None
    by_hop: bool = False
```

- L1 出力（context 拡張）
```python
context = {
  'centers': [...],
  'candidate_edges': [...],
  'candidate_edges_by_hop': {0:[...],1:[...]},  # optional
  'norm_spec': {...},
}
```

---

## テスト計画（要約）
- 単体：自動候補 OFF/ダウンシフト、ΔH=cand_mask の一致性、by-hop 候補優先、benefit 表記の非存在、符号固定
- パイプライン：A/B vs L3 の ΔH 一致、union 系列で best_h>0、AG→拡張→次サイクルで候補/IG 増加
- 受け渡し：DecisionController の Action と ExpansionRequest の往復を検証

---

## RAG 実験への配慮（壊さないために）
- 研究用の経路/ノブは `experimental.*` に隔離し、既定無効
- メイン経路は責務分離・契約監査・再現性メタを厳守
- 迷路/共通のリグレッションテストを CI に追加（ΔH/IG の sign/傾き、best_h の存在）

---

## 受け入れ基準
- ΔH：A/B と L3 で sign 一致（cand_mask + 重み + after‑before）
- ダウンシフト：契約違反で必ず発動、メタ記録
- 判断：DecisionController が hop/候補の拡張・停止を司る
- RAG：迷路の判断ロジックを壊さず拡張可能

