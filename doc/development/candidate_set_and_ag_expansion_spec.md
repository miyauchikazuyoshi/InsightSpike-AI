# Candidate Set Ownership and AG-Triggered Expansion (L1 vs L3)

目的: 候補集団（Ecand）の選定責務を Layer1/L2 に明確化し、Layer3 は「消費のみ」に徹する。さらに、AG（Acceptance Gate）発火時に候補集団を広げるべきか（誰が、どのように）を仕様化する。

---

## 背景と課題

- 現状の L3 は `candidate_edges` が未提供のとき、自前で `graph.x` と `centers` から Top‑K/θ_link により候補を自動生成（autocand）するフォールバックがある。
  - これが L1/L2 本来の責務（Ecand の決定）と衝突し、母集団の不一致を招く（IG/H の挙動差）。
- ΔH（entropy_ig）についても、A/B と L3 で「候補マスク vs 全体/サブグラフ」の差があり、系列がフラットに見える原因となる。

---

## 仕様（責務分離）

### 所有（Ownership）

- L1/L2 の責務
  - centers の決定（Top retrieved など）
  - 候補集団 Ecand の決定（候補ノード/リンクの母集団）
  - NormSpec の決定（metric, 半径/しきい値, effective 値）
  - これらを L3 へ context として明示的に引き渡す。

- L3 の責務
  - 受け取った `centers`/`candidate_edges`/`norm_spec` を用いて ΔGED/ΔH/ΔSP を統合評価し、ゲーティング（AG/DG）を行う。
  - 候補の自動生成（autocand）はデフォルト禁止。研究用途でのみ明示フラグにより有効化可能（初期値は OFF）。

### インタフェース（context）

```python
context = {
  "centers": [int],                          # L1/L2 提供（必須）
  "candidate_edges": [ (u, v, {"score": s}) ],  # L1/L2 提供（sp_engine=cached_incr で必須）
  "candidate_selection": { ... },            # 任意（ログ/根拠）
  "norm_spec": { ... },                      # L1/L2 提供（effective 可）
}
```

### ΔH の母集団（統一）

- A/B と L3 で一致させるため、ΔH は「候補マスク母集団」で計算する。
  - 対象ノード集合: `centers ∪ endpoints(candidate_edges)`（候補未提供時は L3 側で ΔH 計算をスキップ or 全体にするが、autocand は行わない）
  - 候補以外は特徴ベクトルを 0 化し、`entropy_ig(after_before)` を適用。
  - 迷路 8 次元の重み `WEIGHT_VECTOR` を適用（A/B と一致）。
  - 正規化: `norm_strategy='before'`（A/B と一致）。
  - メトリクス: `metrics['h_scope']='cand_mask'` を記録。

### フォールバック

- `candidate_edges` 未提供で `sp_engine=cached_incr` が指定された場合:
  - デフォルト動作: `cached` にダウンシフト（ΔSP は between-graphs の固定ペア、候補逐次は行わない）。
  - `metrics['input_contract_ok']=False`、`metrics['autocand_used']=False` を記録し、ログ警告。
  - autocand はデフォルト禁止。研究モードでのみ有効（例: `graph.allow_autocand=true`）。

---

## AG 発火時の候補拡張（誰が行うか）

議題: AG が発火（採用）した直後に候補集団を広げる必要がある。責務は L1/L2 に差し戻すべきか、L3 独自で行うべきか。

### 案A: L1/L2 に差し戻し（推奨）

- 概要: L3 は「拡張要求（ExpansionRequest）」イベントを発し、L1/L2 が新たな Ecand を構築して返す。
- 利点:
  - 候補母集団の決定責務を L1/L2 に一元化（ベクトルインデックス/しきい値/半径などの一貫性）。
  - 将来的なスケール/最適化（ANN インデックスや sharding）を L1/L2 に寄せられる。
- フロー案:
  1) L3 で AG 採用（候補 k を採用）
  2) L3 → L1: `ExpansionRequest{ centers, prev_Ecand, reason=AG, budget, radius_hint, theta_link_hint }`
  3) L1/L2 が更新された `Ecand` を生成
  4) 次サイクルで L3 が新 Ecand を消費

### 案B: L3 局所拡張（限定・旗付き）

- 概要: L3 が centers 近傍でごく小規模な候補拡張を実施（例: N-hop 内の Top‑M のみ）。
- 利点: レイテンシが小さい（L1/L2 呼び戻し不要）。
- 欠点: 責務の混在・母集団の一貫性が崩れやすい。研究用フラグのみに限定。

### 案C: ハイブリッド（推奨の次点）

- 概要: まず L3 がヒント（centers/半径/Top‑M）を L1/L2 に渡し、L1/L2 が正式な Ecand を返す。L3単独拡張はフェイルオーバのみ。

---

## コンフィグ / ノブ

```yaml
graph:
  allow_autocand: false         # L3 の候補自動生成を禁止（既定）
  h_scope: cand_mask            # ΔH は候補マスク母集団で計算（固定）
  cand_expansion:
    mode: l1                    # l1 | l3 | hybrid
    budget: 16                  # 最大追加候補数
    near_centers_hop: 1         # L3局所拡張時のみ
```

ENV は導入しない（母集団や符号が再び分岐しないように固定）。

---

## テスト計画

- 単体
  - `candidate_edges` 未提供時のダウンシフト（cached）を検証し、`autocand_used=false` を確認。
  - ΔH（cand_mask）の符号・正規化の安定性（A/B と L3 の一致）をフラグ付きで比較。
  - 迷路 8D 重みの適用有無で ΔH の差が出ることを確認（重み適用が既定）。

- 結合
  - L1/L2→L3 で centers/Ecand/norm_spec を受け渡し、ΔH/ΔSP/IG が A/B と同方向・同傾き（許容閾内）になること。
  - `cand_expansion.mode=l1` で AG 発火→Ecand 拡張→次サイクルの評価に反映されること（ヒストリ上で候補数/IG が増える）。

- 性能
  - `cached_incr` の逐次採用パスで P95/P99 が予算内に収まること（sources_cap/prune/window/epsilon を活用）。

---

## 実装メモ（反映ポイント）

- L3 自動候補 OFF 既定化
  - `src/insightspike/implementations/layers/layer3_graph_reasoner.py`
    - 自動候補生成（cand 未提供時）のブロックを `allow_autocand` ガードで閉じ、既定 false。
    - cand 未提供 + `sp_engine=cached_incr` は `cached` にダウンシフトし、メトリクスに `input_contract_ok=false` を記録。

- ΔH = cand_mask 方式
  - 同ファイル（通常/union ブロック）で、ΔH 計算前に `centers ∪ endpoints(Ecand)` 以外の特徴を 0 化。
  - 迷路 8D 重み（WEIGHT_VECTOR）を適用。
  - `metrics['h_scope']='cand_mask'` を付与。

- AG 発火後の拡張
  - フック定義: L3 → L1/L2 の `ExpansionRequest`（イベント/コールバック）。
  - 既定 `cand_expansion.mode=l1` で L1/L2 が Ecand 再構築。`mode=l3` は研究用フラグ。

---

## まとめ

- Ecand の責務は L1/L2 に集約。L3 は評価とゲーティングに専念（autocand は OFF 既定）。
- ΔH は候補マスク母集団で計算して A/B と完全一致させる。
- AG 発火時の拡張は原則 L1/L2 が担当。必要ならハイブリッド案で L3 はヒント/フェイルオーバのみに限定。

---

## フローチャート（迷路実験の現在フローと責務色分け）

下図は、迷路実験（run_experiment_query.py）とメインコード（L1/L2/L3）の処理責務を色分けしたフローです。

```mermaid
flowchart TD
    %% Styles
    classDef main fill:#E3F2FD,stroke:#1E88E5,color:#0D47A1
    classDef exp  fill:#FFF3E0,stroke:#FB8C00,color:#E65100
    classDef data fill:#E8F5E9,stroke:#43A047,color:#1B5E20

    %% Inputs
    Q[Query / Environment Step]:::data

    %% L1: Conductor
    subgraph L1[Layer1 (Conductor) — Main]
        A1[Select centers]:::main
        A2[Build Ecand (and optionally by-hop)]:::main
        A3[Derive NormSpec (effective)]:::main
    end

    %% L2: Memory/Index
    subgraph L2[Layer2 (Memory/Index) — Main]
        B1[Index / Ring / ANN lookup]:::main
    end

    %% L3: Reasoner
    subgraph L3[Layer3 (Reasoner) — Main]
        C1[Receive context{\ncenters / Ecand[/by-hop]\n norm_spec}]:::main
        C2[ΔH (entropy_ig after−before)\n cand_mask + WEIGHT_VECTOR\n norm='before']:::main
        C3[ΔSP (cached_incr/cached)\n fixed-before-pairs, endpoint SSSP]:::main
        C4[IG = ΔH + γ·ΔSP]:::main
        C5[ΔGED_norm]:::main
        C6[g(h)=ΔGED−λ·IG, best_h]:::main
        C7[Gate (AG/DG)]:::main
        C8{AG fired?}:::main
        C9[Emit ExpansionRequest\n→ L1 (expand Ecand)]:::main
    end

    %% Experiment runner (build/visualize)
    subgraph EXP[Maze Experiment Runner — Experiment-only]
        E1[Build stage graph per step]:::exp
        E2[Union-of-k-hop node sets]:::exp
        E3[Record logs / Build HTML]:::exp
    end

    %% Flow
    Q --> A1
    A1 --> B1 --> A2 --> A3 --> C1
    E1 -->|provides before/after graphs| C1
    E2 -->|hop node sets (for union eval)| C6
    C1 --> C2 --> C3 --> C4 --> C5 --> C6 --> C7 --> C8
    C8 -- Yes --> C9 --> A2
    C8 -- No  --> E3

    %% Legend
    LLEGEND[Legend:\nBlue=Main code (L1/L2/L3), Orange=Experiment helper, Green=Data]:::data
```

補足:
- L3 の自動候補生成（autocand）は既定 OFF。Ecand 未提供 + cached_incr の場合は cached へダウンシフトし、メトリクスに契約違反を記録する。
- ΔH は cand_mask（centers ∪ endpoints(Ecand) 以外は 0 化）で計算し、A/B と一致させる。迷路 8D は WEIGHT_VECTOR を適用。
- AG 発火後の候補拡張は原則 L1 が実施（L3→L1 に ExpansionRequest）。研究用に限り L3 局所拡張をオプション化可能（既定 OFF）。

---

## 付録: ユニット/パイプライン テスト計画（抜粋）

1) ユニット（L3 自動候補 OFF / ダウンシフト）
- 入力: `sp_engine=cached_incr`, `candidate_edges=None`
- 期待: `metrics.sp_engine='cached'`, `metrics.input_contract_ok=False`, `metrics.autocand_used=False`

2) ユニット（ΔH=cand_mask 一致性）
- 小さな合成グラフで A/B の ΔH と L3 ΔH を比較（同じ sign と近い比率）
- 重み有/無のアブレーションで差が出ることを確認（既定は重み有）

3) ユニット（by-hop 候補優先）
- `candidate_edges_by_hop={0:[...],1:[...]}` 提供時に hop 別候補が使用されること
- 無い場合は単一 `candidate_edges` を共通使用

4) パイプライン（L1→L3 受け渡し）
- L1 スタブが centers/Ecand/norm_spec（必要なら by-hop）を返し、L3 が消費して ΔH/ΔSP/IG を出力
- A/B vs L3 で ΔH の符号と傾きが一致

5) パイプライン（AG→拡張）
- L3 が AG を検出→ ExpansionRequest を L1 に発行→ L1 が Ecand 拡張 → 次サイクルで候補数/IG が増える

6) 回帰
- benefit 指標が出力されないこと（`h_benefit`/`ig_benefit` 非存在）
- ENV/Config による ΔH 符号切替が発生しないこと
