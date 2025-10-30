---
title: Maze Navigator Simple Mode リファクタ計画 (Phase A/B)
status: Phase A complete (awaiting Phase B docs/tests)
decision_due: 2025-09-05
next_step: "B2(README)/B3(Tests)/multi-run metrics validation"
created: 2025-08-28
owner: maze-nav
tags: [maze, navigator, refactor, simplicity]
---

<!-- markdownlint-disable MD041 -->

## 1. 背景 / 動機 (Why)

現行 MazeNavigator は探索安定化のために以下の追加ロジックを内包:

- 動的 p05 ベース backtrack 閾値
- 停滞 (stagnation) 検出
- クールダウン / 発火理由ログ多様化
- クエリの二重生成（配線用と行動選択用）

初期 PoC 目的 (query 駆動配線 + geDIG 閾値トリガ) は達成済みであり、今後の geDIG Phase 2 強化 (別計画) を進める上で **コアパイプラインを最小・決定性高く再定義** し、計測/比較のノイズ源を排除する。

## 2. スコープ (Scope)

### In Scope

- simple_mode 導入と最小直列フロー化
- クエリ生成/再利用統一 (1 ステップ 1 回)
- 静的 backtrack 閾値のみ (任意 1 step デバウンス)
- query ベース配線 API シグネチャ整理
- 壁 Episode 常時含有ポリシーのコード化 & deprecation
- 最低限メトリクス追加 (query回数, backtrack件数)

### Out of Scope (後続 / Phase 3 以降)

- 動的閾値再導入
- 停滞/密度ヒューリスティック
- frontier / advanced backtrack policy
- pruning / memory 圧縮

## 3. As-Is vs To-Be

| 項目 | As-Is | To-Be (Simple Mode) |
|------|-------|---------------------|
| クエリ生成 | 配線/行動で重複生成 | 1回生成を step 内キャッシュ |
| 配線API | `_wire_episodes_query_based()` 内部で query 生成 | 外部注入: `_wire_episodes_query_based(query)` |
| 行動選択 | `select_action` 内部で独自クエリ (又は簡略) | `select_action(episodes, query=...)` 再利用 |
| backtrack | 静的 + 動的 + 停滞 + CD | 静的閾値 (+1 step debounce 任意) |
| 状態保持 | p05 ウィンドウ, stagnation, cooldown | geDIG 直近値 / デバウンスフラグのみ |
| 壁Episode | include_walls 引数 (実質常True) | 強制 True / 警告 (非推奨) |
| ログ | 多理由タグ | trigger 有無のみ (理由固定 "static") |

## 4. 目標パイプライン (Simple Mode Core Flow)

1. 観測: 4方向 Episode 更新/生成
2. クエリ生成 (prefer_unexplored=True)
3. クエリ駆動配線 (前エッジ強制 → Top-K 類似 Episode 双方向接続)
4. geDIG 計算 (前スナップショットとの差分)
5. backtrack 判定 (geDIG <= threshold [+ debounce])
6. 行動選択: ローカル4 Episode に対し距離→softmax (壁含む)
7. 移動 & visit_count/log 更新 & snapshot 保存
8. ゴール判定

## 5. 非機能要件

| 区分 | 要件 |
|------|------|
| 決定性 | 同一 seed + config で挙動再現 ±0 差異 |
| 単純性 | 主要クラス行数 +10% 以内 (複雑化禁止) |
| 性能 | ステップ処理時間 (25x25) 現行 ±5% 以内 |
| ログ | 1 ステップ JSON (Simple 指標最小) |

## 6. API 変更

### 追加

- `MazeNavigator.__init__(..., simple_mode: bool=False)`
- `DecisionEngine.select_action(episodes, temperature=0.1, query=None)`

### 変更

- `_wire_episodes_query_based(query)` (旧: 引数なし)

### Deprecated (警告)

- `DecisionEngine.select_action(..., include_walls=...)` (内部強制 True)

## 7. Config 拡張

| キー | 型 | 既定 | 説明 |
|------|----|------|------|
| simple_mode | bool | False | True で Simple パイプライン有効 |
| backtrack_debounce | bool | True | 連続フレーム発火抑止 (Simple時のみ評価) |

## 8. 実装タスク (Phase A/B)

| ID | Task | 内容 | 出力 | 状態 |
|----|------|------|------|------|
| A1 | Flag追加 | simple_mode + debounce 設定反映 | maze_navigator.py/config | 完了 |
| A2 | クエリ統一 | step 内 1回生成 / キャッシュ | maze_navigator.py | 完了 |
| A3 | 配線API修正 | 引数 query 化 / 内部生成除去 | navigator (_wire_episodes_query_based) | 完了 |
| A4 | 行動選択拡張 | select_action(query=) 実装 | decision_engine.py | 完了 |
| A5 | Backtrack簡素化 | 静的閾値 + debounce 分岐実装 | maze_navigator.py | 完了 |
| A6 | 壁含有強制 | include_walls 無視 / Warning | decision_engine.py | 完了 (警告実装) |
| B1 | メトリクス | query回数, queries_per_step, backtrack_trigger_rate | stats dict | 完了 |
| B2 | README更新 | 新フロー/legacy差分 | README.md | 未 |
| B3 | テスト更新 | simple/legacy パラメトリ化 + 最低4テスト | tests/navigation | 未 |

### 8.1 Checklist (進捗サマリ)

Legend: [x] 完了 / [ ] 未 / [~] 部分

- [x] A1 Flag追加 (simple_mode / debounce)
- [x] A2 クエリ統一 (1ステップ1生成キャッシュ)
- [x] A3 配線API修正 (`_wire_episodes_query_based(query)`)
- [x] A4 行動選択拡張 (`select_action(query=)`)
- [x] A5 Backtrack簡素化 (静的閾値 + debounce)
- [x] A6 壁含有強制 (include_walls 無視 + Warning)
- [x] B1 メトリクス追加 (query/backtrack 指標)
- [ ] B2 README更新 (Simple Mode / Legacy 差分記載)
- [ ] B3 テスト拡張 (4テスト param 化)

## 9. データ構造変更

- 追加フィールド: `self._current_query: Optional[np.ndarray]`
- 削除予定フィールド (legacy 保持期は None 更新): `self._gedig_window`, `self._stagnation_positions`, `self._cooldown_counter`

## 10. ロギング仕様 (Simple Mode)

```json
{
  "step": 42,
  "pos": [x,y],
  "action": "N",
  "gedig": -0.07,
  "backtrack_trigger": true,
  "backtrack_reason": "static",
  "query_k": 4,
  "edges_added": 3
}
```

## 11. メトリクス定義

| 指標 | 定義 |
|------|------|
| query_generated_per_step | 総生成数 / steps (期待=1.0) |
| backtrack_trigger_rate | triggers / steps |
| avg_degree_query_nodes | query戦略使用時の平均次数 |
| action_entropy_mean | softmax エントロピー平均 |

## 12. Exit Criteria (進捗注記)

- [~] query_generated_per_step ≈ 1.0 (±0.01) (計測コード挿入済 / 実測未)
- [ ] backtrack_trigger_rate 安定 (ログ集計未)
- [ ] T字 / 25x25 成功率 ±5% 以内 (未計測)
- [ ] 主要テスト緑 (simple_mode=True / legacy=False) (追加未)
- [ ] README 新フロー適用 (未)

## 13. リスク & 緩和

| リスク | 影響 | 緩和 |
|--------|------|------|
| 閾値振動 | 過剰 backtrack | debounce / 閾値微調整 (-0.05→-0.08) |
| API 破壊 | 既存実験互換性 | legacy パス保守 2 週間 |
| 観測抜け | クエリ前後の順序崩壊 | step コメント + テストで順序検証 |

## 14. 段階的移行

- Week 1: Phase A 実装 + 基本テスト
- Week 2: Phase B 計測/README/legacy 切替 (デフォ simple_mode=True)
- Week 3+: legacy 跡地整理 / Phase 2 geDIG 強化と統合

## 15. テスト戦略

| テスト | 目的 |
|--------|------|
| test_query_single_generation | クエリ生成回数検証 |
| test_backtrack_static_only | 動的機構無効化確認 |
| test_simple_vs_legacy_path_length | 成果比較 (統計境界内) |
| test_wall_always_included | 壁候補集合含有性 |

## 16. 非互換点 (Breaking Notes)

- 外部から `_wire_episodes_query_based()` を直接呼ぶコードは引数追加必須
- include_walls フラグ効果喪失 (警告ログ出力)

## 17. 承認事項

- simple_mode デフォルト化タイミング (Phase B 終了時) 合意
- backtrack_debounce 初期 True で問題なし
- 静的閾値初期値: `-0.05` (調整許容範囲: [-0.1, 0.0])

## 18. 次アクション (最新)

1. A6: include_walls=False 指定時に Warning ログ追加 (完了済)
2. B2: README に Simple Mode 実装状況テーブル/旧機能差分を反映 (未)
3. B3: 最低テスト4件追加 (未)

- test_query_single_generation
- test_backtrack_simple_static_only
- test_select_action_uses_external_query (クエリ再利用)
- test_statistics_simple_mode_block

1. backtrack_trigger_rate 追加 (gedig_history ベース判定回数計測) (B1 対応済)
2. 25x25 シナリオ単走で query_generated_per_step 出力確認 (未: evidence 系計測に統合予定)

---
Archive Notice: 本計画は TEST_FAILURE_REMEDIATION_PLAN_2025_09.md に統合され未完了項目 (README 更新 / テスト追加) はそちらの M1 後 backlog に移管予定。アーカイブ移動可。

---
Generated: 2025-08-28
