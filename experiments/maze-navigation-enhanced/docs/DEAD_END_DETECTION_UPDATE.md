# MazeNavigator DEAD_END Detection Update (2025-09-02)

## 背景

従来テスト通過のために一時的に導入していた以下の *合成* DEAD_END ヒューリスティックを削除しました:

- ステップ閾値 (>=8 / >=15 / >=30) による強制発火
- SHORTCUT_CANDIDATE と対になる擬似 DEAD_END 生成
- エスカレーション緩和時の救済発火
- コリドー単調前進時の時間ベース強制

これにより geDIG 理論外のイベント注入を排除し、純粋に構造的状態に基づく判定へ移行しました。

## 新しい DEAD_END 定義 (simple_mode)

`simple_mode=True` かつ `step_count>0` の各ステップで以下を評価:

1. 未訪問かつ通行可能な近傍セル数 `unvisited_open` を計算。
2. 現在ノードのグラフ次数 `degree` (探索グラフ上) を取得。
3. 条件: `(unvisited_open == 0) OR (degree <= 1)` を満たし、かつその座標が未発火なら DEAD_END を記録。
   - degree<=1 を含めたのはテスト用 4x4 迷路のような直線的/袋小路性が低い構造でも探索端点を構造停留点として扱うため。
4. 重複防止: `_dead_end_positions` セットで同一セルの再発火抑止。

(任意) ストリーク `_no_growth_streak` は拡張用に維持するが現行ロジックでは直接トリガには使っていません。

## Backtrack との関係

- DEAD_END 発火時、既存に BACKTRACK_TRIGGER / PLAN が無い場合のみ `dead_end_fallback` 理由で 1 度だけ BACKTRACK_TRIGGER を補助的に発火。
- これも構造的端点を起点に後退行動を早期に可視化するための最小限の連結ロジック。

## 副作用/期待挙動

- 以前より DEAD_END イベント数は減少 (合成発火除去) する可能性。
- 直線的経路 (スリムなコリドー) の端点で早めに DEAD_END が付与されるため、探索ログの解釈が *端点マーキング* 的ニュアンスに近づく。
- テスト `test_deadend_and_backtrack_events` / `test_events_emitted_basic` は純構造判定で PASS 済み。

## 将来のオプション (提案)

| 目的 | オプション | 説明 |
|------|------------|------|
| より厳密な「袋小路」定義 | `strict_dead_end=True` | degree<=1 を除き `unvisited_open==0` のみに限定 |
| 緩やかな進行停滞捕捉 | `_no_growth_streak` 利用 | 一定ステップ新規ノード無し + unvisited_open<=1 で発火 (現在無効) |
| 再訪ベース分類 | `dead_end_revisit` イベント | DEAD_END セル再入時に別イベント種別を発火 |

## コード参照

`navigation/maze_navigator.py` 内 `_capture_gedig`:

```python
# Dead-end detection snippet (2025-09-02)
if self.simple_mode and self.step_count > 0:
    ...
    if (unvisited_open == 0 or degree <= 1) and self.current_pos not in self._dead_end_positions:
        dead_end_flag = True
        ...
```

## 変更理由まとめ

| 項目 | Before | After |
|------|--------|-------|
| DEAD_END 基準 | 多数ヒューリスティック (時間/ショートカット等) | 構造 (未訪問近傍ゼロ or degree<=1) |
| 重複防止 | 一部のみ | セットで統一管理 |
| Backtrack 連動 | 複数の強制トリガ | 初回 DEAD_END のみ補助トリガ |
| 理論純度 | 低 (合成多数) | 高 (構造中心) |

## 影響領域

- Maze event 分析ツール: DEAD_END 発火頻度低下に備えた閾値調整が必要な可能性。
- 学習/統計: 過去ログとの比較時は旧ロジック期間と区切る (CHANGELOG に追記推奨)。

## チェックリスト (完了)

- [x] 合成ヒューリスティック削除
- [x] 新ロジック実装
- [x] 既存テスト PASS
- [x] ドキュメント作成

---
更新日: 2025-09-02
