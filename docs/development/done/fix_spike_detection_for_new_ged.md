---
status: active
category: meta
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# スパイク検出修正案

## 問題
- 現在のスパイク検出: `delta_ged < -0.5` (負の値を期待)
- 新GED実装: GEDは常に正の値

## 解決策

### Option 1: メトリクスセレクターで変換
`_new_ged_wrapper`で既に実装済み：
```python
if result["structural_improvement"] > 0.1:
    return -result["ged"]  # 構造改善時は負の値を返す
else:
    return result["ged"]
```

### Option 2: スパイク検出ロジックを更新
新しいGEDに対応した判定：
- 構造改善度（structural_improvement）を使用
- または、GEDとIGの組み合わせで判定

## 現状の問題点
1. `_new_ged_wrapper`が呼ばれていない可能性
2. GraphAnalyzerが直接GED値を使用している

## 推奨解決策
1. GraphAnalyzerのdetect_spikeメソッドを修正
2. 新旧GED実装を判別して適切に処理