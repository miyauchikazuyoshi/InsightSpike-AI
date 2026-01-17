# HotpotQA distractor dev (closed-world) 最新まとめ

## 実行条件
- データ: HotpotQA distractor dev (7,405)
- 設定: 各例の context のみを検索対象とする閉世界
- LLM: gpt-4o-mini, temperature=0.0, max_tokens=256

## 主要結果 (dev)
| method | EM | F1 | SF-F1 | p50(ms) | p95(ms) | count | 備考 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| geDIG (lambda=0.8) | 0.3733 | 0.5358 | 0.3043 | 783.20 | 18628.74 | 7405 | gateバランス型 |
| geDIG (lambda=0.5) | 0.3749 | 0.5375 | 0.3043 | 872.52 | 18545.52 | 7405 | DG寄りの挙動 |
| BM25 + GPT-4o-mini | 0.3662 | 0.5229 | 0.3497 | 819.98 | 18532.92 | 7405 | baseline |
| Static GraphRAG | 0.3893 | 0.5594 | 0.3497 | 913.71 | 18503.34 | 7405 | baseline |
| Closed-book | 0.2258 | 0.3544 | 0.0000 | 739.68 | 10065.97 | 4911 | incomplete |

## 比較メモ
- lambda=0.5 は EM/F1 がわずかに高い (+0.0016/+0.0017) が、p50 は遅い (+約89ms)。
- gate挙動は差が大きく、lambda=0.8 は final_ag_fire_rate=0.544 / final_dg_fire_rate=0.291、lambda=0.5 は 0.129 / 0.815。
- geDIG は BM25 より EM/F1 は上、SF-F1 は下 (0.3043 < 0.3497)。
- Static GraphRAG は EM/F1 が最も高いが、p50 は他より遅め (913.71ms)。
- Closed-book は明確に低めだが 4911/7405 のため参考値。

## 次の実行候補
- Contriever (dev)
- Closed-book (dev) 完走

## メモ
- OpenAI RPD 制限の影響で、長時間の連続実行になる可能性あり。
