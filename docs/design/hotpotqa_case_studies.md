# HotPotQA dev Case Studies（geDIG vs baselines）

差分分析から代表例を抽出。F1差分が大きい上位ケースを掲載。


## 使用ファイル

- gedig: `experiments/hotpotqa-benchmark/results/gedig_20260122_100418.jsonl`
- static_graphrag: `experiments/hotpotqa-benchmark/results/static_graphrag_20260115_231836.jsonl`
- bm25: `experiments/hotpotqa-benchmark/results/bm25_20260115_115855.jsonl`

## vs static_graphrag: 勝ちケース (Top 3)

| id | ΔF1 | geDIG F1 | baseline F1 | type | level | question | geDIG pred | baseline pred | gold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5ab92307554299753720f72d | 1.000 | 1.000 | 0.000 | bridge | hard | What Pakistani actor and writer from Islamabad helped write for the 2012 Pakistani come... | Yasir Hussain | Syed Mohammad Ahmed | Yasir Hussain |
| 5a7fb7985542994857a767d2 | 1.000 | 1.000 | 0.000 | bridge | hard | Which retired Argentine footballer who played as a forward was a main player for Valenc... | Claudio Javier López | Pablo Aimar | Claudio Javier López |
| 5ab29426554299545a2cf99f | 1.000 | 1.000 | 0.000 | bridge | hard | Calvin Murphy's record of being the shortest NBA player to play in an All-Star Game was... | Cavaliers | Boston Celtics | Cavaliers |

## vs static_graphrag: 負けケース (Top 3)

| id | ΔF1 | geDIG F1 | baseline F1 | type | level | question | geDIG pred | baseline pred | gold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5a87c13f5542996e4f30890c | -1.000 | 0.000 | 1.000 | bridge | hard | In what city did the "Prince of tenors" star in a film based on an opera by Giacomo Puc... | New York City | Rome | Rome |
| 5a85fb085542994775f606de | -1.000 | 0.000 | 1.000 | bridge | hard | What is the name of the executive producer of the film that has a score composed by Jer... | Francis Ford Coppola | Ronald Shusett | Ronald Shusett |
| 5ab2a186554299295394677b | -1.000 | 0.000 | 1.000 | bridge | hard | When did the English local newspaper, featuring the sculpture and war memorial in the F... | The context does not provide information about when the English local newspaper changed... | 2009 | 2009 |

## vs bm25: 勝ちケース (Top 3)

| id | ΔF1 | geDIG F1 | baseline F1 | type | level | question | geDIG pred | baseline pred | gold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5ac132a755429964131be17c | 1.000 | 1.000 | 0.000 | bridge | hard | Blackfin is a family of processors developed by the company that is headquartered in wh... | Norwood, Massachusetts. | Santa Clara, California. | Norwood, Massachusetts |
| 5aba94e355429901930fa845 | 1.000 | 1.000 | 0.000 | bridge | hard | Jonathan Groff was 30 years old when he appeared in a historical musical written by whom? | Lin-Manuel Miranda | Ian Brennan | Lin-Manuel Miranda |
| 5a7fb7985542994857a767d2 | 1.000 | 1.000 | 0.000 | bridge | hard | Which retired Argentine footballer who played as a forward was a main player for Valenc... | Claudio Javier López | Pablo Aimar | Claudio Javier López |

## vs bm25: 負けケース (Top 3)

| id | ΔF1 | geDIG F1 | baseline F1 | type | level | question | geDIG pred | baseline pred | gold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5ae058e855429945ae959331 | -1.000 | 0.000 | 1.000 | comparison | hard | Are  Chrysalis and Look both women's magazines? | No. | Yes. | yes |
| 5a7a46605542994f819ef1ad | -1.000 | 0.000 | 1.000 | bridge | hard | What year did Roy Rogers and his third wife star in a film directed by Frank McDonald? | 1944 | 1945 | 1945 |
| 5a86769c5542994775f60776 | -1.000 | 0.000 | 1.000 | bridge | hard | Armageddon in Retrospect was written by the author who was best known for what 1969 sat... | The Magic Christian. | "Slaughterhouse-Five" | Slaughterhouse-Five |

## 分析テンプレ（追記予定）

各ケースについて以下を埋める。

- 取得文書: supporting facts の再現率 / distractor 比率
- ゲート挙動: `initial_ag_fired` / `dg_fired` / `graph_edges`
- 失敗モード: 取得不足 / 誤統合 / 生成誤り
- 改善案: しきい値調整 / expansion 設定 / rerank 条件

## 追加で深掘りする候補（pending）

- static_graphrag 負け: `5a87c13f5542996e4f30890c`, `5a85fb085542994775f606de`, `5ab2a186554299295394677b`
- bm25 負け: `5ae058e855429945ae959331`, `5a7a46605542994f819ef1ad`, `5a86769c5542994775f60776`
