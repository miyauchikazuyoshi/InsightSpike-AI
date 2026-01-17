# HotPotQA dev Case Studies（geDIG vs baselines）

差分分析から代表例を抽出。F1差分が大きい上位ケースを掲載。


## 使用ファイル

- gedig: `experiments/hotpotqa-benchmark/results/gedig_20260114_233727.jsonl`
- static_graphrag: `experiments/hotpotqa-benchmark/results/static_graphrag_20260115_231836.jsonl`
- bm25: `experiments/hotpotqa-benchmark/results/bm25_20260115_115855.jsonl`

## vs static_graphrag: 勝ちケース (Top 3)

| id | ΔF1 | geDIG F1 | baseline F1 | type | level | question | geDIG pred | baseline pred | gold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5ae76c625542997ec272763e | 1.000 | 1.000 | 0.000 | bridge | hard | Which founder of Alcatraz East was born on March 31, 1956? | John Morgan | The founder of Alcatraz East born on ... | John Morgan |
| 5a7471f655429979e2882955 | 1.000 | 1.000 | 0.000 | bridge | hard | Which American main title designer designed sequences for the founder of the ... | Wayne Fitzgerald | Kyle Cooper | Wayne Fitzgerald |
| 5a875cd65542996e4f30879d | 1.000 | 1.000 | 0.000 | bridge | hard | What kind of producer is the person who helped created "The Spiderwick Cronic... | Motion picture producer. | The person is a writer and illustrator. | motion picture producer |

## vs static_graphrag: 負けケース (Top 3)

| id | ΔF1 | geDIG F1 | baseline F1 | type | level | question | geDIG pred | baseline pred | gold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5ae45cba5542991a06ce9a4a | -1.000 | 0.000 | 1.000 | bridge | hard | Tommy Faile sang the song "the legend of Brown Moutain Lights", about a serie... | The Brown Mountain Overlook on NC Hig... | Wiseman's View | Wiseman's View |
| 5ac2509755429951e9e68543 | -1.000 | 0.000 | 1.000 | bridge | hard | What is the nationality of the man who finished third in the 2016 Pau Grand P... | Italian | British | British |
| 5ab93781554299131ca422c7 | -1.000 | 0.000 | 1.000 | bridge | hard | Which star in the Maltese Bippy died in 1987? | Marty Feldman. | Dan Rowan. | Dan Rowan |

## vs bm25: 勝ちケース (Top 3)

| id | ΔF1 | geDIG F1 | baseline F1 | type | level | question | geDIG pred | baseline pred | gold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5ab510cb5542996a3a96a00f | 1.000 | 1.000 | 0.000 | comparison | hard | Were both the Battle of Saipan and the Battle of Manila fought by American fo... | Yes. | No, only the Battle of Saipan was fou... | yes |
| 5ae76c625542997ec272763e | 1.000 | 1.000 | 0.000 | bridge | hard | Which founder of Alcatraz East was born on March 31, 1956? | John Morgan | The founder of Alcatraz East born on ... | John Morgan |
| 5a7471f655429979e2882955 | 1.000 | 1.000 | 0.000 | bridge | hard | Which American main title designer designed sequences for the founder of the ... | Wayne Fitzgerald | Dan Curry | Wayne Fitzgerald |

## vs bm25: 負けケース (Top 3)

| id | ΔF1 | geDIG F1 | baseline F1 | type | level | question | geDIG pred | baseline pred | gold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5a7914e155429974737f7973 | -1.000 | 0.000 | 1.000 | bridge | hard | What is the birthday of the oldest among Vicente García and Ximena Sariñana? | January 23, 1833 | March 30, 1983 | March 30, 1983 |
| 5a7172105542994082a3e837 | -1.000 | 0.000 | 1.000 | bridge | hard | What film did Tom Wu appear in that also starred Jason Statham and Ray Liotta? | In the Name of the King: A Dungeon Si... | Revolver | Revolver |
| 5ae13bf1554299422ee99634 | -1.000 | 0.000 | 1.000 | bridge | hard | What American pianist and composer was a guest muscian on the Manhattan Blues... | David Foster | Philip Aaberg | Philip Aaberg |