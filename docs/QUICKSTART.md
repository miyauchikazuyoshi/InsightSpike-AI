# QUICKSTART — 5分で始める InsightSpike-AI

このガイドは「最初の5分で動かす」ための最短経路です。詳細は README.md や docs/paper を参照してください。

## 要件

- Python 3.10 以上
- OS依存なし（macOS/Linux/Windows）

## 1. セットアップ（3ステップ）

```bash
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

## 2. Hello World（2通り）

最短: モックLLMで質問→回答（外部API不要）

```bash
python examples/public_quick_start.py
```

geDIGゲージの最小デモ（ΔEPC/ΔIGとFを表示, paper整合: linkset IG）

```bash
python examples/hello_insight.py
```

### 追加設定（セクション単位の上書き）

`create_agent` に `section__field=value` 形式で渡すと、Pydantic 設定を安全に上書きできます。たとえば LLM 温度や処理サイクルを最短経路で調整可能です。

```python
from insightspike import create_agent

agent = create_agent(
    provider="mock",
    llm__temperature=0.2,       # llm セクションの temperature を直接指定
    processing__max_cycles=3,   # 任意のセクションにネストしてアクセス
)
print(agent.config.llm.temperature, agent.config.processing.max_cycles)
```

最小コード例（paper‑aligned: linkset_info を明示）

```python
from insightspike.algorithms.gedig_core import GeDIGCore
from insightspike.algorithms.linkset_adapter import build_linkset_info
import networkx as nx

g1 = nx.Graph(); g1.add_edges_from([('A','B'),('B','C')])
g2 = nx.Graph(); g2.add_edges_from([('A','B'),('B','C'),('A','C')])

core = GeDIGCore()
linkset = build_linkset_info(
    s_link=[{"index": 1, "similarity": 1.0}],
    candidate_pool=[],
    decision={"index": 1, "similarity": 1.0},
    query_vector=[1.0],
    base_mode="link",
)
res = core.calculate(g_prev=g1, g_now=g2, linkset_info=linkset)
print(res.gedig_value, res.ged_value, res.ig_value)
```

期待される出力例（概略）:

```
F = -0.42  (ΔEPC_norm=0.15,  ΔIG=0.57,  spike=True)
```

## 3. 最短の実験実行

迷路（小規模で高速）

```bash
python examples/maze50_experiment.py --size 15 --max-steps 1000 --verbosity 1
```

RAG（準備中）

- 実験手順は EXPERIMENTS.md を参照してください（フェーズ整理中）。

## トラブルシュート

- 依存関係の競合: 仮想環境を作り直し、`pip install -e .` を再実行
- 速度が遅い: まずは `examples/public_quick_start.py` で挙動確認 → 迷路は `--size 15` 程度から

## 次の一歩

- CONCEPTS.md で用語/理論（ΔEPC/ΔIG, One‑Gauge, AG/DG）を把握
- EXPERIMENTS.md で迷路やRAGの再現を進める
- 論文 v3（EPC基準）: docs/paper/geDIG_onegauge_improved_v3.tex
