# QUICKSTART — 5分で始める InsightSpike-AI

詳しい手順は `docs/QUICKSTART.md` を正としつつ、ここにも最短ルートだけ抜き出しておきます。

## セットアップ

```bash
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

## Hello World（モック LLM）

```bash
python examples/public_quick_start.py
```

## geDIG ゲージの最小デモ

```bash
python examples/hello_insight.py
```

より詳しい説明・トラブルシュート・実験の入口は `docs/QUICKSTART.md` を参照してください。

