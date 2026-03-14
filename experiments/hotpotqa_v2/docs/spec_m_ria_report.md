# Spec M: RIA (Recursive Insight Architecture) — 実験レポート

## 実験概要

論文 RIA（再帰的洞察ベクトルによる段階的強化）を BRIGHT パイプラインに実装。
Phase 2.5 の 1-shot CoT re-retrieval を **β₀-gated iterative loop** に拡張し、
top-5 文書を LLM にフィードバックして新しい検索キーワードを生成、
BM25 再検索で retrieval recall を改善する。

## 結果

### 50q Biology

| Metric | Baseline (Spec H) | Spec M (RIA) | 変化 |
|--------|-------------------|-------------|------|
| nDCG@10 | 0.2496 | **0.2564** | **+2.7%** |
| Recall@10 | 0.2419 | **0.2661** | **+10.0%** |
| MRR | 0.3563 | 0.3815 | +7.1% |

### RIA 動作統計

| 指標 | 値 |
|------|-----|
| RIA 合計追加文書 | 3,050 |
| RIA で発見した gold 文書 | **38** |
| RIA で gold を発見したクエリ | **19/50 (38%)** |
| CoT re-retrieval の gold 文書 | 50 |
| **合計 gold 発見 (CoT + RIA)** | **88** |
| ランキングに gold があるクエリ | 26/50 (52%) |

### ラウンド分布

| ラウンド数 | クエリ数 | 備考 |
|-----------|---------|------|
| 0 | 1 | β₀ ≤ target で即収束 |
| 1 | 38 | 大多数 — β₀ が round 1 で非改善 |
| 2 | 10 | β₀ が round 1 で改善 → round 2 実行 |
| 3 | 1 | 全 3 ラウンド実行 |
| **平均** | **1.2** | |

### RIA gold 発見の効果

| カテゴリ | 平均 nDCG@10 | クエリ数 |
|---------|-------------|---------|
| RIA で gold 発見あり | **0.2977** | 19 |
| RIA で gold 発見なし | 0.2311 | 31 |

## ハイライト

### 成功ケース

- **Q35** (髪が白くなる理由): RIA が 2 ラウンドで **+6 gold** 発見 → **nDCG=1.000** (perfect score!)
  - β₀ 推移: 9→5→7 (round 1 で改善)
- **Q3** (動物の利き手): RIA が +1 gold → nDCG=1.000
- **Q17** (食品のタンパク質): RIA が +2 gold → nDCG=0.762
- **Q47** (アリの行列): nDCG=0.860 — CoT + RIA の相乗効果

### β₀ 停止条件の挙動

β₀ は多くのケースで RIA ラウンド後に**増加** (悪化) する:
- 新しい文書が構造的に切断されているため、connected components が増える
- β₀ が減少 (改善) するケースでは、複数ラウンド実行 → gold 発見率が高い
- **β₀ 減少は「ブリッジング文書の発見」を示唆** — 理論と整合

## 考察

### ポジティブ

1. **Recall 改善が確認** (+10.0%) — RIA の主目的である retrieval recall が改善
2. **38 個の新 gold 文書** — CoT re-retrieval (50 個) に加えて、RIA がさらに gold を発見
3. **コストが低い** — gpt-4o-mini expansion は 1クエリあたり ~$0.01 追加
4. **β₀ 停止条件が機能** — β₀ 減少時に複数ラウンド実行、gold 発見率向上

### 課題

1. **nDCG 改善は微小** (+2.7%) — recall は改善するが ranking 品質が追いつかない
2. **β₀ が増加するケースが多い** — 大半のクエリで 1 ラウンドで停止
3. **Graph slot allocation** — RIA 文書にスロットを割くと BM25 文書が減る → ノイズ増加の可能性

### Spec I/J/K/L との比較

| Spec | 手法 | nDCG@10 | Recall@10 | 判定 |
|------|------|---------|-----------|------|
| H (baseline) | geDIG refine | 0.2496 | 0.2419 | — |
| I | Pointwise rerank (mini) | 0.2340 | 0.2419 | ❌ -6.3% |
| J | Pointwise rerank (mini v2) | — | — | ❌ |
| K | Query decomposition | 0.1978 | — | ❌ -20.8% |
| L | Reasoning rerank (gpt-4o) | 0.2342 | — | ❌ -6.2% |
| **M** | **RIA iterative expansion** | **0.2564** | **0.2661** | **✅ +2.7%** |

**Spec M は Spec H 以降初めての正の改善** — 5連続の negative results を break。

## 次のステップ

1. **β₀ 停止条件の改善**: β₀ 非改善でも最低 2 ラウンドは実行する variant
2. **RIA + max_rounds=2 固定**: β₀ を使わず DIVER と同じ 2 ラウンド固定
3. **Graph slot allocation の最適化**: RIA 文書のスロット比率調整
4. **Spec N (統一グラフ)**: token-level graph で ranking 品質を改善
5. **Multi-domain テスト**: economics, stackoverflow でも検証

## 実行コマンド

```bash
# Smoke test (10q)
export $(cat .env | xargs) && \
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_bright.py \
    --mode cot_retrieval --domains biology \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --output experiments/hotpotqa_v2/results/v13_bright_ria_smoke \
    --limit 10 --graph-top-k 50 --rerank-alpha 0.1 \
    --scoring-mode gedig_refine \
    --ria-loop --ria-max-rounds 3

# 50q test
export $(cat .env | xargs) && \
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_bright.py \
    --mode cot_retrieval --domains biology \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --output experiments/hotpotqa_v2/results/v13_bright_ria_50q \
    --limit 50 --graph-top-k 50 --rerank-alpha 0.1 \
    --scoring-mode gedig_refine \
    --ria-loop --ria-max-rounds 3
```
