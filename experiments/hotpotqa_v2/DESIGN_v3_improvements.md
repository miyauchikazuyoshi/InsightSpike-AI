# geDIG v3 改善設計書 — System 1 / System 2 アーキテクチャ

**作成日**: 2026-03-08
**Status**: Phase 1-2 完了 (E1 tuning + Hybrid実装・評価済み)
**前提**: v2 実験結果 (100問, GPT-4o-mini)

---

## 0. 現状分析

### v2 全8手法比較結果

| Rank | Method | Type | EM | F1 | Latency | LLM Calls/Q |
|:----:|--------|------|:---:|:---:|--------:|:-----------:|
| 1 | IRCoT | Dynamic | **46.0%** | **0.637** | 4,158ms | ~8 |
| 2 | GraphRAG | Static | 43.0% | 0.589 | 813ms | 1 |
| 3 | geDIG-B(β₁) | geDIG | 40.0% | 0.570 | 865ms | 1 |
| 4 | ReAct | Dynamic | 39.0% | 0.536 | 23,564ms | ~7 |
| 5 | geDIG-C(β₀) | geDIG | 38.0% | 0.553 | 878ms | 1 |
| 6 | geDIG-A(SP) | geDIG | 37.0% | 0.545 | 902ms | 1 |
| 7 | geDIG-D(Full) | geDIG | 37.0% | 0.544 | 910ms | 1 |
| 8 | BM25 | Static | 37.0% | 0.536 | 815ms | 1 |

### 発見された弱点

#### 弱点1: DG早期発火 (最大の問題)

Condition-B のゲート発火パターン:

| Gate | N | EM | F1 | 解釈 |
|------|--:|:---:|:---:|------|
| DG_only (即答) | 73 | 38.4% | 0.544 | β₁が強すぎて即停止 |
| AG_only (探索) | 14 | 42.9% | 0.575 | 探索した方が良い |
| Neither (中立) | 13 | **46.2%** | **0.715** | **最高成績** |

- β₁ペナルティで F値が平均 -1.33 に沈み、73% が DG 即答
- **neitherゾーンが最高成績** → DG を発火させすぎている

#### 弱点2: 推論なき判断

IRCoT に負けた18問のうち14問がDG即答。

```
Q: "Greyia と Calibanus、どちらの属の種が多い？"
geDIG-B → DG即答「Calibanus」(不正解、情報不足)
IRCoT   → CoT 3ステップで両方確認 →「Greyia」(正解)
```

geDIG は「何の情報が不足しているか」を理解せず、グラフ統計量だけで判断。

#### 弱点3: Comparison型の不安定さ

| Method | Bridge EM | Comparison EM |
|--------|:---------:|:------------:|
| geDIG-B | 39.5% | 42.9% |
| ReAct | 33.7% | **71.4%** |
| IRCoT | 44.2% | 57.1% |

---

## 1. 設計思想: System 1 / System 2

人間の思考は二重過程理論 (Kahneman, 2011) で記述される:

```
System 1 (速い思考): 直観的・自動的・低コスト
  → 「この問題は簡単そう、答えは明らか」
  → geDIG の DG 即答に対応

System 2 (遅い思考): 分析的・意識的・高コスト
  → 「これは難しい、ステップを踏んで考えよう」
  → IRCoT の CoT 推論に対応
```

### geDIG v3 = Topology-Guided Dual-Process RAG

```
              ┌─────────────┐
  Question →  │  BM25 検索   │ → 初期コンテキスト
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │  geDIG ゲージ │ → F値 (トポロジカル信頼度)
              └──────┬──────┘
                     │
            ┌────────┴────────┐
            │                  │
     F < θ_dg            F ≥ θ_dg
     (高信頼)              (低信頼)
            │                  │
     ┌──────▼──────┐   ┌──────▼──────┐
     │  System 1    │   │  System 2    │
     │  DG 即答     │   │  CoT 推論    │
     │  (1 LLM call)│   │  (2-3 calls) │
     └──────┬──────┘   └──────┬──────┘
            │                  │
            └────────┬────────┘
                     │
              ┌──────▼──────┐
              │    回答      │
              └─────────────┘
```

**核心アイデア**: F値をSystem切替の信頼度スコアとして使う

- F << 0 (例: F < -2.0): 高信頼即答 → System 1 (DG)
- F ≈ 0 (-2.0 < F < θ_ag): 低信頼 → System 2 (CoT)
- F >> 0 (F > θ_ag): 情報不足 → AG拡張 + System 2

---

## 2. 改善打ち手一覧

### Phase 1: パラメータ調整 (config変更のみ, 即日)

#### 打ち手 E1: γ₁=0.5, γ₀=0.3, θ_dg=-0.5

```yaml
# condition_e1.yaml
structural_mode: betti_full
gamma_0: 0.3    # β₀復活 (Comparison型向け)
gamma_1: 0.5    # β₁抑制 (DG早期発火防止)
theta_ag: 0.4   # 現状維持
theta_dg: -0.5  # DG閾値引き下げ
```

期待: DG発火率 73% → ~50%, EM +2~4pt

#### 打ち手 E2: γ₁=0.3, θ_dg=-0.3

```yaml
# condition_e2.yaml
structural_mode: betti_full
gamma_0: 0.3
gamma_1: 0.3    # さらに抑制
theta_dg: -0.3
```

#### 打ち手 E3: γ₁=0.5, θ_dg=0.0 (DG閾値は現状)

```yaml
# condition_e3.yaml
structural_mode: betti
gamma_0: 0.0
gamma_1: 0.5
theta_dg: 0.0
```

### Phase 2: Hybrid geDIG+CoT ★推奨 (半日実装)

#### 打ち手 H1: DG不発時 CoT フォールバック

```python
# adapter.py の process() を拡張

def process(self, example):
    # Step 1: 通常の geDIG パイプライン (System 1 判定)
    retrieved, g_now, extended_f, ag_fired, dg_fired = self._gedig_pipeline(example)

    # Step 2: System 切替
    if dg_fired and not ag_fired:
        # System 1: 高信頼即答 (73% of cases)
        answer = self.answerer.generate(example.question, context)
    else:
        # System 2: CoT推論フォールバック (27% of cases)
        answer = self._cot_fallback(example, retrieved, max_cot_steps=2)

    return result
```

**CoT フォールバック**: IRCoT の CoT ループを 2 ステップに制限

```python
def _cot_fallback(self, example, initial_facts, max_cot_steps=2):
    """System 2: 推論しながら追加検索"""
    context = [f.text for f in initial_facts]
    cot_chain = []

    for step in range(max_cot_steps):
        # 1. CoT文生成 (LLM call)
        cot_sentence = self._generate_cot_sentence(
            example.question, context, cot_chain
        )
        cot_chain.append(cot_sentence)

        # 2. 回答チェック
        if "answer is:" in cot_sentence.lower():
            return self._extract_answer(cot_sentence)

        # 3. CoT文でBM25追加検索 (no LLM call)
        new_facts = self.retriever.retrieve(cot_sentence, corpus, bm25, top_k=3)
        context = self._merge_context(context, new_facts)

    # 最終回答生成 (LLM call)
    return self.answerer.generate(example.question, context + cot_chain)
```

**LLM呼び出し回数**:
- System 1 (DG即答): 1回
- System 2 (CoT 2ステップ): 2~3回
- 加重平均: 0.73×1 + 0.27×3 = **1.5回/問** (IRCoTの1/5)

#### 打ち手 H2: F値ベース3段階切替

```python
def _decide_system(self, extended_f, ag_fired, dg_fired):
    """F値による3段階切替"""
    if extended_f < -2.0:
        return "system1_confident"  # 高信頼即答
    elif extended_f < -0.3:
        return "system2_verify"     # CoT検証1ステップ
    else:
        return "system2_explore"    # AG拡張 + CoT推論
```

### Phase 3: 大規模評価 (Phase 1-2 の最良設定で)

- 500問評価 + McNemar検定 / paired bootstrap
- 論文用結果テーブル

### Phase 4: 発展 (ストレッチ)

#### 打ち手 A1: Adaptive Gate (質問タイプ別閾値)

```python
# 質問の複雑度を軽量LLM呼び出しで推定
complexity = self._estimate_complexity(question)  # "simple" / "multi-hop"
if complexity == "simple":
    theta_dg = 0.0   # 即答OK
else:
    theta_dg = -1.0   # 探索必須
```

#### 打ち手 A2: Confidence-Weighted Ensemble

```python
# geDIG回答とCoT回答の両方を生成し、信頼度で選択
answer_sys1 = self.answerer.generate(question, context)
answer_sys2 = self._cot_fallback(example, retrieved)
confidence_sys1 = abs(extended_f)  # F値の絶対値 = 信頼度
return answer_sys1 if confidence_sys1 > threshold else answer_sys2
```

---

## 3. 実装計画

### ファイル変更一覧

| File | Change | Phase |
|------|--------|-------|
| `configs/condition_e1.yaml` | 新規: γ₁/γ₀/θ_dg 調整 | 1 |
| `configs/condition_e2.yaml` | 新規: 別パターン | 1 |
| `configs/condition_e3.yaml` | 新規: 別パターン | 1 |
| `src/adapter.py` | `_cot_fallback()` 追加 | 2 |
| `src/adapter.py` | `process()` にSystem切替ロジック | 2 |
| `src/adapter.py` | CoTプロンプト定義 | 2 |
| `configs/condition_hybrid.yaml` | 新規: hybrid mode config | 2 |
| `scripts/run_experiment.py` | hybrid mode 対応 | 2 |

### 評価指標

| 指標 | 現状 (B) | 目標 (Hybrid) | IRCoT参考 |
|------|:--------:|:------------:|:---------:|
| EM | 40.0% | **45~48%** | 46.0% |
| F1 | 0.570 | **0.63~0.67** | 0.637 |
| Avg LLM calls | 1.0 | **~1.5** | ~8.0 |
| Latency | 865ms | **~1,500ms** | 4,158ms |
| Cost (100Q) | $0.15 | **~$0.25** | $1.20 |

### Oracle 上限

| 組み合わせ | EM | F1 |
|-----------|:---:|:---:|
| geDIG-B 単体 | 40.0% | 0.570 |
| IRCoT 単体 | 46.0% | 0.637 |
| Oracle(B∪IRCoT) | **54.0%** | **0.713** |

→ 理論上限 EM=54%。Hybrid で半分拾えれば EM≈47%。

---

## 4. 思想的位置づけ

### 二重過程理論 (Dual Process Theory) との対応

| 概念 | System 1 | System 2 |
|------|----------|----------|
| 認知科学 | 直観・自動・速い | 分析・意識的・遅い |
| RAG | DG即答 | CoT推論+反復検索 |
| geDIG信号 | F << 0 (情報十分) | F ≈ 0 (情報不足) |
| コスト | 1 LLM call | 2-3 LLM calls |
| 強み | 効率性 | 正確性 |

### 従来手法との比較

```
BM25/GraphRAG:  System 1 only (常に即答)
IRCoT/ReAct:    System 2 only (常に推論)
geDIG v3:       System 1 + System 2 (トポロジーで切替)
                 ↑
                 「いつ深く考えるか」をグラフ構造が教えてくれる
```

### なぜトポロジーが切替信号として有効か

1. **β₁ (サイクル)**: 冗長情報の検出 → 「もう十分」の信号
2. **β₀ (連結成分)**: 情報の断絶検出 → 「まだ足りない」の信号
3. **F値**: 上記を統合した信頼度スコア → System切替の判断基準

人間も「情報が断片的で繋がらない」と感じたら直観を止めて分析的に考える。
geDIG の β₀ はまさにそれをグラフ理論的に検出している。
