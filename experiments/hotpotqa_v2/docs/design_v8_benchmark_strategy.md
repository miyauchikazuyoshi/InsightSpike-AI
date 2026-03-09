# v8 ベンチマーク戦略 & geDIG-guided Iterative Retrieval 設計案

## 1. 問題提起: geDIGの gauge は「どこで」効くのか？

### FRAMES vs HotpotQA — gauge信号の比較

| 指標 | HotpotQA 500q (E1 mini) | FRAMES 100q (4o tk=15) |
|------|------------------------|------------------------|
| Overall EM | 45.2% | 29.0% |
| System1 EM | 34.6% | 29.8% |
| System2 EM | **51.8%** | 25.0% |
| **S2優位** | **+17.2pt** | **-4.8pt** |
| geDIG score (correct) | 0.33 | 0.50 |
| geDIG score (wrong) | 0.23 | 0.48 |
| **geDIG score差** | **+0.10** | **+0.02 (ほぼ無)** |

### HotpotQA question type別のSystem2優位

| Question Type | S1 EM | S2 EM | S2優位 |
|---------------|-------|-------|--------|
| bridge | 35.2% | 48.2% | +13.0pt |
| comparison | 31.2% | **66.1%** | **+34.9pt** |

### 結論

**geDIGのgauge（β₀/β₁/GED）はHotpotQAでは明確に機能するが、FRAMESでは機能しない。**

理由:
- **HotpotQA**: 10段落（2 gold + 8 distractor）→ グラフにgold/noiseの構造差がある → β₀がその差を検出
- **FRAMES**: 全記事がgold（Wikipedia gold linksから取得）→ グラフが均一に密 → トポロジー信号がノイズ
- **FRAMESのボトルネック**: 検索品質（BM25）が原理的限界。そもそも必要な情報がコーパスに無い（30%の誤答）

---

## 2. geDIGに最適なベンチマーク条件

geDIGの gauge が価値を発揮する条件:

1. **Distractor設定**: 正解に必要な情報と無関係な情報が混在 → β₀が信号/ノイズ分離に使える
2. **情報は存在する**: 検索品質ではなく、選別・構造化が課題
3. **Multi-hop**: グラフの接続性（β₀）やサイクル（β₁）が推論に影響
4. **Gold supporting facts annotated**: SF-F1で検索精度を定量評価可能
5. **Shortcut困難**: single-hopでは解けない（真にmulti-hop推論が必要）

### ベンチマーク適合度マトリクス

| ベンチマーク | Distractor | 情報存在 | Hops | SF注釈 | Shortcut困難 | サイズ | geDIG適合度 |
|------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **HotpotQA (distractor)** | ✅ 8段落 | ✅ | 2 | ✅ | △ | 7,405 | **A** |
| **MuSiQue** | ✅ 20段落 | ✅ | 2-4 | ✅ | **✅** (設計上) | 2,417 | **S** |
| **2WikiMultiHopQA** | ✅ | ✅ | 2-5 | ✅ | △ | 12,576 | **A** |
| HoVer | ✅ | ✅ | 2-4 | ✅ | △ | 4,000 | B |
| IIRC | △ (部分) | △ (要検索) | 1-3 | ✅ | ✅ | 1,300 | B+ |
| FRAMES | ❌ | ❌ (BM25限界) | 2-11 | ❌ | ✅ | 824 | **D** |
| Bamboogle | ❌ (open) | ❌ (要検索) | 2 | ❌ | ✅ | 125 | D |
| FanOutQA | △ | △ | fan-out | ❌ | ✅ | 1,000 | C |

### 最優先: MuSiQue

理由:
- **20段落のdistractor設定** → geDIGのβ₀フィルタリングが最大限活きる
- **2-4 hops** → graph connectivity分析が不可欠
- **Shortcut困難**: 設計上、disconnected reasoning（1段落だけ読んで答える）でF1が30pt低下
  - HotpotQA: shortcutでF1=68.8 → MuSiQue: shortcutでF1=37.8
- **SF注釈あり** → 検索精度を定量評価
- HuggingFace + GitHubで入手容易（StonyBrookNLP/musique）
- HotpotQAと同じJSONL形式に変換容易

---

## 3. geDIG-guided Iterative Retrieval 設計案

### 3a. 現行アーキテクチャの限界

```
現行: Q → BM25検索(1回) → グラフ構築 → gauge判定 → S1/S2 → 回答
                                        ↓
                                β₀高い → CoT(System2) → でも同じコンテキスト
```

System2は**推論**を深めるだけで、**情報**は増えない。
gauge信号は「情報が足りない」ことを検出するが、それを解消する手段がない。

### 3b. 提案: gauge-driven retrieval loop

```
提案: Q → 初回検索 → グラフ構築 → gauge判定 → β₀ > θ?
                                                  ↓ Yes
                                        connected components分析
                                        → 断絶箇所を特定
                                        → targeted query生成
                                        → 追加検索
                                        → グラフ更新
                                        → gauge再計算
                                        → β₀ ≤ θ? → 回答
```

### 3c. 実装ステップ

**Phase A: MuSiQue baseline** (今すぐ実行可能)
- MuSiQue dev setを入手 → HotpotQA形式に変換
- 現行E3Aで baseline EM/F1 計測
- System1 vs System2 のEM差を確認 → geDIG有効性の検証

**Phase B: Component Gap Query** (新規)
- `β₀ > 1` のとき、最大connected componentと2番目のcomponentのノードを取得
- それぞれの代表テキストからLLMで「この2つを繋ぐ情報は何か？」を生成
- 生成したクエリでBM25追加検索
- グラフに追加 → β₀再計算

**Phase C: Iterative Retrieval with convergence** (高度)
- β₀が閾値以下になるか、最大iteration数に達するまでループ
- 各iterationのGED scoreで「情報利得が十分か」を判定
- gauge-driven stopping criterion: `Δβ₀ == 0 かつ GED < ε` → 収束

### 3d. 論文ストーリー

> **Claim**: 知識グラフのBetti数を用いたadaptive retrieval。
> β₀（連結成分数）が情報ギャップを定量的に検出し、
> gaugefield理論に基づくtargeted retrievalを発火させる。
> 既存のIterative RAG（IRCoT, FLARE, Self-RAG）がLLMの
> 主観的判断に依存するのに対し、geDIGはトポロジーの
> 客観的シグナルで検索タイミングと対象を決定する。

差別化ポイント:
1. **検索タイミング**: β₀ > θ → 「断絶がある」→ 検索
2. **検索対象**: connected component分析 → 「何が足りないか」を特定
3. **停止判定**: GED score収束 → 「もう十分」を定量判断
4. **Self-RAGとの違い**: LLM依存の特殊トークン不要、グラフトポロジーで判断

---

## 4. 実験ロードマップ

### 短期 (今週)
1. MuSiQue devセットを入手・変換
2. 現行E3Aでbaseline計測（GPT-4o-mini / GPT-4o）
3. System1/2差、gauge信号品質を確認

### 中期 (2-3週)
4. Phase B: Component Gap Query実装
5. MuSiQue 100q / 500q で効果測定
6. HotpotQA fulldevでも同時検証

### 長期 (1-2ヶ月)
7. Phase C: 完全なIterative Retrieval loop
8. 複数ベンチマーク横断評価（MuSiQue + HotpotQA + 2WikiMultiHop）
9. 論文執筆

---

## 5. FRAMES実験のまとめ

| Config | EM | F1 | Lenient EM | Note |
|--------|------|------|-----------|------|
| mini tk=5 | 21.2% | 0.310 | — | |
| 4o tk=5 | 26.0% | 0.387 | 42.0% | |
| 4o tk=15 | **29.0%** | 0.407 | **43.0%** | best strict EM |
| 4o tk=30 | 29.0% | **0.443** | 41.0% | best F1 |
| 4o tk=15 v2(150cap) | 26.0% | 0.389 | 38.0% | more text = more noise |
| 4o tk=15 + decompose | 27.0% | 0.384 | 41.0% | v7 negative result |
| FRAMES paper (no retrieval) | 40.0% | — | — | |
| FRAMES paper (multi-step RAG) | 66.0% | — | — | iterative web search |

**学び**: FRAMESはretrieval品質テスト。geDIGの土壌ではない。
geDIGの強みは「情報がある中での構造分析」であり、MuSiQueのdistractor設定が最適。
