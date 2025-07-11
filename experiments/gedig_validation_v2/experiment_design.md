# geDIG理論検証実験 v2.0 設計書

## 実験目的
InsightSpikeのgeDIG（Graph Edit Distance + Information Gain）理論に基づく内発報酬メカニズムが、従来のLLMおよびRAGと比較して質的に優れた洞察を生成することを実証する。

## 理論的背景
- **geDIG理論**: 𝓕 = w₁ ΔGED - kT ΔIG
- **情報熱力学的解釈**: 洞察生成時のエントロピー減少と構造簡潔化
- **内発報酬**: ΔGED ≤ -0.5 かつ ΔIG ≥ 0.2 のときEurekaSpike発生

## 実験設計

### 1. 比較手法
1. **Direct LLM**: DistilGPT-2による直接回答
2. **Standard RAG**: 単純な類似度ベースの検索拡張生成
3. **InsightSpike**: geDIG内発報酬による多相知識統合

### 2. 評価指標

#### A. 定量的指標（自動計算）
```python
metrics = {
    "delta_ig": float,      # 情報利得変化量 [bits]
    "delta_ged": float,     # グラフ編集距離変化量 [ノード数]
    "entropy_before": float, # 統合前エントロピー [bits]
    "entropy_after": float,  # 統合後エントロピー [bits]
    "graph_nodes": int,      # グラフノード数
    "graph_edges": int,      # グラフエッジ数
    "graph_density": float,  # グラフ密度
    "complexity_score": float, # 複雑度スコア
    "new_concepts": List[str], # 新規概念リスト
    "phase_integration": int,  # 統合フェーズ数
    "confidence": float,       # 信頼度スコア
    "processing_time": float   # 処理時間 [秒]
}
```

#### B. 質的指標（5段階評価）
```python
human_eval = {
    "novelty": int,       # 新規性 (1-5)
    "usefulness": int,    # 有用性 (1-5)
    "coherence": int,     # 一貫性 (1-5)
    "depth": int,         # 深さ (1-5)
    "integration": int    # 統合度 (1-5)
}
```

### 3. 知識ベース構成
```
Phase 1: Fundamental Concepts (基礎概念)
Phase 2: Mathematical Principles (数学原理)
Phase 3: Physical Theories (物理理論)
Phase 4: Biological Systems (生物システム)
Phase 5: Information Theory (情報理論)
```

### 4. 質問セット（拡張版）

#### カテゴリA: 単一ドメイン質問（ベースライン）
1. "What is entropy in thermodynamics?"
2. "Explain the concept of information in Shannon's theory"
3. "What are the principles of graph theory?"

#### カテゴリB: クロスドメイン質問（洞察期待）
4. "How does information relate to energy?"
5. "What connects biological evolution and information theory?"
6. "How do graph structures emerge in natural systems?"

#### カテゴリC: 抽象概念質問（高次洞察期待）
7. "What is the relationship between order and information?"
8. "How does complexity arise from simple rules?"
9. "What unifies discrete and continuous phenomena?"

### 5. 実験プロトコル

#### Phase 1: データ収集
```python
for question in questions:
    # 各手法で回答生成
    direct_result = direct_llm.generate(question)
    rag_result = standard_rag.generate(question)
    insight_result = insightspike.generate(question)
    
    # 詳細ログ記録
    log_entry = {
        "timestamp": datetime.now(),
        "question": question,
        "category": get_category(question),
        "results": {
            "direct": direct_result,
            "rag": rag_result,
            "insight": insight_result
        },
        "metrics": calculate_all_metrics(insight_result),
        "spike_detected": insight_result.spike_detected,
        "knowledge_sources": insight_result.sources
    }
```

#### Phase 2: 人間評価
```python
# ブラインド評価プロトコル
evaluators = recruit_evaluators(n=10)  # 10名の評価者
shuffled_responses = shuffle_and_anonymize(all_responses)

for evaluator in evaluators:
    for response in shuffled_responses:
        scores = evaluator.evaluate(response, criteria=EVAL_CRITERIA)
        record_human_evaluation(scores)
```

#### Phase 3: 統計分析
```python
# 1. 洞察検出率の比較
detection_rates = calculate_detection_rates(all_results)
fisher_test = fisher_exact_test(detection_rates)

# 2. 品質スコアの比較
quality_scores = aggregate_quality_scores(all_results)
anova_results = one_way_anova(quality_scores)
post_hoc = tukey_hsd(quality_scores)

# 3. 人間評価の一致度
icc_scores = calculate_icc(human_evaluations)
cohen_kappa = calculate_cohen_kappa(human_evaluations)

# 4. 効果量の算出
effect_sizes = {
    "direct_vs_insight": cohen_d(direct_scores, insight_scores),
    "rag_vs_insight": cohen_d(rag_scores, insight_scores)
}
```

### 6. アブレーション研究

#### A. 閾値感度分析
```python
thresholds = {
    "phase_count": [1, 2, 3, 4, 5],
    "similarity": [0.1, 0.2, 0.3, 0.4, 0.5],
    "confidence": [0.4, 0.5, 0.6, 0.7, 0.8]
}

for param, values in thresholds.items():
    sensitivity_results[param] = test_parameter_sensitivity(param, values)
```

#### B. コンポーネント分離
```python
ablation_configs = [
    {"name": "no_phase_integration", "disable": ["phase_filter"]},
    {"name": "no_confidence_filter", "disable": ["confidence_threshold"]},
    {"name": "no_spike_detection", "disable": ["spike_detector"]},
    {"name": "simple_rag_baseline", "disable": ["phase_filter", "spike_detector"]}
]

for config in ablation_configs:
    ablation_results[config["name"]] = run_ablation(config)
```

### 7. 可視化計画

#### A. 定量的結果の可視化
1. **性能比較バーチャート**: 手法別の平均品質スコア
2. **洞察検出率の比較**: 円グラフまたは積み上げ棒グラフ
3. **ΔIG-ΔGED散布図**: 洞察発生条件の可視化
4. **時系列プロット**: 質問順序による累積的変化

#### B. 構造変化の可視化
1. **知識グラフ進化図**: Before/Afterの並列表示
2. **エントロピー変化ヒートマップ**: フェーズ×質問のマトリクス
3. **新規概念ネットワーク**: 生成された概念の関係性

#### C. 質的分析の可視化
1. **人間評価レーダーチャート**: 5指標の比較
2. **ケーススタディ詳細図**: 代表的洞察の生成過程

### 8. 期待される成果

1. **主要仮説の検証**
   - InsightSpikeの洞察検出率 > 80%（他手法 < 20%）
   - 品質スコアで有意差（p < 0.01, Cohen's d > 1.5）
   - 人間評価でも有意な優位性

2. **理論的貢献**
   - geDIG理論の実証的裏付け
   - 情報熱力学的解釈の妥当性
   - 内発報酬メカニズムの有効性

3. **実用的示唆**
   - 最適なパラメータ設定の特定
   - スケーラビリティの検証
   - 応用領域の提案

## 実験スケジュール

1. **Week 1**: 実験環境構築・知識ベース拡張
2. **Week 2**: データ収集・自動評価
3. **Week 3**: 人間評価・統計分析
4. **Week 4**: 可視化・論文執筆

## 必要リソース

- 計算資源: GPU推奨（大規模実験時）
- 評価者: 10名（できれば専門知識を持つ）
- 時間: 約4週間

## 成功基準

1. 統計的有意性: p < 0.05 for all comparisons
2. 効果量: Cohen's d > 1.0 (large effect)
3. 人間評価一致度: ICC > 0.7
4. 再現性: 3回の独立実行で一貫した結果

---

この実験設計により、geDIG理論の妥当性とInsightSpikeの優位性を
科学的に厳密な方法で実証することが可能となる。