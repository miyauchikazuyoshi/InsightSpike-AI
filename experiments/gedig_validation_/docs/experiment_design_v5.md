# geDIG理論検証実験 v5.0 設計書（最終版）

## 変更点（v2からの主要アップデート）

### 1. **プロンプトビルダーの革新的改良**
- GNNメッセージパッシングで生成された洞察を自然言語化
- 統合された理解を明示的にプロンプトに含める
- スパイク検出時の特別な洞察表現

### 2. **低機能LLMの戦略的活用**
- DistilGPT-2（82M）による高速実験（TinyLlamaの20倍速）
- LLM品質がRAG性能評価に影響しないことを実証
- 洞察生成はGNN、LLMは成文化のみという明確な役割分担

### 3. **実験の効率化**
- GPU不要、CPU環境で実行可能
- 1実験あたり5分以内で完了
- リアルタイムでの反復改善が可能

## 実験目的（強化版）

1. **主目的**: InsightSpikeのGNN-based洞察生成メカニズムが、従来手法を超える価値を創出することを実証
2. **副目的**: 低品質LLMでも洞察表現が可能であることを示し、システムの堅牢性を証明

## 理論的背景（更新版）

### geDIG理論の実装
- **𝓕 = w₁ ΔGED - kT ΔIG**
- **メッセージパッシング**: GCN 3層によるエピソード間の情報統合
- **洞察の明示化**: プロンプトビルダーによる自然言語変換

## 実験設計 2.0

### 1. 比較手法（改良版）

```python
configurations = {
    "direct_llm": {
        "llm": "DistilGPT-2",
        "context": None,
        "insights": None
    },
    "standard_rag": {
        "llm": "DistilGPT-2",
        "context": "retrieved_documents",
        "insights": None
    },
    "insightspike": {
        "llm": "DistilGPT-2",
        "context": "retrieved_documents",
        "insights": "gnn_processed_insights"  # 新機能
    }
}
```

### 2. 評価指標（拡張版）

#### A. 自動計算指標
```python
metrics = {
    # 従来指標
    "delta_ig": float,
    "delta_ged": float,
    "spike_detected": bool,
    
    # 新規追加指標
    "insight_expression_score": float,  # 洞察の言語化品質
    "concept_integration_level": int,   # 統合された概念数
    "cross_domain_connections": int,    # ドメイン横断接続数
    "prompt_coherence": float          # プロンプトの一貫性
}
```

#### B. 簡易人間評価（3段階）
```python
human_eval = {
    "insight_present": bool,     # 洞察が含まれているか
    "beyond_retrieval": bool,    # 単純検索を超えているか
    "novel_connection": bool     # 新しい接続が示されているか
}
```

### 3. 知識ベース（コンパクト版）

```json
{
  "domains": {
    "thermodynamics": [
      "Entropy measures disorder in systems",
      "Second law: entropy always increases",
      "Energy cannot be created or destroyed"
    ],
    "information": [
      "Shannon entropy quantifies uncertainty",
      "Information requires energy to process",
      "Landauer's principle links erasure to heat"
    ],
    "biology": [
      "Living systems maintain low entropy",
      "DNA stores hereditary information",
      "Metabolism exports entropy"
    ]
  }
}
```

### 4. 質問セット（厳選版）

```python
questions = [
    # ベースライン
    {"id": "Q1", "text": "What is entropy?", "type": "factual"},
    
    # クロスドメイン
    {"id": "Q2", "text": "How does life violate the second law?", "type": "insight_required"},
    
    # 抽象概念
    {"id": "Q3", "text": "What connects information and energy?", "type": "abstract"}
]
```

### 5. 実験プロトコル（効率化版）

```python
class EfficientExperiment:
    def run(self):
        # 1. 初期化（1回のみ）
        llm = DistilGPT2Provider()
        enhanced_builder = EnhancedPromptBuilder()
        
        # 2. 各構成で実行
        for config_name, config in configurations.items():
            agent = create_agent(config)
            
            # 知識ロード（RAG/InsightSpikeのみ）
            if config["context"]:
                agent.load_knowledge(knowledge_base)
            
            # 質問実行
            for question in questions:
                result = agent.process_question(question)
                
                # 洞察の明示化（InsightSpikeのみ）
                if config["insights"]:
                    result["prompt"] = enhanced_builder.build_with_insights(
                        result["context"], question
                    )
                
                record_result(config_name, question, result)
```

### 6. 新機能：洞察の可視化

```python
def visualize_insight_generation(result):
    """洞察生成プロセスの可視化"""
    
    # 1. 元のエピソード
    plot_original_episodes(result["retrieved_docs"])
    
    # 2. GNN処理後のグラフ
    plot_message_passing_graph(result["graph_features"])
    
    # 3. 統合された洞察
    display_integrated_insights(result["insights"])
    
    # 4. 最終応答
    show_final_response(result["response"])
```

### 7. 期待される成果（現実的版）

1. **定量的成果**
   - InsightSpikeでスパイク検出率 > 30%（他手法 0%）
   - 応答長の増加（洞察を含むため）
   - クロスドメイン質問での優位性

2. **定性的成果**
   - プロンプトに洞察が明示的に含まれる
   - 低品質LLMでも洞察の存在が確認可能
   - 単純検索を超えた価値の創出

3. **技術的貢献**
   - GNN + プロンプトエンジニアリングの新手法
   - LLM品質に依存しない洞察生成の実証
   - 解釈可能なAIシステムの実例

## 実装の要点

### 1. EnhancedPromptBuilder
```python
class EnhancedPromptBuilder:
    def build_with_insights(self, context, question):
        # GNN処理結果から洞察を抽出
        insights = self.extract_insights_from_gnn(context["graph_features"])
        
        # 自然言語化
        insight_text = self.generate_insight_text(insights)
        
        # 構造化プロンプト生成
        return f"""
        ## 発見された洞察
        {insight_text}
        
        ## 根拠となる知識
        {self.format_evidence(context["documents"])}
        
        ## 質問
        {question}
        """
```

### 2. 実験の自動化
```python
# 全実験を自動実行
experiment = GedigV5Experiment()
results = experiment.run_all_configurations()
report = experiment.generate_report(results)
```

## 成功基準（実用的版）

1. **技術的成功**
   - 3構成すべてが正常動作
   - 洞察がプロンプトに含まれる
   - 5分以内に実験完了

2. **科学的成功**
   - InsightSpikeで明確な差別化
   - 洞察の存在が応答に反映
   - 再現可能な結果

## スケジュール（即日実行可能）

1. **30分**: 環境セットアップ
2. **1時間**: 実験実行
3. **30分**: 結果分析・可視化
4. **1時間**: レポート作成

## まとめ

v5実験設計は、プロンプトビルダーの革新と低機能LLMの活用により、geDIG理論の本質を効率的に実証します。GNNによる洞察生成とLLMによる成文化の役割分担を明確にし、InsightSpikeの真の価値を示します。