# Question-Answer実験 再設計案

## 新しい実験コンセプト：最小解選定 vs LLM回答

### 概要
スパイク検出にこだわらず、**geDIGが選ぶ最小解（最適な知識の組み合わせ）とLLMの直接回答を比較**する実験に転換。

### 実験の目的
1. **geDIGの知識選択能力を評価**
   - 質問に対して最適な知識の組み合わせを選べるか
   - 不要な知識を除外できるか

2. **LLMとの比較**
   - LLM単体の回答
   - geDIGが選んだ知識を使ったLLMの回答
   - どちらがより的確か

### 実験設計

#### 1. データセット（既存を活用）
- **知識ベース**: 500エントリ（変更なし）
- **質問セット**: 100問（変更なし）
  - Easy (30): 単一知識で回答可能
  - Medium (40): 2-3知識の統合が必要
  - Hard (20): 複数知識の創造的統合
  - Very Hard (10): 分野横断的な知識統合

#### 2. 評価プロセス
```
質問 → geDIG最小解探索 → 選ばれた知識でLLM回答
  ↓
比較
  ↓  
質問 → LLM直接回答（知識ベースなし）
```

#### 3. 評価メトリクス

##### 定量的評価
1. **知識選択の精度**
   - Precision: 選ばれた知識のうち有用なものの割合
   - Recall: 必要な知識のうちカバーできた割合
   - F1スコア

2. **効率性**
   - 選択された知識数（少ないほど良い）
   - 処理時間

3. **回答品質**（人間評価またはLLM評価）
   - 正確性（0-5）
   - 完全性（0-5）
   - 簡潔性（0-5）

##### 定性的評価
1. **知識選択の妥当性**
   - なぜその知識が選ばれたか
   - 不要な知識を除外できているか

2. **創発的な組み合わせ**
   - 予想外だが有効な知識の組み合わせ
   - 分野横断的な統合

### 実装変更点

#### 1. geDIG最小解探索の実装
```python
def find_minimal_solution(question: str, knowledge_base: List[Knowledge]) -> List[Knowledge]:
    """
    geDIGを使って質問に対する最小限の知識セットを見つける
    
    目的関数: F = w1*|K| + w2*relevance_score
    - |K|: 選択された知識数（少ないほど良い）
    - relevance_score: 質問との関連性スコア
    """
    # 1. 質問をベクトル化
    query_vector = embed(question)
    
    # 2. 各知識の関連性スコアを計算
    relevance_scores = compute_relevance(query_vector, knowledge_base)
    
    # 3. geDIG最適化で最小セットを選択
    selected_knowledge = optimize_knowledge_selection(
        knowledge_base, 
        relevance_scores,
        min_knowledge=1,
        max_knowledge=5
    )
    
    return selected_knowledge
```

#### 2. 比較実験の実装
```python
def compare_approaches(question: str, knowledge_base: List[Knowledge]):
    # 1. geDIG最小解アプローチ
    selected_knowledge = find_minimal_solution(question, knowledge_base)
    gedig_answer = llm_answer_with_knowledge(question, selected_knowledge)
    
    # 2. LLM直接アプローチ
    direct_answer = llm_answer_direct(question)
    
    # 3. 評価
    return {
        'question': question,
        'selected_knowledge_count': len(selected_knowledge),
        'selected_knowledge': [k.id for k in selected_knowledge],
        'gedig_answer': gedig_answer,
        'direct_answer': direct_answer,
        'evaluation': evaluate_answers(gedig_answer, direct_answer)
    }
```

### 期待される結果

1. **Easy問題（30問）**
   - geDIGは1-2個の知識を正確に選択
   - LLM直接回答と同等以上の品質

2. **Medium問題（40問）**
   - geDIGは2-3個の最適な知識を選択
   - 不要な知識を除外し、効率的な回答

3. **Hard/Very Hard問題（30問）**
   - geDIGが予想外の有効な組み合わせを発見
   - LLM単体では思いつかない洞察

### 成功基準

1. **知識選択の効率性**
   - 平均選択知識数 < 3（必要最小限）
   - F1スコア > 0.7

2. **回答品質**
   - geDIG+LLM ≥ LLM直接（70%以上のケース）
   - 特にMedium以上の問題で優位性

3. **創発的発見**
   - 5%以上のケースで予想外の有効な組み合わせ

### 実験のアピールポイント

1. **実用的価値**
   - RAGシステムの知識選択最適化
   - 大規模知識ベースからの効率的な情報抽出

2. **理論的貢献**
   - geDIGによる組合せ最適化の実証
   - 最小解探索の有効性

3. **明確な比較結果**
   - LLM単体 vs geDIG+LLM
   - 定量的・定性的な優位性の証明

この再設計により、スパイク検出に依存せず、**geDIGの本質的な価値（最適な知識の組み合わせ探索）を実証**できます。