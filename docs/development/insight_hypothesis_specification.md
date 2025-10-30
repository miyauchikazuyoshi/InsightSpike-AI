---
status: active
category: insight
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# Insight Hypothesis Specification
# 洞察仮説システム仕様書

## 1. Overview / 概要

The Insight Hypothesis System treats insights discovered by InsightSpike as testable hypotheses rather than absolute truths. Each hypothesis starts with a low confidence value (C-value) that evolves based on usage and validation.

InsightSpikeが発見した洞察を、絶対的な真実ではなく検証可能な仮説として扱うシステム。各仮説は低い信頼度（C値）から始まり、使用と検証に基づいて進化する。

## 2. Core Concepts / 核心概念

### 2.1 Insight Hypothesis / 洞察仮説
- One-sentence statement capturing a discovered relationship
- Stored as an Episode with `episode_type = HYPOTHESIS`
- Initial C-value: 0.3 (low confidence)
- 発見された関係性を表す一文
- `episode_type = HYPOTHESIS`のEpisodeとして保存
- 初期C値: 0.3（低信頼度）

### 2.2 C-value Dynamics / C値の動態
```python
# Support case / 支持された場合
new_c = min(0.95, current_c + (1 - current_c) * 0.1)

# Contradiction case / 矛盾した場合  
new_c = max(0.05, current_c * 0.9)

# Refinement case / 改良された場合
new_c = max(0.3, current_c * 0.9)
```

### 2.3 Episode Types / エピソードタイプ
```python
class EpisodeType(Enum):
    FACT = "fact"                # Regular knowledge (high C-value)
    HYPOTHESIS = "hypothesis"    # Insight hypothesis (low initial C-value)
    QUESTION = "question"        # User question
    REASONING = "reasoning"      # Supporting reasoning for hypothesis
```

## 3. Data Structure / データ構造

### 3.1 HypothesisEpisode
```python
@dataclass
class HypothesisEpisode(Episode):
    # Core Episode fields
    text: str                    # Hypothesis text
    vec: np.ndarray             # Embedding vector
    c: float                    # Confidence value (0.0-1.0)
    timestamp: float            # Creation time
    
    # Hypothesis-specific fields
    episode_type: EpisodeType
    hypothesis_id: Optional[str]
    parent_hypothesis_ids: List[str]      # Evolution tracking
    support_episodes: List[str]           # Supporting evidence
    contradiction_episodes: List[str]     # Contradicting evidence
```

### 3.2 Metadata Structure
```python
metadata = {
    'original_question': str,        # Question that generated this
    'reasoning_id': str,            # Linked reasoning episode
    'usage_history': [              # Usage tracking
        {
            'timestamp': float,
            'context': str,
            'outcome': str,         # 'supported'/'contradicted'/'refined'
            'c_before': float,
            'c_after': float
        }
    ],
    'update_count': int             # Number of updates
}
```

## 4. System Architecture / システムアーキテクチャ

### 4.1 Hypothesis Lifecycle / 仮説のライフサイクル
```
1. Question asked / 質問
   ↓
2. Spike detected / スパイク検出
   ↓
3. Hypothesis created (C=0.3) / 仮説生成（C=0.3）
   ↓
4. Usage & validation / 使用と検証
   ├→ Supported: C increases / 支持: C値上昇
   ├→ Contradicted: C decreases / 矛盾: C値低下
   └→ Refined: New hypothesis created / 改良: 新仮説生成
```

### 4.2 Prompt Integration / プロンプト統合

#### Standard Format
```
Question: {question}

Facts:
{facts}

Relevant hypotheses from previous analyses:

Hypothesis 1 (confidence: 85%):
HYPOTHESIS: {hypothesis_text}
RATIONALE: {reasoning_excerpt}
EVIDENCE: Supported 5 times, contradicted 1 time
CONTEXT: Originally from "{original_question}"

Task: Respond in this exact format:
CORE INSIGHT: [One sentence hypothesis]
SUPPORTING ANALYSIS: [Detailed explanation]
```

### 4.3 Retrieval Algorithm / 検索アルゴリズム
```python
def retrieve_hypotheses(query: str, min_c: float = 0.2) -> List[HypothesisEpisode]:
    # 1. Filter by minimum C-value
    candidates = [h for h in hypotheses if h.c >= min_c]
    
    # 2. Calculate weighted score
    for hypothesis in candidates:
        similarity = cosine_similarity(query_vec, hypothesis.vec)
        weighted_score = similarity * hypothesis.c
    
    # 3. Return top-k by weighted score
    return sorted(candidates, key=lambda h: h.weighted_score)[:k]
```

## 5. C-value Update Rules / C値更新ルール

### 5.1 Support Event / 支持イベント
- Occurs when hypothesis helps answer correctly
- C-value increase with diminishing returns
- Maximum C-value: 0.95
- 仮説が正しい回答に貢献した場合
- 収穫逓減でC値増加
- 最大C値: 0.95

### 5.2 Contradiction Event / 矛盾イベント
- Occurs when hypothesis conflicts with new evidence
- C-value decreases by 10%
- Minimum C-value: 0.05
- 新しい証拠と矛盾した場合
- C値を10%減少
- 最小C値: 0.05

### 5.3 Refinement Event / 改良イベント
- Parent hypothesis C decreases slightly
- Child hypothesis inherits boosted C-value
- Boost factor: 1.2x parent's C-value
- 親仮説のC値をわずかに減少
- 子仮説は向上したC値を継承
- ブースト係数: 親のC値の1.2倍

## 6. Implementation Guidelines / 実装ガイドライン

### 6.1 Creating Hypothesis
```python
def create_hypothesis_episode(question: str, 
                            hypothesis: str, 
                            reasoning: str) -> str:
    # 1. Generate embeddings
    hyp_embedding = embedder.encode(hypothesis)
    
    # 2. Create episode with low C-value
    episode = HypothesisEpisode(
        text=hypothesis,
        vec=hyp_embedding,
        c=0.3,  # Low initial confidence
        episode_type=EpisodeType.HYPOTHESIS
    )
    
    # 3. Store and return ID
    return store_episode(episode)
```

### 6.2 Using Hypothesis
```python
def use_hypothesis(hyp_id: str, outcome: str):
    hypothesis = get_episode(hyp_id)
    old_c = hypothesis.c
    
    # Update C-value based on outcome
    hypothesis.update_c_from_usage(outcome)
    
    # Log usage
    hypothesis.metadata['usage_history'].append({
        'timestamp': time.time(),
        'outcome': outcome,
        'c_before': old_c,
        'c_after': hypothesis.c
    })
```

## 7. Experiment Design / 実験設計

### 7.1 Scale Testing / スケールテスト
- 1000 episodes (mixed types)
- 100 questions
- Track C-value evolution
- Measure retrieval accuracy

### 7.2 Metrics / 評価指標
```python
metrics = {
    'c_value_progression': [],      # C値の推移
    'hypothesis_reuse_rate': 0.0,   # 仮説再利用率
    'refinement_rate': 0.0,         # 改良率
    'contradiction_rate': 0.0,      # 矛盾率
    'retrieval_precision': 0.0,     # 検索精度
    'average_final_c': 0.0          # 最終平均C値
}
```

### 7.3 Expected Outcomes / 期待される結果
- Good hypotheses converge to C > 0.7
- Bad hypotheses drop below C < 0.2
- Refined hypotheses outperform originals
- Retrieval improves over time
- 良い仮説はC > 0.7に収束
- 悪い仮説はC < 0.2に低下
- 改良仮説が元仮説を上回る
- 時間とともに検索精度向上

## 8. Integration Points / 統合ポイント

### 8.1 With Existing Episode System
- Extends core Episode class
- Compatible with existing memory systems
- Uses same embedding infrastructure

### 8.2 With Graph Search
- C-value influences edge weights
- High-C hypotheses form stronger connections
- Low-C hypotheses naturally pruned

### 8.3 With Prompt Builder
- Includes both hypothesis and rationale
- Shows confidence and evidence
- Limits to top 3 relevant hypotheses

## 9. Future Extensions / 将来の拡張

### 9.1 Hypothesis Merging
- Combine similar high-C hypotheses
- Create meta-hypotheses
- Track lineage

### 9.2 Contradiction Analysis
- Learn from contradictions
- Generate counter-hypotheses
- Build nuanced understanding

### 9.3 Domain Adaptation
- Different C-value dynamics per domain
- Transfer learning between domains
- Specialized hypothesis stores

## 10. Similarity-Based C-value Updates / 類似度ベースのC値更新

### 10.1 Semantic Similarity Thresholds / 意味的類似度の閾値

Based on experimental results with real embeddings (all-MiniLM-L6-v2):

```
Similarity Range  | Interpretation        | Example
----------------- | -------------------- | -------
0.98 - 1.00      | Identical/variations | "Energy relates to information" vs "energy relates to information"
0.70 - 0.90      | Semantic paraphrase  | "Energy and information are connected" vs "Information requires energy"
0.50 - 0.70      | Related concept      | Same domain but different aspects
0.30 - 0.50      | Weak relationship    | Different domains with minor overlap
0.00 - 0.30      | Unrelated           | Independent concepts
```

### 10.2 C-value Update Rules by Similarity / 類似度によるC値更新ルール

#### Query Match (New question matches hypothesis)
```python
if similarity > 0.9:
    boost = 0.15  # Almost identical - strong confirmation
elif similarity > 0.7:
    boost = 0.10  # Semantic paraphrase - moderate confirmation  
elif similarity > 0.5:
    boost = 0.05  # Somewhat related - slight confirmation
else:
    boost = 0.02  # Weak match - minimal boost

new_c = min(0.95, current_c + (1 - current_c) * boost)
```

#### Hypothesis Conflict (Contradicting hypotheses)
```python
if similarity > 0.8:
    penalty = 0.3   # Very similar but conflicting - major issue
elif similarity > 0.6:
    penalty = 0.15  # Related but conflicting - moderate issue
else:
    penalty = 0.05  # Different domains - minor issue

new_c = max(0.05, current_c * (1 - penalty))
```

#### Hypothesis Support (Mutually reinforcing)
```python
if similarity > 0.7:
    boost = 0.08   # Strong semantic support
else:
    boost = 0.03   # Weak support

new_c = min(0.95, current_c + (1 - current_c) * boost)
```

### 10.3 Initial C-value Assignment / 初期C値の割り当て

When creating a new hypothesis, check similarity to existing hypotheses:

```python
def calculate_initial_c(hypothesis: str, existing_hypotheses: List[HypothesisEpisode]) -> float:
    max_similarity = 0.0
    conflicting_c = 0.0
    
    for existing in existing_hypotheses:
        similarity = cosine_similarity(hypothesis_vec, existing.vec)
        
        if similarity > max_similarity:
            max_similarity = similarity
            if is_conflicting(hypothesis, existing):
                conflicting_c = existing.c
    
    # Determine initial C-value
    if conflicting_c > 0:
        return 0.15  # Conflicts with existing - start low
    elif max_similarity > 0.8:
        return 0.4   # Very similar to existing - inherit some confidence
    elif max_similarity > 0.5:
        return 0.35  # Somewhat related - slight boost
    else:
        return 0.3   # Novel hypothesis - default
```

### 10.4 Advantages of Similarity-Based System / 類似度ベースシステムの利点

1. **Semantic Clustering**: Similar insights naturally cluster together
2. **Duplicate Prevention**: Near-identical hypotheses don't artificially inflate confidence
3. **Novelty Detection**: Truly novel insights are identified and tracked
4. **Conflict Resolution**: Semantic similarity determines conflict severity
5. **Knowledge Graph**: C-values and similarities form natural knowledge graph

### 10.5 Implementation Notes / 実装上の注意

- Use cached embeddings for efficiency
- Store similarity matrix for frequently accessed hypotheses
- Consider approximate nearest neighbor search for scale
- Log similarity scores with C-value updates for analysis

### 10.6 Experimental Validation / 実験的検証

Results from similarity testing:
- Same insight variations: 0.985-0.990 similarity
- Semantic paraphrases: 0.7-0.8 similarity  
- Unrelated statements: <0.3 similarity
- Discrimination ratio: 5-11x between related and unrelated

These thresholds have been validated through experiments with real LLM-generated insights and embeddings.

---

**Version**: 1.1  
**Date**: 2025-01-25  
**Status**: Ready for Implementation
**Latest Update**: Added Section 10 - Similarity-Based C-value Updates