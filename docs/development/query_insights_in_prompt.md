# Query Transformation Insights in Prompts

## 概要

Query Transformationで発見された洞察を、LLMへのプロンプトに含めるように修正しました。これにより、メッセージパッシングで創発した洞察がLLMの応答生成に活用されます。

## 修正内容

### 1. Layer4 LLM Interface の拡張

`layer4_llm_interface.py` の `_build_prompt` メソッドを拡張：

```python
# Query Transformationの洞察を含める
query_state = context.get("query_state")
if query_state and hasattr(query_state, "insights_discovered") and query_state.insights_discovered:
    context_parts.append("\n[Discovered Insights from Query Evolution]")
    for insight in query_state.insights_discovered[:3]:
        context_parts.append(f"- {insight}")

if query_state and hasattr(query_state, "absorbed_concepts") and query_state.absorbed_concepts:
    context_parts.append("\n[Key Concepts Absorbed]")
    context_parts.append(f"- {', '.join(query_state.absorbed_concepts[:5])}")
```

### 2. 簡潔プロンプトモードでの対応

軽量モデル（GPT-2など）向けの簡潔プロンプトでも、最も重要な洞察を含めるように対応：

```python
# 簡潔モードでは最初の洞察のみを短縮して含める
if query_state and hasattr(query_state, "insights_discovered") and query_state.insights_discovered:
    context_parts.append(f"[{query_state.insights_discovered[0][:50]}]")
```

## プロンプトの例

### 標準モード（詳細）
```
Retrieved Information:
1. Entropy measures disorder in thermodynamics (relevance: 0.92)
2. Shannon entropy quantifies information content (relevance: 0.88)

[Discovered Insights from Query Evolution]
- Thermodynamics and information theory share mathematical structure
- Entropy bridges physical and informational domains
- Maxwell's demon connects computation and physics

[Key Concepts Absorbed]
- entropy, information, reversibility, computation, energy

Question: How are entropy concepts related?

Answer:
```

### 簡潔モード（GPT-2向け）
```
Context: Entropy measures disorder... Shannon entropy quantifies... [Entropy connects physics and information theory fund]
Q: How are entropy concepts related?
A:
```

## 利点

1. **洞察の活用**: Query Transformationで発見された洞察がLLMに伝達される
2. **文脈の豊富化**: 吸収された概念により、より深い理解に基づく応答が可能
3. **説明可能性**: どのような洞察から応答が生成されたかが明確
4. **柔軟性**: 軽量モデルでも重要な洞察は保持

## 使用方法

Query Transformation機能を有効にしてエージェントを使用：

```python
from insightspike.implementations.agents import ConfigurableAgent

agent = ConfigurableAgent(
    enable_query_transform=True,
    config=config
)

result = agent.process_question("How are entropy concepts related?")
```

## 技術的詳細

- **制限**: 洞察は最大3つ、概念は最大5つまでプロンプトに含める
- **互換性**: query_stateがない場合も正常に動作
- **軽量対応**: 簡潔モードでは50文字に切り詰め

## まとめ

この修正により、Query Transformationの真の価値が発揮されます。メッセージパッシングで創発した洞察が、単なる内部状態ではなく、実際のLLM応答生成に活用されるようになりました。