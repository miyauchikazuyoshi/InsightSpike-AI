# InsightSpike Layer4 プロンプト生成フロー

## プロンプト生成の流れ

### 1. MainAgent → L4LLMProvider
```python
# main_agent.py (line 282-287)
llm_result = self.l4_llm.generate_response_detailed(llm_context, question)
# または
response = self.l4_llm.generate_response(llm_context, question)
```

### 2. L4LLMProvider.generate_response()
```python
# layer4_llm_provider.py (line 49-52)
# PromptBuilderを使ってプロンプトを構築
prompt = self.prompt_builder.build_prompt(
    {"context": context_str, "reasoning_quality": reasoning_quality},
    question,
)
```

### 3. L4PromptBuilder.build_prompt()
```python
# layer4_prompt_builder.py
# 以下の要素を組み合わせてプロンプトを構築：
sections = []
sections.append(self._build_system_instruction())  # システム指示
sections.append(self._build_document_context(documents))  # ドキュメント
sections.append(self._build_reasoning_context(graph_info))  # 推論情報
sections.append(self._build_previous_context(previous_state))  # 前の状態
sections.append(self._build_question_section(question, reasoning_quality))  # 質問
return "\n\n".join(sections)
```

### 4. L4LLMProvider._generate_sync()
```python
# layer4_llm_provider.py (line 520)
formatted_prompt = self._format_prompt(prompt)  # ここで特殊トークンを追加！
```

### 5. L4LLMProvider._format_prompt()
```python
# layer4_llm_provider.py (line 544-553)
return f"""<|system|>
You are a helpful AI assistant. Answer the question based on the provided context.

<|user|>
{prompt}

<|assistant|>
"""
```

## 具体的な指示内容

### システム指示 (_build_system_instruction)
```
You are an advanced AI assistant specialized in analytical reasoning and insight generation. 

Your role is to:
1. Analyze provided documents and context carefully
2. Identify key patterns, connections, and insights
3. Provide well-reasoned answers based on evidence
4. Acknowledge uncertainty when information is insufficient
5. Highlight novel insights or "spikes of understanding" when they emerge

Always base your responses on the provided context and clearly distinguish between what the evidence supports versus speculative reasoning.
```

### ドキュメントコンテキスト (_build_document_context)
```
## Retrieved Context Documents
The following documents are relevant to your query:

### Document 1 🟢 (High Confidence)
**Relevance:** 0.900 | **Confidence:** 0.800
Energy is the capacity to do work.

### Document 2 🟡 (Medium Confidence)
...
```

### 推論状態 (_build_reasoning_context)
```
## Current Reasoning State
**Graph Analysis Metrics:**
- ΔGED (Graph Edit Distance Change): -0.150
- ΔIG (Information Gain Change): 0.250
- Conflict Level: 0.123

🧠 **INSIGHT SPIKE DETECTED** - This query may represent a significant improvement in understanding!
```

### 質問セクション (_build_question_section)
```
## User Question
"What is energy?"

## Instructions
High reasoning quality detected. Provide a comprehensive, well-structured answer.

Please:
1. Synthesize information from the provided context
2. Highlight key insights and connections
3. Indicate confidence levels in your reasoning
4. Note any novel patterns or 'insight spikes' you detect
5. Provide a clear, actionable answer
```

## 問題点

1. **複雑すぎる**: 1000文字以上の複雑な指示
2. **特殊トークン**: DistilGPT2が理解できない`<|system|>`, `<|user|>`, `<|assistant|>`
3. **英語のみ**: すべての指示が英語
4. **メタデータ過多**: ΔGED、ΔIG、Confidence値など、モデルには不要な情報

## 結論

Layer4は非常に高度なプロンプトエンジニアリングを行っていますが、DistilGPT2のような小さなモデルには不適切です。これらの指示は、GPT-4やClaude 3のような大規模モデル向けに設計されているようです。