# プロンプトフォーマットの決定場所

## 結論: **src以下で決定されている**

### プロンプトフォーマットの決定フロー：

```
実験スクリプト
    ↓ (1) agent.process_question()を呼ぶだけ
MainAgent 
    ↓ (2) llm_contextを作成してl4_llm.generate_response_detailed()に渡す
L4LLMProvider
    ↓ (3) self.prompt_builder.build_prompt()でプロンプトを構築
L4PromptBuilder 
    ↓ (4) 複雑な英語プロンプトを構築
L4LLMProvider._generate_sync()
    ↓ (5) self._format_prompt()で特殊トークンを追加
最終プロンプト
```

## 詳細：

### 1. 実験スクリプト (`run_comparison_experiment.py`)
```python
# プロンプトフォーマットには一切関与しない
result = agent.process_question(
    question,
    max_cycles=3,
    verbose=True
)
```

### 2. MainAgent (`main_agent.py`)
```python
# コンテキストを準備するだけ、フォーマットは指定しない
llm_context = {
    "retrieved_documents": retrieved_docs,
    "graph_analysis": graph_analysis,
    "previous_state": self.previous_state,
    "reasoning_quality": graph_analysis.get("reasoning_quality", 0.0),
}
llm_result = self.l4_llm.generate_response_detailed(llm_context, question)
```

### 3. L4LLMProvider (`layer4_llm_provider.py`)
```python
# PromptBuilderに委譲
prompt = self.prompt_builder.build_prompt(
    {"context": context_str, "reasoning_quality": reasoning_quality},
    question,
)
# その後、_format_promptで特殊トークンを追加
formatted_prompt = self._format_prompt(prompt)
```

### 4. L4PromptBuilder (`layer4_prompt_builder.py`)
```python
# ここで英語の複雑なプロンプトを構築
sections = []
sections.append(self._build_system_instruction())  # "You are an advanced AI assistant..."
sections.append(self._build_document_context(documents))  # "## Retrieved Context Documents..."
sections.append(self._build_reasoning_context(graph_info))  # "ΔGED: -0.150..."
sections.append(self._build_question_section(question, reasoning_quality))  # "Please: 1. Synthesize..."
```

### 5. 特殊トークンの追加 (`layer4_llm_provider.py`)
```python
def _format_prompt(self, prompt: str) -> str:
    return f"""<|system|>
You are a helpful AI assistant. Answer the question based on the provided context.

<|user|>
{prompt}

<|assistant|>
"""
```

## 実験スクリプトでできること：

### オプション1: Configで制御（現状なし）
```python
# 現在のConfigには、プロンプトフォーマットを制御する設定がない
config = Config()
# config.prompt_format = "simple"  # こういう設定は存在しない
```

### オプション2: Monkey Patching（実験的）
```python
# run_fixed_experiment.pyで試みたアプローチ
def simple_format(prompt):
    return prompt  # 特殊トークンを追加しない

agent.l4_llm._format_prompt = simple_format
```

### オプション3: カスタムLLMプロバイダー
```python
# 独自のLLMプロバイダーを作成
class SimplePromptLLMProvider(L4LLMProvider):
    def _format_prompt(self, prompt: str) -> str:
        return prompt  # シンプルに
```

## 結論：
- プロンプトフォーマットは**完全にsrc以下で決定**されている
- 実験スクリプトからは制御できない（設定オプションがない）
- 変更するには、srcのコードを修正するか、Monkey Patchingが必要