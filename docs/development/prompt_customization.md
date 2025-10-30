---
status: active
category: llm
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# プロンプトカスタマイゼーション機能

## 概要

InsightSpikeのプロンプト生成を、モデルの能力に応じてYAML設定でカスタマイズできるようになりました。特に、GPT-2のような軽量モデル向けの簡潔なプロンプトをサポートします。

## 設定オプション

### LLMConfig の新しいフィールド

```yaml
llm:
  # 基本設定
  provider: local
  model: distilgpt2
  
  # プロンプトカスタマイゼーション
  use_simple_prompt: true      # 簡潔なプロンプトを使用
  prompt_style: minimal        # minimal/standard/detailed
  max_context_docs: 2          # コンテキストに含む文書数
  include_metadata: false      # メタデータ（relevance等）を含むか
  
  # カスタムテンプレート
  prompt_template: |
    {context}
    Q: {question}
    A:
```

## プロンプトスタイル

### 1. Minimal（軽量モデル向け）
```
Context: First 150 chars of doc1... First 150 chars of doc2...
Q: What is AI?
A:
```
- 文書数: 最大2
- 文字数: 各150文字まで
- メタデータ: なし

### 2. Standard（バランス型）
```
Context:
Machine learning is a subset of AI...
Deep learning uses neural networks...

Question: What is AI?

Answer:
```
- 文書数: 3-5
- 文字数: 制限なし
- メタデータ: オプション

### 3. Detailed（大規模モデル向け）
```
Retrieved Information:
1. Machine learning is... (relevance: 0.92)
2. Deep learning uses... (relevance: 0.87)

Insight Detection: Significant pattern identified

Question: What is AI?

Answer:
```
- 文書数: 5以上
- 文字数: 制限なし
- メタデータ: 完全に含む

## 使用例

### GPT-2用の最小設定
```yaml
llm:
  provider: local
  model: distilgpt2
  use_simple_prompt: true
  prompt_style: minimal
  max_context_docs: 2
```

### カスタムテンプレート
```yaml
llm:
  prompt_template: |
    Based on the following: {context}
    Please answer: {question}
    Response:
```

### 条件付き設定
```yaml
# 実験用設定
experiment:
  llm:
    use_simple_prompt: true
    
# 本番用設定  
production:
  llm:
    prompt_style: detailed
    include_metadata: true
```

## プロンプトサイズの比較

| スタイル | 文字数 | GPT-2適性 | 情報量 |
|---------|--------|-----------|---------|
| Minimal | ~200 | ◎ | 基本のみ |
| Standard | ~500 | ○ | 中程度 |
| Detailed | ~800+ | × | 完全 |

## 実装の詳細

### _build_simple_prompt メソッド
```python
def _build_simple_prompt(self, context, question):
    # 最初の2文書のみ
    # 各150文字に制限
    # シンプルなフォーマット
    return f"Context: {context}\nQ: {question}\nA:"
```

### プロンプトスタイルの自動選択
```python
if self.config.use_simple_prompt or self.config.prompt_style == "minimal":
    return self._build_simple_prompt(context, question)
```

## メリット

1. **パフォーマンス向上**: 軽量モデルに適切な長さ
2. **柔軟性**: モデルに応じた最適化
3. **実験の容易さ**: YAML設定で簡単に切り替え
4. **後方互換性**: 既存の設定も動作

## まとめ

この機能により、InsightSpikeは様々な規模のLLMに対応できるようになりました。特にGPT-2のような軽量モデルでも、適切なプロンプト設定により良好な結果が得られます。