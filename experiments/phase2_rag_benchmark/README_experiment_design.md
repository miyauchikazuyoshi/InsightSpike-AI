# Phase 2: RAG比較実験

## 📋 実験設計

### 🎯 目標
- **応答速度**: 2.5倍（150%）向上
- **メモリ削減**: 50%削減
- **精度向上**: FactScore 0.85+達成

### 🔬 実験設計

#### 比較対象システム
1. **LangChain + FAISS**
   - HuggingFaceEmbeddings使用
   - RetrievalQA chain
   - 実装フォールバック対応

2. **LlamaIndex**
   - VectorStoreIndex
   - Document based approach
   - 自動インデックス構築

3. **Haystack**
   - Pipeline based system
   - 柔軟なコンポーネント構成

4. **InsightSpike-AI**
   - MainAgentベース
   - 動的洞察生成
   - 統合記憶システム

#### テストデータ
- **文書数**: 50, 100, 200文書で段階的評価
- **クエリ数**: 各文書数あたり10-20クエリ
- **データソース**: 20Newsgroups、MS MARCO、Natural Questions等

#### 評価指標
- **応答速度（ms）**: エンドツーエンドの応答時間
- **検索速度（ms）**: 関連文書検索時間
- **生成速度（ms）**: 回答生成時間
- **メモリ使用量（MB）**: ピーク時メモリ消費
- **インデックスサイズ（MB）**: 検索インデックスサイズ
- **FactScore（0-1）**: 事実正確性スコア
- **BLEU/ROUGE**: テキスト品質指標
- **幻覚率（0-1）**: 不正確な情報生成率

### 🔧 実装詳細

#### LangChain実装
```python
class LangChainRAGSystem:
    def build_index(self, documents: List[str]) -> float:
        self.vectorstore = FAISS.from_texts(documents, self.embeddings)
        self.qa_chain = RetrievalQA.from_chain_type(...)
```

#### InsightSpike実装
```python
class InsightSpikeRAGSystem:
    def query(self, question: str) -> Tuple[str, float]:
        if self.agent:
            result = self.agent.process_question(question)
```

### 🛡️ 安全性機能
- 実験前の自動データバックアップ
- 外部依存関係のフォールバック処理
- モック実装による再現性確保

### 📊 ベンチマークデータセット
- **20 Newsgroups**: 分類・検索評価
- **MS MARCO**: QA性能評価
- **Natural Questions**: 自然言語理解
- **HotpotQA**: 多段推論評価
- **BEIR**: 検索ベンチマーク
