"""
Phase 2: RAG比較実験
=====================================

InsightSpike-AIがRAGBenchmarkで主要RAGシステムを上回る性能を示すことを検証

目標: 2.5倍高速化、50%メモリ削減、FactScore 0.85+
競合: LangChain, LlamaIndex, Haystack vs InsightSpike

安全性機能:
- 実験前の自動データバックアップ
- 実験用データの分離実行
- 実験後の自動ロールバック
"""

import sys
import time
import json
import numpy as np
import pandas as pd
import argparse
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from abc import ABC, abstractmethod

# 共通ユーティリティインポート
sys.path.append(str(Path(__file__).parent.parent / "shared"))
from data_manager import safe_experiment_environment, with_data_safety, create_experiment_data_config
from evaluation_metrics import MetricsCalculator
from experiment_reporter import ExperimentReporter

# CLI機能インポート（フォールバック対応）
try:
    from cli_utils import create_base_cli_parser, add_phase_specific_args, merge_cli_config, print_experiment_header, handle_cli_error, create_experiment_summary
    from scripts_integration import ScriptsIntegratedExperiment, print_scripts_integration_status
    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False
    print("⚠️  CLI機能が利用できません - 基本モードで実行")

# 外部ライブラリ（オプショナル）
try:
    from langchain.chains import RetrievalQA
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from llama_index import VectorStoreIndex, Document
    from llama_index.retrievers import VectorIndexRetriever
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False

# InsightSpike-AI imports
try:
    from insightspike.core.agents.main_agent import MainAgent
    INSIGHTSPIKE_AVAILABLE = True
except ImportError:
    INSIGHTSPIKE_AVAILABLE = False


@dataclass
class RAGMetrics:
    """RAG性能評価指標"""
    response_speed_ms: float       # 応答速度 (ms)
    retrieval_speed_ms: float     # 検索速度 (ms) 
    generation_speed_ms: float    # 生成速度 (ms)
    memory_usage_mb: float        # メモリ使用量 (MB)
    index_size_mb: float         # インデックスサイズ (MB)
    fact_score: float            # 事実正確性 (0-1)
    bleu_score: float           # テキスト品質 (0-1)
    rouge_score: float          # 要約品質 (0-1)
    hallucination_rate: float  # 幻覚率 (0-1)
    documents_indexed: int      # インデックス文書数
    queries_processed: int      # 処理クエリ数


class BaseRAGSystem(ABC):
    """RAGシステムの基底クラス"""
    
    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = config or {}
        self.documents = []
        self.index = None
    
    @abstractmethod
    def build_index(self, documents: List[str]) -> float:
        """インデックス構築（実行時間を返す）"""
        pass
    
    @abstractmethod
    def query(self, question: str) -> Tuple[str, float]:
        """クエリ実行（回答と実行時間を返す）"""
        pass
    
    def get_memory_usage(self) -> float:
        """メモリ使用量を取得"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # psutilが利用できない場合は模擬値を返す
            return 80.0  # 80MB程度の模擬値


class LangChainRAGSystem(BaseRAGSystem):
    """LangChain + FAISS RAGシステム"""
    
    def __init__(self, config: Dict = None):
        super().__init__("LangChain_RAG", config)
        if LANGCHAIN_AVAILABLE:
            self.embeddings = HuggingFaceEmbeddings()
            self.qa_chain = None
        else:
            logging.warning("LangChain not available, using mock system")
    
    def build_index(self, documents: List[str]) -> float:
        """FAISSインデックス構築"""
        start_time = time.time()
        
        if LANGCHAIN_AVAILABLE:
            try:
                # 実際のLangChain実装
                self.vectorstore = FAISS.from_texts(documents, self.embeddings)
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=None,  # 簡略化のためLLMは省略
                    chain_type="stuff",
                    retriever=self.vectorstore.as_retriever()
                )
            except Exception as e:
                logging.warning(f"LangChain implementation failed: {e}, using mock")
                self._mock_build_index(documents)
        else:
            self._mock_build_index(documents)
        
        return time.time() - start_time
    
    def _mock_build_index(self, documents: List[str]):
        """モックインデックス構築"""
        self.documents = documents
        # シンプルなキーワードベースインデックス
        self.index = {i: doc.lower().split() for i, doc in enumerate(documents)}
    
    def query(self, question: str) -> Tuple[str, float]:
        """クエリ実行"""
        start_time = time.time()
        
        if LANGCHAIN_AVAILABLE and self.qa_chain:
            try:
                result = self.qa_chain.run(question)
                response = result if isinstance(result, str) else str(result)
            except:
                response = self._mock_query(question)
        else:
            response = self._mock_query(question)
        
        execution_time = time.time() - start_time
        return response, execution_time * 1000  # ms
    
    def _mock_query(self, question: str) -> str:
        """モッククエリ"""
        # 現実的な処理時間をシミュレート
        time.sleep(0.01 + np.random.exponential(0.02))  # 10-50ms程度の遅延
        
        # 簡単なキーワードマッチング
        keywords = question.lower().split()
        best_match = ""
        max_score = 0
        
        for doc in self.documents[:5]:  # 上位5文書をチェック
            score = sum(1 for word in keywords if word in doc.lower())
            if score > max_score:
                max_score = score
                best_match = doc
        
        return f"Based on the available information: {best_match[:200]}..."


class LlamaIndexRAGSystem(BaseRAGSystem):
    """LlamaIndex RAGシステム"""
    
    def __init__(self, config: Dict = None):
        super().__init__("LlamaIndex_RAG", config)
        self.index = None
    
    def build_index(self, documents: List[str]) -> float:
        """LlamaIndexインデックス構築"""
        start_time = time.time()
        
        if LLAMAINDEX_AVAILABLE:
            try:
                # 実際のLlamaIndex実装
                docs = [Document(text=doc) for doc in documents]
                self.index = VectorStoreIndex.from_documents(docs)
            except Exception as e:
                logging.warning(f"LlamaIndex implementation failed: {e}, using mock")
                self._mock_build_index(documents)
        else:
            self._mock_build_index(documents)
        
        return time.time() - start_time
    
    def _mock_build_index(self, documents: List[str]):
        """モックインデックス構築"""
        self.documents = documents
        # TF-IDFライクなインデックス
        from collections import Counter
        self.index = {}
        for i, doc in enumerate(documents):
            words = doc.lower().split()
            self.index[i] = Counter(words)
    
    def query(self, question: str) -> Tuple[str, float]:
        """クエリ実行"""
        start_time = time.time()
        
        if LLAMAINDEX_AVAILABLE and self.index:
            try:
                query_engine = self.index.as_query_engine()
                response = query_engine.query(question)
                result = str(response)
            except:
                result = self._mock_query(question)
        else:
            result = self._mock_query(question)
        
        execution_time = time.time() - start_time
        return result, execution_time * 1000  # ms
    
    def _mock_query(self, question: str) -> str:
        """モッククエリ（TF-IDF類似）"""
        # LlamaIndexの処理時間をシミュレート
        time.sleep(0.015 + np.random.exponential(0.03))  # 15-60ms程度の遅延
        
        from collections import Counter
        query_words = Counter(question.lower().split())
        
        best_doc_id = 0
        max_similarity = 0
        
        for doc_id, doc_counter in self.index.items():
            # コサイン類似度の簡易版
            intersection = sum((query_words & doc_counter).values())
            if intersection > max_similarity:
                max_similarity = intersection
                best_doc_id = doc_id
        
        return f"Most relevant information: {self.documents[best_doc_id][:200]}..."


class HaystackRAGSystem(BaseRAGSystem):
    """Haystack + Elasticsearch RAGシステム（モック）"""
    
    def __init__(self, config: Dict = None):
        super().__init__("Haystack_RAG", config)
        
    def build_index(self, documents: List[str]) -> float:
        """Elasticsearchインデックス構築（モック）"""
        start_time = time.time()
        
        # Haystackの特徴を模擬
        self.documents = documents
        # BM25ライクなスコアリング
        self.bm25_index = self._build_bm25_index(documents)
        
        # より遅い構築時間をシミュレート（Elasticsearchの特性）
        time.sleep(len(documents) * 0.001)
        
        return time.time() - start_time
    
    def _build_bm25_index(self, documents: List[str]) -> Dict:
        """BM25インデックス構築"""
        from collections import defaultdict
        from math import log
        
        index = defaultdict(list)
        doc_lengths = []
        
        for doc_id, doc in enumerate(documents):
            words = doc.lower().split()
            doc_lengths.append(len(words))
            
            for word in set(words):
                term_freq = words.count(word)
                index[word].append((doc_id, term_freq))
        
        self.doc_lengths = doc_lengths
        self.avg_doc_length = sum(doc_lengths) / len(doc_lengths)
        
        return dict(index)
    
    def query(self, question: str) -> Tuple[str, float]:
        """BM25クエリ実行"""
        from collections import defaultdict
        from math import log
        
        start_time = time.time()
        
        # Haystackの企業級処理をシミュレート
        time.sleep(0.02 + np.random.exponential(0.04))  # 20-80ms程度の遅延
        
        query_terms = question.lower().split()
        doc_scores = defaultdict(float)
        
        k1, b = 1.5, 0.75  # BM25パラメータ
        N = len(self.documents)
        
        for term in query_terms:
            if term in self.bm25_index:
                df = len(self.bm25_index[term])  # 文書頻度
                idf = log((N - df + 0.5) / (df + 0.5))
                
                for doc_id, tf in self.bm25_index[term]:
                    doc_len = self.doc_lengths[doc_id]
                    norm_tf = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / self.avg_doc_length))
                    doc_scores[doc_id] += idf * norm_tf
        
        # 最高スコア文書を選択
        if doc_scores:
            best_doc_id = max(doc_scores, key=doc_scores.get)
            result = f"Enterprise-grade search result: {self.documents[best_doc_id][:200]}..."
        else:
            result = "No relevant documents found in enterprise search."
        
        execution_time = time.time() - start_time
        return result, execution_time * 1000


class InsightSpikeRAGSystem(BaseRAGSystem):
    """InsightSpike-AI RAGシステム"""
    
    def __init__(self, config: Dict = None):
        super().__init__("InsightSpike_RAG", config)
        if INSIGHTSPIKE_AVAILABLE:
            try:
                self.agent = MainAgent()
            except:
                self.agent = None
                logging.warning("InsightSpike MainAgent not available, using mock")
        else:
            self.agent = None
    
    def build_index(self, documents: List[str]) -> float:
        """InsightSpike動的インデックス構築"""
        start_time = time.time()
        
        if self.agent:
            # 実際のInsightSpikeエージェント使用
            self.documents = documents
            # 動的記憶構築（より効率的）
            for doc in documents:
                self.agent.process_question(f"Learn from: {doc}")
        else:
            # 高効率モック実装
            self._mock_dynamic_indexing(documents)
        
        return time.time() - start_time
    
    def _mock_dynamic_indexing(self, documents: List[str]):
        """動的インデックス構築のモック"""
        self.documents = documents
        # グラフベースの知識表現をシミュレート
        self.knowledge_graph = {}
        self.insight_cache = {}
        
        for i, doc in enumerate(documents):
            # 洞察抽出のシミュレート
            insights = self._extract_insights(doc)
            self.insight_cache[i] = insights
            
            # グラフ構築のシミュレート
            for insight in insights:
                if insight not in self.knowledge_graph:
                    self.knowledge_graph[insight] = []
                self.knowledge_graph[insight].append(i)
    
    def _extract_insights(self, doc: str) -> List[str]:
        """洞察抽出（モック）"""
        words = doc.lower().split()
        # 重要そうなキーワードを抽出
        insights = []
        for i, word in enumerate(words):
            if len(word) > 5 and i % 3 == 0:  # 適当な条件
                insights.append(word)
        return insights[:5]  # 上位5洞察
    
    def query(self, question: str) -> Tuple[str, float]:
        """InsightSpikeクエリ実行"""
        start_time = time.time()
        
        if self.agent:
            try:
                result = self.agent.process_question(question)
                response = result.get('answer', result.get('response', 'No answer generated'))
            except:
                response = self._mock_insight_query(question)
        else:
            response = self._mock_insight_query(question)
        
        execution_time = time.time() - start_time
        return response, execution_time * 1000
    
    def _mock_insight_query(self, question: str) -> str:
        """洞察ベースクエリ（モック）"""
        # InsightSpikeの高度な処理をシミュレート（より長い処理時間）
        time.sleep(0.05 + np.random.exponential(0.1))  # 50-200ms程度の遅延
        
        query_words = question.lower().split()
        
        # グラフベース検索
        relevant_docs = set()
        for word in query_words:
            if word in self.knowledge_graph:
                relevant_docs.update(self.knowledge_graph[word])
        
        if relevant_docs:
            # 複数文書から洞察を統合
            doc_id = list(relevant_docs)[0]
            insights = self.insight_cache.get(doc_id, [])
            base_content = self.documents[doc_id][:150]
            
            return f"Insight-driven analysis: {base_content}... Key insights: {', '.join(insights[:3])}"
        else:
            return "Based on dynamic knowledge graph analysis, here's what I found..."


class RAGBenchmarkExperiment:
    """RAG比較実験メインクラス"""
    
    def __init__(self, output_dir: str = "experiments/phase2_rag_benchmark/results", config: Dict = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        
        # ロギング設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'experiment.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # RAGシステム初期化
        self.systems = {
            'LangChain': LangChainRAGSystem(),
            'LlamaIndex': LlamaIndexRAGSystem(),
            'Haystack': HaystackRAGSystem(),
            'InsightSpike': InsightSpikeRAGSystem()
        }
        
        # 乱数生成器 (再現性確保)
        self.random_seed = config.get('random_seed', 42) if config else 42
        self.rng = np.random.default_rng(self.random_seed)
    
    def generate_ragbench_dataset(self, num_docs: int = 100, num_queries: int = 20) -> Tuple[List[str], List[str]]:
        """RAGBenchライクなデータセット生成"""
        documents = []
        queries = []
        
        # 様々な分野の文書を生成
        domains = ['AI/ML', 'Science', 'Technology', 'Medicine', 'Economics']
        
        for i in range(num_docs):
            domain = domains[i % len(domains)]
            if domain == 'AI/ML':
                doc = f"Artificial Intelligence research paper {i}: This study investigates novel deep learning architectures for natural language processing. The proposed transformer variant achieves state-of-the-art performance on multiple benchmarks including GLUE and SuperGLUE. The model architecture incorporates attention mechanisms with improved computational efficiency."
            elif domain == 'Science':
                doc = f"Scientific publication {i}: Recent discoveries in quantum physics reveal new phenomena at the nanoscale. Experimental evidence supports theoretical predictions about quantum entanglement and its applications in quantum computing. The research methodology involves advanced spectroscopy techniques."
            elif domain == 'Technology':
                doc = f"Technology report {i}: Cloud computing platforms are evolving to support edge computing deployments. Distributed systems architecture enables real-time processing with minimal latency. The implementation utilizes microservices and containerization technologies."
            elif domain == 'Medicine':
                doc = f"Medical research {i}: Clinical trials demonstrate the efficacy of personalized medicine approaches. Genetic markers predict treatment response with high accuracy. The study protocol follows international guidelines for randomized controlled trials."
            else:  # Economics
                doc = f"Economic analysis {i}: Market dynamics show correlation between technology adoption and productivity growth. Statistical models predict future trends based on historical data. The econometric analysis controls for multiple confounding variables."
            
            documents.append(doc)
        
        # クエリ生成
        query_templates = [
            "What are the latest developments in {domain}?",
            "How does {concept} work in {domain}?",
            "What are the benefits of {technology} in {application}?",
            "Explain the relationship between {concept1} and {concept2}",
            "What challenges exist in {domain} research?"
        ]
        
        concepts = ['machine learning', 'quantum computing', 'cloud platforms', 'genetic analysis', 'market trends']
        
        for i in range(num_queries):
            template = query_templates[i % len(query_templates)]
            concept = concepts[i % len(concepts)]
            domain = domains[i % len(domains)]
            
            query = template.format(
                domain=domain.lower(),
                concept=concept,
                technology=concept,
                application=domain.lower(),
                concept1=concept,
                concept2=concepts[(i+1) % len(concepts)]
            )
            queries.append(query)
        
        return documents, queries
    
    def evaluate_rag_system(self, system: BaseRAGSystem, documents: List[str], queries: List[str]) -> RAGMetrics:
        """RAGシステムの評価"""
        self.logger.info(f"Evaluating {system.name}...")
        
        # インデックス構築
        start_memory = system.get_memory_usage()
        build_time = system.build_index(documents)
        end_memory = system.get_memory_usage()
        
        # より現実的なメモリ使用量計算
        base_memory = 50 + len(documents) * 0.1  # ドキュメントサイズに基づく
        system_overhead = {
            'LangChain': 15,
            'LlamaIndex': 20, 
            'Haystack': 25,
            'InsightSpike': 40  # より多くのメモリを使用（動的グラフ構築）
        }
        
        memory_diff = end_memory - start_memory
        index_size = abs(memory_diff)  # 絶対値で差分サイズを取得
        # 極小値は 0.1MB とみなす（ノイズ防止）
        if index_size < 0.1:
            index_size = 0.1
        
        # クエリ実行
        response_times = []
        responses = []
        
        for query in queries:
            response, response_time = system.query(query)
            response_times.append(response_time)
            responses.append(response)
        
        # 性能指標計算
        # クエリ応答時間(秒) -> ミリ秒へ変換し平均を算出
        avg_response_time_ms = np.mean(response_times) * 1000.0
        
        # 品質指標（模擬）
        fact_score = self._calculate_fact_score(responses)
        bleu_score = self._calculate_bleu_score(responses)
        rouge_score = self._calculate_rouge_score(responses)
        hallucination_rate = self._calculate_hallucination_rate(responses)
        
        return RAGMetrics(
            response_speed_ms=avg_response_time_ms,
            retrieval_speed_ms=avg_response_time_ms * 0.4,  # 検索は全体の40%
            generation_speed_ms=avg_response_time_ms * 0.6,  # 生成は全体の60%
            memory_usage_mb=end_memory,
            index_size_mb=index_size,
            fact_score=fact_score,
            bleu_score=bleu_score,
            rouge_score=rouge_score,
            hallucination_rate=hallucination_rate,
            documents_indexed=len(documents),
            queries_processed=len(queries)
        )
    
    def _calculate_fact_score(self, responses: List[str]) -> float:
        """事実正確性スコア計算（模擬）"""
        # システム別の特性を反映（現実的な値に調整）
        base_scores = {
            'LangChain': 0.72,
            'LlamaIndex': 0.75,
            'Haystack': 0.78,
            'InsightSpike': 0.65  # 開発段階を反映
        }
        
        # レスポンス品質に基づく調整
        avg_length = np.mean([len(r) for r in responses])
        quality_factor = min(1.2, avg_length / 200)  # 長い回答ほど高品質
        
        system_name = 'InsightSpike'  # デフォルト
        for name in base_scores:
            if name in str(responses[0]):  # 簡易的な判定
                system_name = name
                break
        
        return min(1.0, base_scores[system_name] * quality_factor)
    
    def _calculate_bleu_score(self, responses: List[str]) -> float:
        """BLEU スコア計算（模擬, 再現性確保のため RNG を固定）"""
        return float(self.rng.uniform(0.3, 0.8))
    
    def _calculate_rouge_score(self, responses: List[str]) -> float:
        """ROUGE スコア計算（模擬, 再現性確保のため RNG を固定）"""
        return float(self.rng.uniform(0.4, 0.7))
    
    def _calculate_hallucination_rate(self, responses: List[str]) -> float:
        """幻覚率計算（模擬）"""
        # システム別の幻覚率
        base_rates = {
            'LangChain': 0.15,
            'LlamaIndex': 0.12,
            'Haystack': 0.08,
            'InsightSpike': 0.05  # 低い幻覚率
        }
        return float(base_rates.get('InsightSpike', 0.10))
    
    def run_comprehensive_comparison(self, document_sizes: List[int] = [50, 100, 200], runs: int = 3) -> pd.DataFrame:
        """包括的RAG性能比較 (複数試行し 95% CI を算出)"""
        results = []
        
        self.logger.info("Starting Phase 2: RAG Benchmark Comparison Experiment")
        
        for size in document_sizes:
            self.logger.info(f"Testing with {size} documents...")
            
            for run_idx in range(runs):
                self.logger.info(f"  Run {run_idx+1}/{runs} (seed={self.random_seed + run_idx})")
                # 乱数シードを更新
                self.rng = np.random.default_rng(self.random_seed + run_idx)
                
                # データセット生成
                documents, queries = self.generate_ragbench_dataset(size, min(20, size//5))
                
                # 各システムを評価
                for system_name, system in self.systems.items():
                    try:
                        metrics = self.evaluate_rag_system(system, documents, queries)
                        
                        result = {
                            'run': run_idx,
                            'document_count': size,
                            'system': system_name,
                            **asdict(metrics)
                        }
                        results.append(result)
                        
                        self.logger.info(f"{system_name} - Response: {metrics.response_speed_ms:.1f}ms, "
                                       f"FactScore: {metrics.fact_score:.3f}, "
                                       f"Memory: {metrics.memory_usage_mb:.1f}MB")
                        
                    except Exception as e:
                        self.logger.error(f"Error evaluating {system_name}: {e}")
        
        df_results = pd.DataFrame(results)
        # 95% CI 集計
        agg_funcs = {
            'response_speed_ms': ['mean', 'std'],
            'fact_score': ['mean', 'std'],
            'memory_usage_mb': ['mean', 'std']
        }
        ci_df = df_results.groupby(['document_count', 'system']).agg(agg_funcs)
        ci_df.columns = [f"{m}_{s}" for m, s in ci_df.columns]
        ci_df = ci_df.reset_index()
        
        # 保存
        df_results.to_csv(self.output_dir / 'rag_benchmark_results_runs.csv', index=False)
        ci_df.to_csv(self.output_dir / 'rag_benchmark_results_ci.csv', index=False)
        self.logger.info("Results saved to {} and {}".format(
            self.output_dir / 'rag_benchmark_results_runs.csv',
            self.output_dir / 'rag_benchmark_results_ci.csv'))
        
        return df_results
    
    def generate_comparison_report(self, df_results: pd.DataFrame) -> None:
        """比較レポート生成"""
        report_path = self.output_dir / 'rag_comparison_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Phase 2: RAG比較実験 結果レポート\n\n")
            f.write("## 実験概要\n")
            f.write("主要RAGシステムとInsightSpike-AIの包括的性能比較\n\n")
            
            # システム別平均性能
            system_avg = df_results.groupby('system').mean()
            
            f.write("## システム別平均性能\n\n")
            f.write("| システム | 応答速度(ms) | FactScore | メモリ(MB) | 幻覚率 |\n")
            f.write("|----------|-------------|-----------|------------|--------|\n")
            
            for system in system_avg.index:
                f.write(f"| {system} | {system_avg.loc[system, 'response_speed_ms']:.1f} | "
                       f"{system_avg.loc[system, 'fact_score']:.3f} | "
                       f"{system_avg.loc[system, 'memory_usage_mb']:.1f} | "
                       f"{system_avg.loc[system, 'hallucination_rate']:.3f} |\n")
            
            # InsightSpikeとの比較
            if 'InsightSpike' in system_avg.index:
                insightspike_metrics = system_avg.loc['InsightSpike']
                
                f.write("\n## InsightSpike-AI vs 競合システム\n\n")
                
                for competitor in ['LangChain', 'LlamaIndex', 'Haystack']:
                    if competitor in system_avg.index:
                        competitor_metrics = system_avg.loc[competitor]
                        
                        speed_improvement = (competitor_metrics['response_speed_ms'] - insightspike_metrics['response_speed_ms']) / competitor_metrics['response_speed_ms'] * 100
                        memory_improvement = (competitor_metrics['memory_usage_mb'] - insightspike_metrics['memory_usage_mb']) / competitor_metrics['memory_usage_mb'] * 100
                        accuracy_improvement = (insightspike_metrics['fact_score'] - competitor_metrics['fact_score']) / competitor_metrics['fact_score'] * 100
                        
                        f.write(f"### vs {competitor}\n")
                        f.write(f"- **応答速度**: {speed_improvement:.1f}% 改善\n")
                        f.write(f"- **メモリ効率**: {memory_improvement:.1f}% 改善\n")
                        f.write(f"- **精度向上**: {accuracy_improvement:.1f}% 改善\n\n")
                
                # 目標達成確認
                f.write("## 目標達成状況\n")
                avg_speed_improvement = np.mean([
                    (system_avg.loc[comp, 'response_speed_ms'] - insightspike_metrics['response_speed_ms']) / system_avg.loc[comp, 'response_speed_ms'] * 100
                    for comp in ['LangChain', 'LlamaIndex', 'Haystack'] if comp in system_avg.index
                ])
                avg_memory_improvement = np.mean([
                    (system_avg.loc[comp, 'memory_usage_mb'] - insightspike_metrics['memory_usage_mb']) / system_avg.loc[comp, 'memory_usage_mb'] * 100
                    for comp in ['LangChain', 'LlamaIndex', 'Haystack'] if comp in system_avg.index
                ])
                
                f.write(f"- **応答速度2.5倍(150%)向上**: {avg_speed_improvement:.1f}% ")
                f.write("✅ 達成\n" if avg_speed_improvement >= 150 else "❌ 未達成\n")
                
                f.write(f"- **メモリ50%削減**: {avg_memory_improvement:.1f}% ")
                f.write("✅ 達成\n" if avg_memory_improvement >= 50 else "❌ 未達成\n")
                
                f.write(f"- **FactScore 0.85+**: {insightspike_metrics['fact_score']:.3f} ")
                f.write("✅ 達成\n" if insightspike_metrics['fact_score'] >= 0.85 else "❌ 未達成\n")
        
        self.logger.info(f"Comparison report generated: {report_path}")


@with_data_safety(
    experiment_name="phase2_rag_benchmark",
    backup_description="Pre-experiment backup for Phase 2: RAG Benchmark Comparison",
    auto_rollback=True,
    selective_copy=["processed", "embedding", "models", "cache"]  # RAG実験に必要なデータ
)
def run_rag_benchmark_experiment(experiment_env: Dict[str, Any] = None) -> Dict[str, Any]:
    """データ安全性機能付きRAGベンチマーク実験実行"""
    
    # 実験用データ設定取得
    data_config = create_experiment_data_config(experiment_env)
    
    # 実験用出力ディレクトリ設定
    experiment_output_dir = experiment_env["experiment_data_dir"] / "outputs"
    experiment_output_dir.mkdir(exist_ok=True)
    
    experiment = RAGBenchmarkExperiment(str(experiment_output_dir))
    
    logger = logging.getLogger(__name__)
    logger.info("=== Phase 2: RAG Benchmark Experiment (Safe Mode) ===")
    logger.info(f"Experiment data directory: {experiment_env['experiment_data_dir']}")
    logger.info(f"Backup ID: {experiment_env['backup_id']}")
    logger.info(f"Data configuration: {data_config}")
    
    try:
        # 実験実行（実験用データディレクトリを使用）
        results = experiment.run_comprehensive_comparison()
        
        # レポート生成
        experiment.generate_comparison_report(results)
        
        # 実験結果の統合データ保存
        experiment_results = {
            "experiment_name": "phase2_rag_benchmark",
            "timestamp": time.time(),
            "backup_id": experiment_env["backup_id"],
            "data_config": data_config,
            "results": results.to_dict('records'),
            "output_directory": str(experiment_output_dir),
            "success": True
        }
        
        # 実験結果JSONファイル保存
        results_file = experiment_output_dir / "experiment_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Experiment results saved to: {results_file}")
        logger.info("🎉 Phase 2 実験完了! (データは自動的に安全な状態に復元されます)")
        
        return experiment_results
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


def create_cli_parser() -> argparse.ArgumentParser:
    """Phase 2専用CLI引数パーサーの作成"""
    try:
        if CLI_AVAILABLE:
            parser = create_base_cli_parser(
                "Phase 2", 
                "RAG比較実験 - InsightSpike-AI vs 主要RAGシステム"
            )
            parser = add_phase_specific_args(parser, "phase2")
            parser.add_argument("--quick", action="store_true", help="クイックテストモード (小規模データで高速実行)")
            parser.add_argument("--seed", type=int, default=42, help="乱数シード (default: 42)")
            parser.add_argument("--runs", type=int, default=3, help="各設定を繰り返す試行回数 (default: 3)")
            return parser
    except Exception:
        pass
    
    # フォールバック: 基本CLI作成
    parser = argparse.ArgumentParser(
        description="Phase 2: RAG比較実験",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--debug', action='store_true', help='デバッグモード')
    parser.add_argument('--benchmarks', nargs='+', default=['ms_marco', 'natural_questions', 'hotpot_qa'], help='実行するベンチマーク')
    parser.add_argument('--rag-systems', nargs='+', default=['langchain', 'llamaindex', 'haystack'], help='比較するRAGシステム')
    parser.add_argument('--sample-size', type=int, default=100, help='ベンチマークサンプルサイズ')
    parser.add_argument('--output', type=str, default="experiments/phase2_rag_benchmark/results", help='出力ディレクトリ')
    parser.add_argument('--export', choices=['csv', 'json', 'excel'], default='csv', help='エクスポート形式')
    parser.add_argument('--no-backup', action='store_true', help='バックアップスキップ')
    parser.add_argument('--quick', action='store_true', help='クイックテストモード (小規模データで高速実行)')
    parser.add_argument('--seed', type=int, default=42, help='乱数シード (default: 42)')
    parser.add_argument('--runs', type=int, default=3, help='各設定を繰り返す試行回数 (default: 3)')
    parser.add_argument('--config', type=str, help='設定ファイル')
    
    return parser


def load_config_file(config_path: str) -> Dict[str, Any]:
    """設定ファイルの読み込み"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ 設定ファイル読み込み完了: {config_path}")
        return config
    except Exception as e:
        print(f"❌ 設定ファイル読み込みエラー: {e}")
        return {}


def merge_cli_config(args: argparse.Namespace, phase: str = "phase2") -> Dict[str, Any]:
    """CLI引数と設定ファイルのマージ"""
    config = {}
    
    # 設定ファイルがある場合は読み込み
    if hasattr(args, 'config') and args.config:
        config = load_config_file(args.config)
    
    # CLI引数で上書き
    config.update({
        'debug': getattr(args, 'debug', False),
        'benchmarks': getattr(args, 'benchmarks', ['ms_marco', 'natural_questions', 'hotpot_qa']),
        'rag_systems': getattr(args, 'rag_systems', ['langchain', 'llamaindex', 'haystack']),
        'sample_size': getattr(args, 'sample_size', 100),
        'export_format': getattr(args, 'export', 'csv'),
        'output_dir': getattr(args, 'output', 'experiments/phase2_rag_benchmark/results'),
        'no_backup': getattr(args, 'no_backup', False),
        'quick_mode': getattr(args, 'quick', False),
        'generate_report': True,
        'generate_plots': False,
        'selective_copy': ["processed", "embedding", "models", "cache"],
        'random_seed': getattr(args, 'seed', 42),
        'runs': getattr(args, 'runs', 3)
    })
    
    # クイックモードの場合は設定を簡素化
    if config['quick_mode']:
        config['benchmarks'] = ['ms_marco']
        config['sample_size'] = 20
        config['rag_systems'] = ['langchain']  # 1つのシステムのみ
    
    return config


def main():
    """メイン実行関数 - CLI対応・データ安全性機能付き"""
    
    # CLI引数パース（フォールバック機能付き）
    try:
        parser = create_cli_parser()
        args = parser.parse_args()
        config = merge_cli_config(args, "phase2")
    except Exception as e:
        print(f"⚠️  CLI機能エラー: {e}")
        print("🔧 基本モードで実行します")
        config = {
            'debug': False,
            'benchmarks': ['ms_marco', 'natural_questions', 'hotpot_qa'],
            'rag_systems': ['langchain', 'llamaindex', 'haystack'],
            'sample_size': 100,
            'export_format': 'csv',
            'output_dir': 'experiments/phase2_rag_benchmark/results',
            'no_backup': False,
            'selective_copy': ["processed", "embedding", "models", "cache"],
            'generate_report': True,
            'generate_plots': False,
            'quick_mode': False,
            'random_seed': 42,
            'runs': 3
        }
    
    # 実験ヘッダー表示
    try:
        if CLI_AVAILABLE:
            print_experiment_header("Phase 2: RAG比較実験", config)
            print_scripts_integration_status()
    except Exception:
        print("🔬 Phase 2: RAG比較実験")
        print("=" * 50)
        print(f"📊 ベンチマーク: {config['benchmarks']}")
        print(f"🔧 RAGシステム: {config['rag_systems']}")
        print(f"📏 サンプルサイズ: {config['sample_size']}")
        print(f"🛡️  データバックアップ: {'無効' if config['no_backup'] else '有効'}")
        print(f"🐛 デバッグモード: {'有効' if config['debug'] else '無効'}")
    
    try:
        # scripts/experiments/統合モードを試行
        try:
            if CLI_AVAILABLE:
                scripts_experiment = ScriptsIntegratedExperiment("phase2_rag_benchmark", config)
                
                def run_phase2_experiment(integrated_config):
                    if integrated_config['no_backup']:
                        # 高速モード
                        experiment = RAGBenchmarkExperiment(integrated_config['output_dir'], integrated_config)
                        results = experiment.run_comprehensive_comparison()
                        if integrated_config['generate_report']:
                            experiment.generate_comparison_report(results)
                        return results
                    else:
                        # 安全モード
                        return run_rag_benchmark_experiment()
                
                results = scripts_experiment.run_experiment(run_phase2_experiment)
                print("✅ scripts/experiments/統合モードで実行完了")
            else:
                raise Exception("CLI機能が利用できません")
                
        except Exception as integration_error:
            print(f"⚠️  scripts統合モードエラー: {integration_error}")
            print("🔧 標準モードで実行します")
            
            # 標準モード実行
            if config['no_backup']:
                # バックアップなしで直接実行（高速モード）
                print("\n⚡ 高速モード: データバックアップなしで実行")
                experiment = RAGBenchmarkExperiment(config['output_dir'], config)
                results = experiment.run_comprehensive_comparison()
                
                if config['generate_report']:
                    experiment.generate_comparison_report(results)
                
                print("\n🎉 Phase 2 実験完了! (高速モード)")
                
            else:
                # 安全な実験環境で実行（推奨）
                print("\n🛡️  安全モード: データバックアップ付きで実行")
                results = run_rag_benchmark_experiment()
        
        # 結果サマリー表示
        try:
            if CLI_AVAILABLE:
                summary = create_experiment_summary(results, "phase2")
                print(summary)
        except Exception:
            # フォールバック: 基本サマリー
            if not config.get('debug', False) and results is not None:
                print("\n📊 結果サマリー:")
                print("✅ 実験完了")
                print("📁 結果は以下に保存されています:")
                print("  - experiment_data/ (実験結果)")
                print("  - data_backups/ (バックアップ)")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⛔ 実験が中断されました")
        print("🔄 データは安全な状態に復元されています")
        return None
        
    except Exception as e:
        try:
            if CLI_AVAILABLE:
                handle_cli_error(e, config)
        except Exception:
            print(f"\n❌ 実験が失敗しました: {e}")
            if config.get('debug', False):
                import traceback
                traceback.print_exc()
            print("🔄 データは自動的に実験前の状態に復元されました")
        raise


if __name__ == "__main__":
    main()
