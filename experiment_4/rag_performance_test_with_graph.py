#!/usr/bin/env python3
"""
RAG Performance Test with Proper Graph Embeddings
現在の動的グラフ成長データ構造でのRAG性能テスト
"""

import os
import sys
import json
import torch
import time
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.insightspike.core.agents.main_agent import MainAgent

class RAGPerformanceEvaluator:
    """RAG性能を評価するクラス"""
    
    def __init__(self):
        self.agent = MainAgent()
        self.agent.initialize()  # エージェントを初期化
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def create_test_queries(self):
        """テストクエリとその正解ドキュメントを定義"""
        # 現在のデータに基づいたテストクエリ
        test_queries = [
            {
                "query": "artificial intelligence in healthcare",
                "relevant_keywords": ["healthcare", "medical", "ai", "deep learning", "diagnosis"],
                "query_type": "factual"
            },
            {
                "query": "machine learning applications",
                "relevant_keywords": ["machine learning", "ml", "applications", "algorithm"],
                "query_type": "factual"
            },
            {
                "query": "quantum computing future",
                "relevant_keywords": ["quantum", "computing", "future", "technology"],
                "query_type": "exploratory"
            },
            {
                "query": "cybersecurity threats",
                "relevant_keywords": ["cybersecurity", "security", "threats", "protection"],
                "query_type": "factual"
            },
            {
                "query": "data science techniques",
                "relevant_keywords": ["data science", "analytics", "techniques", "mining"],
                "query_type": "factual"
            },
            {
                "query": "robotics in manufacturing",
                "relevant_keywords": ["robotics", "manufacturing", "automation", "production"],
                "query_type": "factual"
            },
            {
                "query": "blockchain finance applications",
                "relevant_keywords": ["blockchain", "finance", "fintech", "cryptocurrency"],
                "query_type": "factual"
            },
            {
                "query": "edge computing benefits",
                "relevant_keywords": ["edge computing", "distributed", "latency", "benefits"],
                "query_type": "factual"
            },
            {
                "query": "natural language processing advances",
                "relevant_keywords": ["nlp", "natural language", "processing", "advances"],
                "query_type": "factual"
            },
            {
                "query": "renewable energy AI optimization",
                "relevant_keywords": ["renewable", "energy", "ai", "optimization", "sustainability"],
                "query_type": "exploratory"
            }
        ]
        
        return test_queries
    
    def evaluate_retrieval(self, query, results, relevant_keywords):
        """検索結果の評価"""
        # 検索結果が関連キーワードを含むかチェック
        relevant_count = 0
        for result in results:
            text = result.get('text', '').lower()
            if any(keyword in text for keyword in relevant_keywords):
                relevant_count += 1
        
        precision = relevant_count / len(results) if results else 0
        
        return {
            'precision': precision,
            'relevant_count': relevant_count,
            'total_retrieved': len(results)
        }
    
    def run_rag_tests(self):
        """RAG性能テストを実行"""
        print("=== RAG Performance Test with Graph Embeddings ===\n")
        
        # データの読み込み
        with open('data/episodes.json', 'r') as f:
            episodes = json.load(f)
        
        graph = torch.load('data/graph_pyg.pt')
        
        print(f"Dataset size:")
        print(f"  Episodes: {len(episodes)}")
        print(f"  Graph nodes: {graph.num_nodes}")
        print(f"  Graph edges: {graph.edge_index.size(1) if hasattr(graph, 'edge_index') else 0}")
        print()
        
        # テストクエリの準備
        test_queries = self.create_test_queries()
        
        # 結果を格納
        results = {
            'factual_queries': [],
            'exploratory_queries': [],
            'overall_metrics': {}
        }
        
        total_precision = 0
        total_latency = 0
        
        for query_data in test_queries:
            query = query_data['query']
            keywords = query_data['relevant_keywords']
            query_type = query_data['query_type']
            
            print(f"\nQuery: '{query}'")
            print(f"Type: {query_type}")
            
            # 検索実行と時間計測
            start_time = time.time()
            
            # InsightSpike検索 - _search_memoryメソッドを使用
            try:
                # MainAgentの内部メモリ検索を直接使用
                memory_results = self.agent._search_memory(query)
                
                # 検索結果を抽出
                search_results = []
                if 'similar_episodes' in memory_results:
                    for episode in memory_results['similar_episodes'][:5]:
                        search_results.append({
                            'text': episode.get('text', ''),
                            'score': episode.get('similarity', 0)
                        })
                
                # もし結果がなければ、process_questionを使用
                if not search_results:
                    response = self.agent.process_question(query)
                    if response:
                        search_results = [{'text': response}]
            except Exception as e:
                print(f"  Error during search: {e}")
                search_results = []
            
            latency = (time.time() - start_time) * 1000  # ms
            
            # 評価
            eval_metrics = self.evaluate_retrieval(query, search_results, keywords)
            
            print(f"  Precision: {eval_metrics['precision']:.2f}")
            print(f"  Relevant: {eval_metrics['relevant_count']}/{eval_metrics['total_retrieved']}")
            print(f"  Latency: {latency:.2f}ms")
            
            # 結果の保存
            result = {
                'query': query,
                'precision': eval_metrics['precision'],
                'relevant_count': eval_metrics['relevant_count'],
                'total_retrieved': eval_metrics['total_retrieved'],
                'latency': latency
            }
            
            if query_type == 'factual':
                results['factual_queries'].append(result)
            else:
                results['exploratory_queries'].append(result)
            
            total_precision += eval_metrics['precision']
            total_latency += latency
        
        # 全体的なメトリクス
        avg_precision = total_precision / len(test_queries)
        avg_latency = total_latency / len(test_queries)
        
        # タイプ別の平均
        factual_precision = np.mean([r['precision'] for r in results['factual_queries']])
        exploratory_precision = np.mean([r['precision'] for r in results['exploratory_queries']])
        
        results['overall_metrics'] = {
            'average_precision': avg_precision,
            'average_latency_ms': avg_latency,
            'factual_precision': factual_precision,
            'exploratory_precision': exploratory_precision,
            'total_queries': len(test_queries)
        }
        
        print("\n=== Summary ===")
        print(f"Overall Average Precision: {avg_precision:.3f}")
        print(f"Factual Query Precision: {factual_precision:.3f}")
        print(f"Exploratory Query Precision: {exploratory_precision:.3f}")
        print(f"Average Latency: {avg_latency:.2f}ms")
        
        # グラフ埋め込みの効果を分析
        self.analyze_graph_impact(results, graph)
        
        # 結果を保存
        with open('rag_performance_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results
    
    def analyze_graph_impact(self, search_results, graph):
        """グラフ埋め込みの影響を分析"""
        print("\n=== Graph Embedding Impact Analysis ===")
        
        # グラフの密度
        if hasattr(graph, 'edge_index'):
            num_edges = graph.edge_index.size(1)
            num_nodes = graph.num_nodes
            density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
            
            print(f"Graph Density: {density:.4f}")
            print(f"Average Degree: {(2 * num_edges) / num_nodes:.2f}")
            
            # この密度が検索に与える影響
            if density > 0.1:
                print("⚠️  High graph density may cause over-connection")
                print("   This could lead to retrieving less relevant results")
            else:
                print("✓  Graph density is reasonable")
    
    def compare_with_baseline(self):
        """ベースラインRAGとの比較"""
        print("\n=== Comparison with Baseline RAG ===")
        
        # シンプルなベクトル検索との比較
        from sklearn.metrics.pairwise import cosine_similarity
        
        # エピソードの埋め込みを取得
        with open('data/episodes.json', 'r') as f:
            episodes = json.load(f)
        
        texts = [ep['text'] for ep in episodes]
        embeddings = self.model.encode(texts)
        
        test_queries = self.create_test_queries()
        baseline_precision = 0
        
        for query_data in test_queries[:3]:  # 最初の3つで比較
            query = query_data['query']
            keywords = query_data['relevant_keywords']
            
            # ベースライン検索
            query_emb = self.model.encode([query])
            similarities = cosine_similarity(query_emb, embeddings)[0]
            top_5_idx = np.argsort(similarities)[-5:][::-1]
            
            # 評価
            relevant = 0
            for idx in top_5_idx:
                if any(kw in texts[idx].lower() for kw in keywords):
                    relevant += 1
            
            precision = relevant / 5
            baseline_precision += precision
            
            print(f"\nQuery: '{query}'")
            print(f"  Baseline Precision: {precision:.2f}")
        
        avg_baseline = baseline_precision / 3
        print(f"\nAverage Baseline Precision: {avg_baseline:.3f}")
        print("(Simple vector search without graph structure)")

def main():
    """メイン実行関数"""
    evaluator = RAGPerformanceEvaluator()
    
    # RAGテストの実行
    results = evaluator.run_rag_tests()
    
    # ベースラインとの比較
    evaluator.compare_with_baseline()
    
    print("\n✅ RAG performance test completed!")
    print("Results saved to: rag_performance_results.json")

if __name__ == "__main__":
    main()