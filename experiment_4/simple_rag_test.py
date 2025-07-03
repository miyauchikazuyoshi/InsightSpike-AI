#!/usr/bin/env python3
"""
Simple RAG Performance Test with Current Graph Data
シンプルなRAG性能テスト
"""

import json
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

def load_data():
    """データの読み込み"""
    with open('data/episodes.json', 'r') as f:
        episodes = json.load(f)
    
    graph = torch.load('data/graph_pyg.pt')
    
    # FAISSインデックスも確認
    index_exists = False
    try:
        index = faiss.read_index('data/index.faiss')
        index_exists = True
    except:
        index = None
    
    return episodes, graph, index, index_exists

def test_rag_performance():
    """RAG性能をテスト"""
    print("=== Simple RAG Performance Test ===\n")
    
    # データ読み込み
    episodes, graph, index, index_exists = load_data()
    
    print(f"Data Statistics:")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Graph nodes: {graph.num_nodes}")
    print(f"  Graph edges: {graph.edge_index.size(1) if hasattr(graph, 'edge_index') else 0}")
    print(f"  FAISS index: {'Found' if index_exists else 'Not found'}")
    print()
    
    # 埋め込みモデル
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # テストクエリ
    test_queries = [
        {
            "query": "artificial intelligence in healthcare",
            "keywords": ["healthcare", "medical", "ai", "artificial intelligence"],
            "type": "factual"
        },
        {
            "query": "machine learning applications",
            "keywords": ["machine learning", "applications", "ml"],
            "type": "factual"
        },
        {
            "query": "quantum computing",
            "keywords": ["quantum", "computing"],
            "type": "factual"
        },
        {
            "query": "future of AI and robotics",
            "keywords": ["ai", "robotics", "future", "artificial intelligence"],
            "type": "exploratory"
        },
        {
            "query": "cross-domain applications of deep learning",
            "keywords": ["deep learning", "applications", "cross", "domain"],
            "type": "exploratory"
        }
    ]
    
    # エピソードのテキストと埋め込みを準備
    texts = [ep['text'] for ep in episodes]
    
    # 埋め込みを取得（グラフから or 新規計算）
    if hasattr(graph, 'x') and graph.x is not None and graph.x.shape[0] >= len(episodes):
        print("Using embeddings from graph...")
        embeddings = graph.x[:len(episodes)].numpy()
    else:
        print("Computing embeddings...")
        embeddings = model.encode(texts)
    
    print(f"Embeddings shape: {embeddings.shape}\n")
    
    # 結果を格納
    results = {
        'queries': [],
        'summary': {}
    }
    
    total_precision = 0
    factual_precision = 0
    exploratory_precision = 0
    factual_count = 0
    exploratory_count = 0
    
    for query_data in test_queries:
        query = query_data['query']
        keywords = query_data['keywords']
        query_type = query_data['type']
        
        print(f"\nQuery: '{query}'")
        print(f"Type: {query_type}")
        
        # クエリの埋め込み
        query_embedding = model.encode([query])
        
        # 検索実行
        start_time = time.time()
        
        # コサイン類似度で検索
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_5_indices = np.argsort(similarities)[-5:][::-1]
        
        latency = (time.time() - start_time) * 1000
        
        # 結果の評価
        relevant_count = 0
        print("  Top 5 results:")
        for i, idx in enumerate(top_5_indices):
            text = texts[idx].lower()
            is_relevant = any(keyword in text for keyword in keywords)
            if is_relevant:
                relevant_count += 1
            print(f"    {i+1}. [{'✓' if is_relevant else '✗'}] {texts[idx][:60]}... (score: {similarities[idx]:.3f})")
        
        precision = relevant_count / 5
        print(f"  Precision@5: {precision:.2f} ({relevant_count}/5 relevant)")
        print(f"  Latency: {latency:.2f}ms")
        
        # 結果を保存
        results['queries'].append({
            'query': query,
            'type': query_type,
            'precision': precision,
            'relevant_count': relevant_count,
            'latency': latency,
            'top_scores': similarities[top_5_indices].tolist()
        })
        
        total_precision += precision
        if query_type == 'factual':
            factual_precision += precision
            factual_count += 1
        else:
            exploratory_precision += precision
            exploratory_count += 1
    
    # サマリー
    avg_precision = total_precision / len(test_queries)
    avg_factual = factual_precision / factual_count if factual_count > 0 else 0
    avg_exploratory = exploratory_precision / exploratory_count if exploratory_count > 0 else 0
    
    print("\n=== Summary ===")
    print(f"Overall Precision@5: {avg_precision:.3f}")
    print(f"Factual Query Precision: {avg_factual:.3f}")
    print(f"Exploratory Query Precision: {avg_exploratory:.3f}")
    
    results['summary'] = {
        'overall_precision': avg_precision,
        'factual_precision': avg_factual,
        'exploratory_precision': avg_exploratory,
        'total_queries': len(test_queries),
        'data_size': {
            'episodes': len(episodes),
            'graph_nodes': graph.num_nodes,
            'graph_edges': graph.edge_index.size(1) if hasattr(graph, 'edge_index') else 0
        }
    }
    
    # グラフの影響を分析
    print("\n=== Graph Impact Analysis ===")
    if hasattr(graph, 'edge_index') and graph.edge_index.size(1) > 0:
        num_nodes = graph.num_nodes
        num_edges = graph.edge_index.size(1)
        density = num_edges / (num_nodes * (num_nodes - 1))
        avg_degree = (2 * num_edges) / num_nodes
        
        print(f"Graph Density: {density:.4f}")
        print(f"Average Degree: {avg_degree:.1f}")
        
        if density > 0.1:
            print("⚠️  Very high graph density - may cause over-connection in search")
        elif density > 0.05:
            print("⚡ High graph density - rich connectivity but may affect precision")
        else:
            print("✓ Moderate graph density")
    
    # 結果を保存
    with open('simple_rag_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Results saved to simple_rag_test_results.json")
    
    return results

if __name__ == "__main__":
    test_rag_performance()