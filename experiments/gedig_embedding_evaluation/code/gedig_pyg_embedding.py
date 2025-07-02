#!/usr/bin/env python3
"""
PyTorch Geometric版 geDIG Embedding
==================================

既存のInsightSpike-AI PyG実装を活用した高速グラフベースembedding
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
import time
import warnings
from typing import Dict, List, Tuple, Any

# プロジェクトのsrcディレクトリをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

warnings.filterwarnings('ignore')

# InsightSpike-AI components
try:
    from insightspike.algorithms.graph_edit_distance import GraphEditDistance, OptimizationLevel
    from insightspike.algorithms.information_gain import InformationGain, EntropyMethod
    from insightspike.core.layers.layer3_graph_reasoner import L3GraphReasoner
    from insightspike.utils.graph_metrics import GraphMetrics
    print("✅ InsightSpike-AI PyG components imported successfully")
    INSIGHTSPIKE_AVAILABLE = True
except ImportError as e:
    print(f"❌ InsightSpike-AI import error: {e}")
    INSIGHTSPIKE_AVAILABLE = False

class PyGGeDIGEmbedding(nn.Module):
    """
    PyTorch Geometric版 geDIG Embedding
    GPU対応・GNNベースの高速実装
    """
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # GNNエンコーダー（InsightSpike-AIと同じアーキテクチャ）
        self.gnn_encoder = nn.Sequential(
            GCNConv(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            GCNConv(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            GCNConv(hidden_dim, output_dim),
        )
        
        # ΔGED × ΔIG 統合層
        self.gedig_fusion = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        
        # GPU利用可能チェック
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        print(f"✅ PyG geDIG Embedding initialized on {self.device}")
        
        # InsightSpike-AI components
        if INSIGHTSPIKE_AVAILABLE:
            self.ged_calculator = GraphEditDistance(optimization_level=OptimizationLevel.FAST)
            self.ig_calculator = InformationGain(method=EntropyMethod.SHANNON)
    
    def text_to_pyg_graph(self, text: str) -> Data:
        """テキストをPyTorch Geometricグラフに変換"""
        
        words = text.lower().split()[:30]  # 最大30ノード
        
        # ノード特徴量（単語埋め込み）
        node_features = []
        for word in words:
            # 簡易埋め込み（実際はBERTやWord2Vec使用推奨）
            features = [
                len(word) / 10.0,  # 正規化された長さ
                sum(1 for c in word if c in 'aeiou') / max(len(word), 1),  # 母音比率
                sum(1 for c in word if c.isdigit()) / max(len(word), 1),  # 数字比率
                ord(word[0]) / 255.0 if word else 0,  # 最初の文字
            ]
            # input_dimまでパディング
            features.extend([0.0] * (self.input_dim - len(features)))
            node_features.append(features[:self.input_dim])
        
        x = torch.tensor(node_features, dtype=torch.float32)
        
        # エッジ構築（隣接関係 + 類似性）
        edge_list = []
        
        # 隣接エッジ
        for i in range(len(words) - 1):
            edge_list.append([i, i + 1])
            edge_list.append([i + 1, i])  # 双方向
        
        # 類似性エッジ（文字の重複が多い場合）
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                if i != j and len(set(words[i]).intersection(set(words[j]))) >= 2:
                    edge_list.append([i, j])
                    edge_list.append([j, i])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            # エッジがない場合の処理
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # PyG Dataオブジェクト作成
        data = Data(x=x, edge_index=edge_index)
        data.num_nodes = len(words)
        
        return data.to(self.device)
    
    def forward(self, graph1: Data, graph2: Data) -> torch.Tensor:
        """2つのグラフからgeDIG embeddingを生成"""
        
        # GNNエンコーディング（各層を個別に呼び出し）
        x1 = graph1.x
        x2 = graph2.x
        
        for layer in self.gnn_encoder:
            if isinstance(layer, GCNConv):
                x1 = layer(x1, graph1.edge_index)
                x2 = layer(x2, graph2.edge_index)
            else:
                x1 = layer(x1)
                x2 = layer(x2)
        
        # グラフレベル表現（平均プーリング）
        batch1 = torch.zeros(graph1.num_nodes, dtype=torch.long).to(self.device)
        batch2 = torch.zeros(graph2.num_nodes, dtype=torch.long).to(self.device)
        
        graph1_repr = global_mean_pool(x1, batch1)
        graph2_repr = global_mean_pool(x2, batch2)
        
        # ΔGED × ΔIG 融合
        combined = torch.cat([graph1_repr, graph2_repr], dim=1)
        gedig_embedding = self.gedig_fusion(combined)
        
        return gedig_embedding
    
    def calculate_fast_ged_ig(self, graph1: Data, graph2: Data) -> Tuple[float, float]:
        """高速ΔGED × ΔIG計算（GPU対応）"""
        
        # グラフ特徴量を使った近似計算
        with torch.no_grad():
            # ノード数の差
            node_diff = abs(graph1.num_nodes - graph2.num_nodes)
            
            # エッジ数の差
            edge_diff = abs(graph1.edge_index.size(1) - graph2.edge_index.size(1))
            
            # ノード特徴量の差（コサイン距離）
            x1_mean = graph1.x.mean(dim=0)
            x2_mean = graph2.x.mean(dim=0)
            feature_sim = torch.cosine_similarity(x1_mean, x2_mean, dim=0)
            
            # 近似ΔGED
            delta_ged = node_diff + edge_diff * 0.5 + (1 - feature_sim) * 10
            
            # 近似ΔIG（特徴量の分散差）
            var1 = graph1.x.var()
            var2 = graph2.x.var()
            delta_ig = torch.abs(var1 - var2) / (var1 + var2 + 1e-6)
            
            return delta_ged.item(), delta_ig.item()
    
    def embed_texts(self, texts: List[str], reference_text: str = None) -> np.ndarray:
        """テキストリストをgeDIG embeddingに変換"""
        
        if reference_text is None:
            reference_text = texts[0]
        
        reference_graph = self.text_to_pyg_graph(reference_text)
        embeddings = []
        
        print(f"🧠 Generating PyG geDIG embeddings for {len(texts)} texts...")
        
        with torch.no_grad():
            for i, text in enumerate(texts):
                if i % 50 == 0:
                    print(f"   Processing {i}/{len(texts)}...")
                
                text_graph = self.text_to_pyg_graph(text)
                
                # geDIG embedding生成
                embedding = self.forward(text_graph, reference_graph)
                
                # ΔGED × ΔIG計算
                delta_ged, delta_ig = self.calculate_fast_ged_ig(text_graph, reference_graph)
                
                # 最終embedding（ΔGED×ΔIG重み付け）
                gedig_weight = delta_ged * delta_ig
                weighted_embedding = embedding * gedig_weight
                
                embeddings.append(weighted_embedding.cpu().numpy())
        
        return np.vstack(embeddings)

def benchmark_pyg_vs_original():
    """PyG版と元のgeDIG embedding性能比較"""
    
    print("🚀 PyTorch Geometric geDIG Embedding Benchmark")
    print("=" * 60)
    
    # テストデータ
    test_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming artificial intelligence",
        "Quantum computing will revolutionize cryptography",
        "Natural language processing enables human-computer interaction",
        "Deep neural networks learn hierarchical representations"
    ] * 20  # 100テキスト
    
    reference = "Artificial intelligence and machine learning research"
    
    # 1. PyG版ベンチマーク
    print("\n🧠 PyG geDIG Embedding:")
    pyg_embedder = PyGGeDIGEmbedding()
    
    start_time = time.time()
    pyg_embeddings = pyg_embedder.embed_texts(test_texts[:10], reference)
    pyg_time = time.time() - start_time
    
    print(f"   ✅ Time: {pyg_time:.3f}s")
    print(f"   📊 Shape: {pyg_embeddings.shape}")
    print(f"   🚀 Speed: {len(test_texts[:10])/pyg_time:.1f} texts/sec")
    print(f"   💾 Device: {pyg_embedder.device}")
    
    # 2. 元の実装との比較（存在する場合）
    try:
        from gedig_embedding_experiment import GeDIGEmbedding
        
        print("\n📊 Original geDIG Embedding:")
        original_embedder = GeDIGEmbedding(embedding_dim=128)
        
        start_time = time.time()
        original_embeddings = original_embedder.embed_corpus(test_texts[:10], reference)
        original_time = time.time() - start_time
        
        print(f"   ✅ Time: {original_time:.3f}s")
        print(f"   📊 Shape: {original_embeddings.shape}")
        print(f"   🚀 Speed: {len(test_texts[:10])/original_time:.1f} texts/sec")
        
        # 速度比較
        speedup = original_time / pyg_time
        print(f"\n⚡ PyG Speedup: {speedup:.2f}x faster!")
        
    except ImportError:
        print("\n⚠️ Original implementation not found for comparison")
    
    # 3. GPU vs CPU比較（GPU利用可能な場合）
    if torch.cuda.is_available():
        print("\n🔥 GPU Performance Test:")
        
        # より大きなデータセットでテスト
        large_texts = test_texts * 10  # 500テキスト
        
        start_time = time.time()
        gpu_embeddings = pyg_embedder.embed_texts(large_texts[:50], reference)
        gpu_time = time.time() - start_time
        
        print(f"   ✅ GPU Time (50 texts): {gpu_time:.3f}s")
        print(f"   🚀 GPU Speed: {50/gpu_time:.1f} texts/sec")
        
        # 理論上の1000問処理時間
        estimated_1000 = (1000 / 50) * gpu_time
        print(f"   📈 Estimated time for 1000 texts: {estimated_1000:.1f}s")
    
    print("\n✅ PyG geDIG Embedding benchmark completed!")

if __name__ == "__main__":
    benchmark_pyg_vs_original()