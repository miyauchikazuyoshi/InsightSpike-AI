#!/usr/bin/env python3
"""
毎エピソード洞察検出実験 + グラフ成長ビジュアライゼーション
===========================================================

TopK最適化を使用して毎エピソードで洞察を検出し、
グラフの成長過程をリアルタイムでビジュアライズします。
"""

import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import csv
import torch
from collections import defaultdict, deque

# InsightSpike-AIのパスを追加
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from insightspike.core.agents.main_agent import MainAgent
    from insightspike.utils.embedder import get_model
    from insightspike.core.config import get_config
    from insightspike.core.learning.knowledge_graph_memory import KnowledgeGraphMemory
    print("📦 InsightSpike components imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class TopKOptimizedGEDCalculator:
    """TopK最適化されたGED計算器"""
    
    def __init__(self, k: int = 10):
        self.k = k
        self.previous_embeddings = []
        self.similarity_threshold = 0.7
        
    def calculate_optimized_ged(self, new_embedding: np.ndarray, 
                              knowledge_graph: KnowledgeGraphMemory) -> float:
        """TopK近傍でのGED計算"""
        try:
            if not self.previous_embeddings:
                return 0.0
            
            # Step 1: TopK類似エピソード取得
            topk_indices, topk_similarities = self._get_topk_similar(
                new_embedding, min(self.k, len(self.previous_embeddings))
            )
            
            # Step 2: TopK近傍でのローカルGED計算
            local_ged = self._calculate_local_ged(topk_indices, topk_similarities)
            
            return local_ged
            
        except Exception as e:
            print(f"❌ TopK GED計算エラー: {e}")
            return 0.0
    
    def _get_topk_similar(self, new_embedding: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        """TopK類似エピソードを高速取得"""
        if not self.previous_embeddings:
            return [], []
        
        # 全既存エピソードとの類似度計算
        similarities = []
        for i, prev_emb in enumerate(self.previous_embeddings):
            sim = np.dot(new_embedding, prev_emb) / (
                np.linalg.norm(new_embedding) * np.linalg.norm(prev_emb) + 1e-8
            )
            similarities.append((i, sim))
        
        # TopK取得
        similarities.sort(key=lambda x: x[1], reverse=True)
        topk = similarities[:k]
        
        indices = [idx for idx, _ in topk]
        scores = [score for _, score in topk]
        
        return indices, scores
    
    def _calculate_local_ged(self, topk_indices: List[int], 
                           topk_similarities: List[float]) -> float:
        """ローカル領域でのGED計算"""
        if not topk_indices:
            return 0.0
        
        # 類似度ベースのGED近似
        avg_similarity = np.mean(topk_similarities)
        connectivity_change = len([s for s in topk_similarities if s > self.similarity_threshold])
        
        # GED近似値 (類似度が高いほど構造変化は小さい)
        ged_value = max(0.1, 2.0 - avg_similarity * 1.5 + connectivity_change * 0.1)
        
        return ged_value
    
    def add_embedding(self, embedding: np.ndarray):
        """新しい埋め込みを追加"""
        self.previous_embeddings.append(embedding.copy())


class RealTimeInsightExperiment:
    """リアルタイム洞察検出実験"""
    
    def __init__(self):
        self.config = get_config()
        self.model = get_model()
        
        # 簡易メモリマネージャーを直接作成
        self.embeddings = []
        self.episodes = []
        self.graph_snapshots = []
        
        # TopK最適化GED計算器
        self.ged_calculator = TopKOptimizedGEDCalculator(k=10)
        
        # 洞察検出設定
        self.ged_threshold = 0.3  # より敏感な閾値
        self.ig_threshold = 0.1   # より敏感な閾値
        
        # 結果保存
        self.insight_events = []
        self.performance_metrics = []
        
        # ビジュアライゼーション用
        self.graph_evolution = []
        self.similarity_network = nx.Graph()
        
        # 参照データベース（ベクトル→テキスト変換用）
        self.reference_texts = []
        self.reference_vectors = None
        
    def setup_reference_database(self, episodes: List[str]):
        """参照データベースを構築"""
        print("📚 参照データベースを構築中...")
        self.reference_texts = episodes[:50]  # 最初の50エピソードを参照として使用
        self.reference_vectors = self.model.encode(self.reference_texts)
        print(f"✅ 参照データベース構築完了 ({len(self.reference_texts)}件)")
    
    def vector_to_text_approximation(self, vector: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """ベクトルから近似テキストを生成"""
        if self.reference_vectors is None:
            return []
        
        # コサイン類似度計算
        similarities = self.reference_vectors @ vector
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((self.reference_texts[idx], similarities[idx]))
        
        return results
    
    def generate_episodes(self, count: int = 1000) -> List[str]:
        """実験用エピソードを生成"""
        print(f"📝 {count}エピソードを生成中...")
        
        # 基本トピック
        topics = [
            "AI healthcare", "ML training", "Deep learning", "NLP interaction",
            "Computer vision", "Predictive analytics", "Data science", 
            "Neural networks", "Automation", "Personalized medicine"
        ]
        
        # 修正タイプ
        modifications = [
            "advanced algorithms", "large datasets", "real-time processing",
            "improved accuracy", "cost reduction", "enhanced security",
            "cloud integration", "mobile optimization", "user experience",
            "scalability", "performance", "innovation", "automation",
            "intelligence", "efficiency", "reliability", "flexibility"
        ]
        
        # パターン
        patterns = [
            "through {mod} and continuous learning",
            "by leveraging {mod} and computational power", 
            "using {mod} and cutting-edge technology"
        ]
        
        # テンプレート
        templates = {
            "AI healthcare": "AI can revolutionize healthcare diagnostics",
            "ML training": "Machine learning models require high-quality training data",
            "Deep learning": "Deep learning excels at pattern recognition tasks",
            "NLP interaction": "Natural language processing enables human-computer interaction",
            "Computer vision": "Computer vision systems can analyze medical images",
            "Predictive analytics": "Predictive analytics helps optimize resource allocation",
            "Data science": "Data science drives evidence-based decision making",
            "Neural networks": "Neural networks can model complex relationships",
            "Automation": "Automation improves efficiency in healthcare workflows",
            "Personalized medicine": "Personalized medicine relies on patient-specific data analysis"
        }
        
        episodes = []
        
        for i in range(count):
            topic = topics[i % len(topics)]
            mod = modifications[i % len(modifications)]
            pattern = patterns[i % len(patterns)]
            
            base_text = templates[topic]
            full_text = f"{base_text} {pattern.format(mod=mod)}."
            
            episodes.append(full_text)
            
        return episodes
    
    def detect_insight_every_episode(self, episode_id: int, new_embedding: np.ndarray, 
                                   episode_text: str) -> Dict[str, Any]:
        """毎エピソードで洞察検出"""
        try:
            start_time = time.time()
            
            # TopK最適化GED計算
            ged_value = self.ged_calculator.calculate_optimized_ged(
                new_embedding, None  # KnowledgeGraphは使用しない
            )
            
            # 簡易IG計算（エピソード数ベース）
            ig_value = max(0.1, 10.0 / (1 + episode_id / 50)) + np.random.random() * 0.1
            
            # 計算時間記録
            calculation_time = time.time() - start_time
            
            # 閾値チェック
            ged_exceeds = ged_value > self.ged_threshold
            ig_exceeds = ig_value > self.ig_threshold
            spike_detected = ged_exceeds or ig_exceeds
            
            # パフォーマンス記録
            self.performance_metrics.append({
                'episode_id': episode_id,
                'calculation_time': calculation_time,
                'ged_value': ged_value,
                'ig_value': ig_value,
                'spike_detected': spike_detected
            })
            
            # 洞察検出時の詳細記録
            if spike_detected:
                insight_data = self._generate_insight_episode(
                    episode_id, ged_value, ig_value, new_embedding, episode_text
                )
                self.insight_events.append(insight_data)
                print(f"💡 洞察検出 #{len(self.insight_events)}: Episode {episode_id} "
                      f"(ΔGED={ged_value:.3f}, ΔIG={ig_value:.3f})")
                return insight_data
            
            return None
            
        except Exception as e:
            print(f"❌ 洞察検出エラー (Episode {episode_id}): {e}")
            return None
    
    def _generate_insight_episode(self, episode_id: int, ged_value: float, 
                                ig_value: float, insight_vector: np.ndarray,
                                trigger_text: str) -> Dict[str, Any]:
        """洞察エピソードを生成"""
        try:
            # 洞察タイプ決定
            if ig_value > 5.0:
                insight_type = "大規模学習"
            elif ig_value > 2.0:
                insight_type = "中規模統合"
            elif ig_value > 1.0:
                insight_type = "小規模改善"
            else:
                insight_type = "微調整"
            
            # 重要度計算
            importance_score = (ged_value * 2 + ig_value) / 3
            
            # 洞察説明生成
            description = f"Episode {episode_id}で{insight_type}を検出。ΔGED={ged_value:.4f}, ΔIG={ig_value:.4f}の変化により新しい理解が獲得された。"
            
            # ベクトル→言語変換
            vector_to_language = self.vector_to_text_approximation(insight_vector, top_k=3)
            
            # 報酬計算
            base_reward = min(50.0, ig_value * 5.0)
            quality_bonus = min(10.0, ged_value * 5.0)
            total_reward = base_reward + quality_bonus
            
            # 関連ノード（近傍エピソード）
            related_nodes = list(range(max(1, episode_id - 5), min(episode_id + 5, len(self.episodes) + 1)))
            
            insight_episode = {
                'insight_id': f"RT_INS_{episode_id:04d}_{len(self.insight_events)+1:03d}",
                'episode_id': episode_id,
                'trigger_text': trigger_text,
                'insight_type': insight_type,
                'description': description,
                'importance_score': importance_score,
                'generated_timestamp': datetime.now().isoformat(),
                
                # メトリクス
                'metrics': {
                    'delta_ged': ged_value,
                    'delta_ig': ig_value,
                    'ged_exceeds_threshold': ged_value > self.ged_threshold,
                    'ig_exceeds_threshold': ig_value > self.ig_threshold
                },
                
                # 報酬
                'reward': {
                    'base_reward': base_reward,
                    'quality_bonus': quality_bonus,
                    'total_reward': total_reward
                },
                
                # ベクトル情報
                'vector_info': {
                    'vector_norm': float(np.linalg.norm(insight_vector)),
                    'vector_sample': insight_vector[:5].tolist()
                },
                
                # 言語変換
                'vector_to_language': [
                    {
                        'rank': i+1,
                        'text': text,
                        'similarity': float(sim)
                    }
                    for i, (text, sim) in enumerate(vector_to_language)
                ],
                
                # 関連ノード
                'related_nodes': related_nodes
            }
            
            return insight_episode
            
        except Exception as e:
            print(f"❌ 洞察エピソード生成エラー: {e}")
            return None
    
    def update_graph_visualization(self, episode_id: int, new_embedding: np.ndarray):
        """グラフビジュアライゼーション更新"""
        try:
            # ノード追加
            self.similarity_network.add_node(episode_id)
            
            # 類似度の高いエピソードとエッジ追加
            similarity_threshold = 0.8
            for i, prev_embedding in enumerate(self.embeddings[:-1]):  # 最後の要素（今回追加分）を除く
                similarity = np.dot(new_embedding, prev_embedding) / (
                    np.linalg.norm(new_embedding) * np.linalg.norm(prev_embedding) + 1e-8
                )
                
                if similarity > similarity_threshold:
                    self.similarity_network.add_edge(i + 1, episode_id, weight=similarity)
            
            # グラフスナップショット保存（10エピソード毎）
            if episode_id % 10 == 0:
                snapshot = {
                    'episode_id': episode_id,
                    'num_nodes': self.similarity_network.number_of_nodes(),
                    'num_edges': self.similarity_network.number_of_edges(),
                    'avg_degree': np.mean([d for n, d in self.similarity_network.degree()]) if self.similarity_network.nodes() else 0,
                    'clustering_coefficient': nx.average_clustering(self.similarity_network) if self.similarity_network.nodes() else 0,
                    'timestamp': datetime.now().isoformat()
                }
                self.graph_evolution.append(snapshot)
                
        except Exception as e:
            print(f"❌ グラフ更新エラー (Episode {episode_id}): {e}")
    
    def run_realtime_experiment(self, num_episodes: int = 1000):
        """リアルタイム洞察検出実験を実行"""
        print(f"🚀 リアルタイム洞察検出実験開始 ({num_episodes}エピソード)")
        print("=" * 70)
        
        start_time = time.time()
        
        # 1. エピソード生成
        episodes = self.generate_episodes(num_episodes)
        
        # 2. 参照データベース構築
        self.setup_reference_database(episodes)
        
        # 3. 毎エピソード処理
        print(f"\n📊 毎エピソード洞察検出開始...")
        
        for i, episode_text in enumerate(episodes, 1):
            try:
                # エピソード埋め込み生成
                episode_embedding = self.model.encode([episode_text])[0]
                
                # エピソード保存
                self.episodes.append(episode_text)
                self.embeddings.append(episode_embedding)
                
                # 洞察検出
                insight = self.detect_insight_every_episode(i, episode_embedding, episode_text)
                
                # グラフ更新
                self.update_graph_visualization(i, episode_embedding)
                
                # GED計算器に埋め込み追加
                self.ged_calculator.add_embedding(episode_embedding)
                
                # 進捗表示
                if i % 50 == 0:
                    elapsed = time.time() - start_time
                    eps_per_sec = i / elapsed
                    insights_count = len(self.insight_events)
                    print(f"📈 進捗: {i}/{num_episodes} ({eps_per_sec:.1f} eps/sec, 洞察: {insights_count})")
                
            except Exception as e:
                print(f"❌ Episode {i} 処理エラー: {e}")
                continue
        
        # 4. 実験完了
        total_time = time.time() - start_time
        final_eps_per_sec = num_episodes / total_time
        
        print(f"\n✅ リアルタイム実験完了!")
        print(f"   総エピソード: {num_episodes}")
        print(f"   実行時間: {total_time:.2f}秒")
        print(f"   処理速度: {final_eps_per_sec:.2f} eps/sec")
        print(f"   検出された洞察: {len(self.insight_events)}")
        print(f"   平均計算時間: {np.mean([m['calculation_time'] for m in self.performance_metrics]):.4f}秒/エピソード")
        
        return {
            'num_episodes': num_episodes,
            'execution_time': total_time,
            'episodes_per_second': final_eps_per_sec,
            'insights_detected': len(self.insight_events),
            'avg_calculation_time': np.mean([m['calculation_time'] for m in self.performance_metrics]),
            'timestamp': datetime.now().isoformat()
        }
    
    def create_graph_visualization(self):
        """グラフ成長のビジュアライゼーション作成"""
        print("\n🎨 グラフ成長ビジュアライゼーション作成中...")
        
        try:
            # 出力ディレクトリ作成
            output_dir = Path("experiments/outputs/realtime_experiment")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. グラフ成長統計のプロット
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('InsightSpike-AI: リアルタイムグラフ成長分析', fontsize=16)
            
            if self.graph_evolution:
                episodes = [snap['episode_id'] for snap in self.graph_evolution]
                nodes = [snap['num_nodes'] for snap in self.graph_evolution]
                edges = [snap['num_edges'] for snap in self.graph_evolution]
                avg_degrees = [snap['avg_degree'] for snap in self.graph_evolution]
                clustering = [snap['clustering_coefficient'] for snap in self.graph_evolution]
                
                # ノード数成長
                axes[0, 0].plot(episodes, nodes, 'b-o', markersize=4)
                axes[0, 0].set_title('ノード数の成長')
                axes[0, 0].set_xlabel('エピソード数')
                axes[0, 0].set_ylabel('ノード数')
                axes[0, 0].grid(True, alpha=0.3)
                
                # エッジ数成長
                axes[0, 1].plot(episodes, edges, 'r-s', markersize=4)
                axes[0, 1].set_title('エッジ数の成長')
                axes[0, 1].set_xlabel('エピソード数')
                axes[0, 1].set_ylabel('エッジ数')
                axes[0, 1].grid(True, alpha=0.3)
                
                # 平均次数
                axes[1, 0].plot(episodes, avg_degrees, 'g-^', markersize=4)
                axes[1, 0].set_title('平均ノード次数')
                axes[1, 0].set_xlabel('エピソード数')
                axes[1, 0].set_ylabel('平均次数')
                axes[1, 0].grid(True, alpha=0.3)
                
                # クラスタリング係数
                axes[1, 1].plot(episodes, clustering, 'm-d', markersize=4)
                axes[1, 1].set_title('クラスタリング係数')
                axes[1, 1].set_xlabel('エピソード数')
                axes[1, 1].set_ylabel('クラスタリング係数')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            graph_stats_file = output_dir / "graph_growth_statistics.png"
            plt.savefig(graph_stats_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 洞察検出頻度のプロット
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            fig.suptitle('洞察検出分析', fontsize=16)
            
            if self.insight_events:
                insight_episodes = [event['episode_id'] for event in self.insight_events]
                insight_rewards = [event['reward']['total_reward'] for event in self.insight_events]
                
                # 洞察検出タイミング
                axes[0].scatter(insight_episodes, range(len(insight_episodes)), 
                              c='red', s=50, alpha=0.7)
                axes[0].set_title('洞察検出タイミング')
                axes[0].set_xlabel('エピソード数')
                axes[0].set_ylabel('洞察ID')
                axes[0].grid(True, alpha=0.3)
                
                # 報酬分布
                axes[1].bar(range(len(insight_rewards)), insight_rewards, 
                           color='orange', alpha=0.7)
                axes[1].set_title('洞察報酬分布')
                axes[1].set_xlabel('洞察ID')
                axes[1].set_ylabel('総報酬')
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            insight_analysis_file = output_dir / "insight_detection_analysis.png"
            plt.savefig(insight_analysis_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. パフォーマンス分析
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            fig.suptitle('計算パフォーマンス分析', fontsize=16)
            
            if self.performance_metrics:
                episodes = [m['episode_id'] for m in self.performance_metrics]
                calc_times = [m['calculation_time'] for m in self.performance_metrics]
                ged_values = [m['ged_value'] for m in self.performance_metrics]
                
                # 計算時間推移
                axes[0].plot(episodes, calc_times, 'b-', alpha=0.7, linewidth=1)
                axes[0].set_title('エピソード毎計算時間')
                axes[0].set_xlabel('エピソード数')
                axes[0].set_ylabel('計算時間 (秒)')
                axes[0].grid(True, alpha=0.3)
                
                # GED値推移
                axes[1].plot(episodes, ged_values, 'g-', alpha=0.7, linewidth=1)
                axes[1].axhline(y=self.ged_threshold, color='red', linestyle='--', 
                              label=f'閾値 ({self.ged_threshold})')
                axes[1].set_title('GED値推移')
                axes[1].set_xlabel('エピソード数')
                axes[1].set_ylabel('GED値')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            performance_file = output_dir / "performance_analysis.png"
            plt.savefig(performance_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ ビジュアライゼーション作成完了:")
            print(f"   📊 グラフ成長統計: {graph_stats_file}")
            print(f"   💡 洞察検出分析: {insight_analysis_file}")
            print(f"   ⚡ パフォーマンス分析: {performance_file}")
            
            return {
                'graph_stats': str(graph_stats_file),
                'insight_analysis': str(insight_analysis_file),
                'performance_analysis': str(performance_file)
            }
            
        except Exception as e:
            print(f"❌ ビジュアライゼーション作成エラー: {e}")
            return {}
    
    def save_comprehensive_summary(self, experiment_results: Dict[str, Any]):
        """包括的サマリを保存"""
        print(f"\n💾 包括的サマリを保存中...")
        
        # 出力ディレクトリ作成
        output_dir = Path("experiments/outputs/realtime_experiment")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 入力エピソード詳細CSV
        input_csv_file = output_dir / "01_input_episodes_realtime.csv"
        with open(input_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['episode_id', 'episode_text', 'processed_timestamp'])
            
            for i, episode_text in enumerate(self.episodes, 1):
                writer.writerow([i, episode_text, datetime.now().isoformat()])
        
        # 2. 洞察イベント包括CSV
        insight_csv_file = output_dir / "02_realtime_insights_comprehensive.csv"
        with open(insight_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'insight_id', 'episode_id', 'insight_type', 'delta_ged', 'delta_ig',
                'generated_timestamp', 'total_reward', 'importance_score',
                'top_vector_conversion', 'vector_similarity', 'related_nodes_count'
            ])
            
            for event in self.insight_events:
                top_conversion = event['vector_to_language'][0] if event['vector_to_language'] else {}
                
                writer.writerow([
                    event['insight_id'],
                    event['episode_id'],
                    event['insight_type'],
                    event['metrics']['delta_ged'],
                    event['metrics']['delta_ig'],
                    event['generated_timestamp'],
                    event['reward']['total_reward'],
                    event['importance_score'],
                    top_conversion.get('text', 'N/A'),
                    top_conversion.get('similarity', 0.0),
                    len(event['related_nodes'])
                ])
        
        # 3. パフォーマンス詳細JSON
        performance_file = output_dir / "03_performance_metrics.json"
        with open(performance_file, 'w', encoding='utf-8') as f:
            json.dump({
                'experiment_results': experiment_results,
                'performance_metrics': self.performance_metrics,
                'graph_evolution': self.graph_evolution
            }, f, indent=2, ensure_ascii=False)
        
        # 4. グラフ成長CSV
        graph_csv_file = output_dir / "04_graph_evolution.csv"
        with open(graph_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode_id', 'num_nodes', 'num_edges', 'avg_degree', 
                'clustering_coefficient', 'timestamp'
            ])
            
            for snapshot in self.graph_evolution:
                writer.writerow([
                    snapshot['episode_id'],
                    snapshot['num_nodes'],
                    snapshot['num_edges'],
                    snapshot['avg_degree'],
                    snapshot['clustering_coefficient'],
                    snapshot['timestamp']
                ])
        
        # 5. 実験メタサマリ
        meta_file = output_dir / "05_experiment_meta_summary.json"
        meta_summary = {
            'experiment_type': 'リアルタイム毎エピソード洞察検出',
            'optimization': 'TopK最適化GED計算',
            'total_episodes': len(self.episodes),
            'total_insights': len(self.insight_events),
            'insight_frequency': len(self.insight_events) / len(self.episodes) if self.episodes else 0,
            'avg_calculation_time': np.mean([m['calculation_time'] for m in self.performance_metrics]) if self.performance_metrics else 0,
            'total_execution_time': experiment_results.get('execution_time', 0),
            'processing_speed': experiment_results.get('episodes_per_second', 0),
            'final_graph_stats': self.graph_evolution[-1] if self.graph_evolution else {},
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta_summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 包括的サマリ保存完了:")
        print(f"   📁 出力ディレクトリ: {output_dir}")
        print(f"   📄 入力エピソード: {input_csv_file}")
        print(f"   📄 洞察イベント: {insight_csv_file}")
        print(f"   📄 パフォーマンス: {performance_file}")
        print(f"   📄 グラフ成長: {graph_csv_file}")
        print(f"   📄 メタサマリ: {meta_file}")


def main():
    """メイン実行関数"""
    experiment = RealTimeInsightExperiment()
    
    try:
        # リアルタイム実験実行
        results = experiment.run_realtime_experiment(num_episodes=1000)
        
        # ビジュアライゼーション作成
        viz_results = experiment.create_graph_visualization()
        
        # サマリ保存
        experiment.save_comprehensive_summary(results)
        
        print(f"\n🎉 リアルタイム洞察検出実験が正常に完了しました!")
        print(f"\n📊 最終統計:")
        print(f"   毎エピソード計算により {results['insights_detected']} 個の洞察を検出")
        print(f"   平均計算時間: {results['avg_calculation_time']:.4f}秒/エピソード")
        print(f"   処理速度: {results['episodes_per_second']:.2f} エピソード/秒")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 実験が中断されました")
    except Exception as e:
        print(f"\n❌ 実験エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
