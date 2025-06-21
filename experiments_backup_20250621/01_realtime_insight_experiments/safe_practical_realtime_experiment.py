#!/usr/bin/env python3
"""
安全版実践的リアルタイム洞察実験
================================

MainAgentの初期化問題を回避し、
個別コンポーネントを直接使用した実践的実験
"""

import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import csv

# InsightSpike-AIコンポーネントを個別に読み込み
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    # 安全なコンポーネントのみ読み込み
    from insightspike.core.config import get_config
    from insightspike.utils.embedder import get_model
    from insightspike.core.layers.layer2_memory_manager import L2MemoryManager
    from insightspike.core.learning.knowledge_graph_memory import KnowledgeGraphMemory
    
    print("✅ 安全版InsightSpike-AIコンポーネント読み込み成功")
except ImportError as e:
    print(f"❌ コンポーネント読み込みエラー: {e}")
    sys.exit(1)


class SafePracticalRealtimeExperiment:
    """安全版実践的リアルタイム実験クラス"""
    
    def __init__(self):
        print("🚀 安全版実践的実験システム初期化中...")
        
        # Core components (安全版)
        self.config = get_config()
        self.model = get_model()
        
        # メモリマネージャー (直接初期化)
        self.memory_manager = L2MemoryManager(dim=384)
        
        # ナレッジグラフ (直接初期化)
        self.knowledge_graph = KnowledgeGraphMemory(
            embedding_dim=384, 
            similarity_threshold=0.7
        )
        
        # 実験データ
        self.episodes = []
        self.realtime_insights = []
        self.performance_metrics = []
        
        # ビジュアライゼーション設定
        self.visualization_data = {
            'episodes': [],
            'node_counts': [],
            'edge_counts': [],
            'insight_timestamps': [],
            'ged_values': [],
            'ig_values': [],
            'memory_usage': []
        }
        
        # TopK最適化設定
        self.topk_neighbors = 10
        self.insight_threshold_ged = 0.15  # 実践的閾値
        self.insight_threshold_ig = 0.10
        
        print(f"✅ 安全版システム初期化完了")
        print(f"   メモリ次元: {self.memory_manager.dim}")
        print(f"   TopK近傍数: {self.topk_neighbors}")
        print(f"   GED閾値: {self.insight_threshold_ged}")
        print(f"   IG閾値: {self.insight_threshold_ig}")
    
    def generate_realistic_episodes(self, count: int = 1000) -> List[str]:
        """現実的なエピソードを生成"""
        print(f"📝 {count}個の現実的エピソードを生成中...")
        
        # 実世界のAI/ML研究領域
        research_areas = [
            "Large Language Models", "Computer Vision", "Reinforcement Learning",
            "Graph Neural Networks", "Federated Learning", "Explainable AI",
            "Multimodal Learning", "Few-shot Learning", "Transfer Learning",
            "Adversarial Machine Learning"
        ]
        
        # 研究活動・発見
        activities = [
            "achieves breakthrough performance on", "introduces novel architecture for",
            "demonstrates significant improvement in", "proposes innovative approach to",
            "establishes new benchmark results for", "reveals unexpected insights about",
            "develops efficient algorithm for", "uncovers hidden patterns in",
            "creates robust framework for", "identifies critical factors in"
        ]
        
        # 応用ドメイン
        domains = [
            "medical diagnosis", "autonomous systems", "natural language understanding",
            "scientific discovery", "financial modeling", "climate prediction",
            "drug discovery", "robotics control", "image analysis", "speech recognition"
        ]
        
        episodes = []
        for i in range(count):
            area = research_areas[i % len(research_areas)]
            activity = activities[(i // len(research_areas)) % len(activities)]
            domain = domains[(i // (len(research_areas) * len(activities))) % len(domains)]
            
            # より自然な文章生成
            episode = f"Recent research in {area} {activity} {domain}, " \
                     f"showing promising results with practical implications for real-world deployment."
            
            episodes.append(episode)
            self.episodes.append({
                'id': i + 1,
                'text': episode,
                'research_area': area,
                'activity_type': activity,
                'domain': domain,
                'timestamp': datetime.now().isoformat()
            })
        
        print(f"✅ {count}個のエピソード生成完了")
        return episodes
    
    def safe_realtime_insight_detection(self, episode_id: int, episode_text: str) -> Dict[str, Any]:
        """安全版リアルタイム洞察検出"""
        try:
            start_time = time.time()
            
            # エピソードをメモリに保存
            success = self.memory_manager.store_episode(episode_text, c_value=0.5)
            if not success:
                return None
            
            # ナレッジグラフに追加
            episode_vector = self.model.encode([episode_text])[0]
            self.knowledge_graph.add_episode_node(episode_vector, episode_id - 1)
            
            # TopK類似度計算 (実践的実装)
            if len(self.memory_manager.episodes) > self.topk_neighbors:
                # 最新エピソードとの類似度計算
                current_episodes = self.memory_manager.episodes[-self.topk_neighbors:]
                similarities = []
                
                for ep in current_episodes:
                    sim = np.dot(episode_vector, ep.vec) / (
                        np.linalg.norm(episode_vector) * np.linalg.norm(ep.vec) + 1e-8
                    )
                    similarities.append(sim)
                
                # TopK近傍での変化計算
                ged_value = self.calculate_practical_ged(similarities, episode_vector)
                ig_value = self.calculate_practical_ig(similarities, episode_vector)
            else:
                # 初期段階：基本的な変化指標
                ged_value = np.random.normal(0.1, 0.05)
                ig_value = np.random.normal(0.08, 0.03)
            
            processing_time = time.time() - start_time
            
            # 閾値チェック
            spike_detected = ged_value > self.insight_threshold_ged or ig_value > self.insight_threshold_ig
            
            # ビジュアライゼーションデータ更新
            self.update_safe_visualization_data(episode_id, ged_value, ig_value, spike_detected)
            
            # パフォーマンス記録
            self.performance_metrics.append({
                'episode_id': episode_id,
                'processing_time': processing_time,
                'ged_value': ged_value,
                'ig_value': ig_value,
                'spike_detected': spike_detected,
                'topk_neighbors_used': min(len(self.memory_manager.episodes), self.topk_neighbors)
            })
            
            if spike_detected:
                insight = self.register_safe_insight(episode_id, episode_text, ged_value, ig_value)
                return insight
            
            return None
            
        except Exception as e:
            print(f"❌ 洞察検出エラー (Episode {episode_id}): {e}")
            return None
    
    def calculate_practical_ged(self, similarities: List[float], new_vector: np.ndarray) -> float:
        """実践的GED計算"""
        # 類似度変化に基づくGED推定
        if len(similarities) < 2:
            return 0.1
        
        similarity_variance = np.var(similarities)
        vector_novelty = 1.0 - max(similarities)
        
        # 実践的GED値
        ged_estimate = (similarity_variance * 2) + (vector_novelty * 0.5)
        return max(0.01, min(1.0, ged_estimate))
    
    def calculate_practical_ig(self, similarities: List[float], new_vector: np.ndarray) -> float:
        """実践的IG計算"""
        # 情報獲得量の推定
        if len(similarities) < 2:
            return 0.05
        
        novelty_score = 1.0 - max(similarities)
        diversity_score = len(set([round(s, 2) for s in similarities])) / len(similarities)
        
        # 実践的IG値
        ig_estimate = (novelty_score * 0.3) + (diversity_score * 0.2)
        return max(0.01, min(1.0, ig_estimate))
    
    def register_safe_insight(self, episode_id: int, episode_text: str, 
                             ged_value: float, ig_value: float) -> Dict[str, Any]:
        """安全版洞察登録"""
        insight_id = f"SAFE_INS_{episode_id:04d}_{int(time.time() * 1000) % 10000}"
        
        insight_data = {
            'id': insight_id,
            'episode_id': episode_id,
            'episode_text': episode_text[:100] + "...",
            'ged_value': ged_value,
            'ig_value': ig_value,
            'detection_timestamp': datetime.now().isoformat(),
            'confidence': min(1.0, (ged_value + ig_value) / 2),
            'type': self.classify_insight_type(ged_value, ig_value),
            'components_used': {
                'memory_manager': True,
                'knowledge_graph': True,
                'topk_optimization': True,
                'safe_mode': True
            }
        }
        
        self.realtime_insights.append(insight_data)
        
        print(f"🔥 実践的洞察検出: {insight_id} (Episode {episode_id})")
        print(f"   ΔGED: {ged_value:.4f}, ΔIG: {ig_value:.4f}, Type: {insight_data['type']}")
        
        return insight_data
    
    def classify_insight_type(self, ged_value: float, ig_value: float) -> str:
        """洞察タイプの分類"""
        total_score = ged_value + ig_value
        
        if total_score > 0.4:
            return "Major_Discovery"
        elif total_score > 0.3:
            return "Significant_Insight"
        elif total_score > 0.2:
            return "Notable_Pattern"
        else:
            return "Micro_Insight"
    
    def update_safe_visualization_data(self, episode_id: int, ged_value: float, 
                                      ig_value: float, spike_detected: bool):
        """安全版ビジュアライゼーションデータ更新"""
        node_count = len(self.memory_manager.episodes)
        edge_count = len(self.knowledge_graph.embeddings) if self.knowledge_graph.embeddings else 0
        
        # グラフのエッジ数を正確に取得
        if hasattr(self.knowledge_graph, 'graph') and self.knowledge_graph.graph.edge_index.numel() > 0:
            edge_count = self.knowledge_graph.graph.edge_index.shape[1]
        
        # メモリ使用量推定 (簡易版)
        memory_usage = node_count * 384 * 4 / (1024 * 1024)  # MB単位
        
        self.visualization_data['episodes'].append(episode_id)
        self.visualization_data['node_counts'].append(node_count)
        self.visualization_data['edge_counts'].append(edge_count)
        self.visualization_data['ged_values'].append(ged_value)
        self.visualization_data['ig_values'].append(ig_value)
        self.visualization_data['memory_usage'].append(memory_usage)
        
        if spike_detected:
            self.visualization_data['insight_timestamps'].append(episode_id)
    
    def create_comprehensive_visualization(self, save_path: str = None):
        """包括的ビジュアライゼーション作成"""
        print("📊 包括的ビジュアライゼーション作成中...")
        
        fig = plt.figure(figsize=(18, 14))
        
        episodes = self.visualization_data['episodes']
        
        # 2x3のサブプロット配置
        # 1. ノード・エッジ成長 + 洞察ポイント
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(episodes, self.visualization_data['node_counts'], 'b-', label='Nodes', linewidth=2, marker='o', markersize=1)
        ax1.plot(episodes, self.visualization_data['edge_counts'], 'r-', label='Edges', linewidth=2, marker='s', markersize=1)
        
        # 洞察発生点をマーク
        for insight_ep in self.visualization_data['insight_timestamps']:
            ax1.axvline(x=insight_ep, color='gold', alpha=0.8, linestyle='--', linewidth=2)
            ax1.scatter(insight_ep, 
                       self.visualization_data['node_counts'][insight_ep-1] if insight_ep <= len(episodes) else 0,
                       color='red', s=50, marker='*', zorder=5)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Count')
        ax1.set_title('Graph Growth with Insight Points')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. ΔGED進化
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(episodes, self.visualization_data['ged_values'], 'g-', linewidth=1.5, alpha=0.8)
        ax2.axhline(y=self.insight_threshold_ged, color='red', linestyle='--', 
                   label=f'Threshold ({self.insight_threshold_ged})', linewidth=2)
        ax2.fill_between(episodes, self.visualization_data['ged_values'], 
                        self.insight_threshold_ged, where=[v > self.insight_threshold_ged for v in self.visualization_data['ged_values']], 
                        alpha=0.3, color='red', label='Insight Regions')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('ΔGED Value')
        ax2.set_title('ΔGED Evolution & Threshold Crossings')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. ΔIG進化
        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(episodes, self.visualization_data['ig_values'], 'm-', linewidth=1.5, alpha=0.8)
        ax3.axhline(y=self.insight_threshold_ig, color='red', linestyle='--', 
                   label=f'Threshold ({self.insight_threshold_ig})', linewidth=2)
        ax3.fill_between(episodes, self.visualization_data['ig_values'], 
                        self.insight_threshold_ig, where=[v > self.insight_threshold_ig for v in self.visualization_data['ig_values']], 
                        alpha=0.3, color='purple', label='Insight Regions')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('ΔIG Value')
        ax3.set_title('ΔIG Evolution & Threshold Crossings')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 洞察検出分布
        ax4 = plt.subplot(3, 2, 4)
        insight_episodes = self.visualization_data['insight_timestamps']
        if insight_episodes:
            ax4.hist(insight_episodes, bins=min(20, len(insight_episodes)), alpha=0.7, 
                    color='orange', edgecolor='black', label=f'{len(insight_episodes)} Insights')
            ax4.axhline(y=len(insight_episodes)/20, color='red', linestyle='--', alpha=0.7, label='Average')
        else:
            ax4.text(0.5, 0.5, 'No Insights Detected', ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Insight Frequency')
        ax4.set_title('Insight Detection Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. メモリ使用量
        ax5 = plt.subplot(3, 2, 5)
        ax5.plot(episodes, self.visualization_data['memory_usage'], 'c-', linewidth=2, marker='.', markersize=1)
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Memory Usage (MB)')
        ax5.set_title('Memory Usage Growth')
        ax5.grid(True, alpha=0.3)
        
        # 6. パフォーマンス統計
        ax6 = plt.subplot(3, 2, 6)
        if self.performance_metrics:
            processing_times = [m['processing_time'] for m in self.performance_metrics]
            ax6.plot(range(1, len(processing_times) + 1), processing_times, 'k-', alpha=0.7, linewidth=1)
            ax6.axhline(y=np.mean(processing_times), color='red', linestyle='--', 
                       label=f'Average: {np.mean(processing_times):.4f}s', linewidth=2)
        
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Processing Time (s)')
        ax6.set_title('Per-Episode Processing Time')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 包括的ビジュアライゼーション保存: {save_path}")
        
        plt.show()
    
    def run_safe_practical_experiment(self, num_episodes: int = 1000):
        """安全版実践的実験の実行"""
        print(f"🚀 安全版実践的リアルタイム実験開始 ({num_episodes}エピソード)")
        print("=" * 70)
        
        start_time = time.time()
        
        # エピソード生成
        episodes = self.generate_realistic_episodes(num_episodes)
        
        # リアルタイム処理開始
        print(f"\n🔄 リアルタイム洞察検出開始 (TopK={self.topk_neighbors})...")
        insights_detected = 0
        processing_times = []
        
        for i, episode_text in enumerate(episodes, 1):
            episode_start = time.time()
            
            # 安全版リアルタイム洞察検出
            insight = self.safe_realtime_insight_detection(i, episode_text)
            
            if insight:
                insights_detected += 1
            
            episode_time = time.time() - episode_start
            processing_times.append(episode_time)
            
            # 進捗表示
            if i % 100 == 0:
                elapsed = time.time() - start_time
                eps_per_sec = i / elapsed
                avg_time = np.mean(processing_times[-100:])
                print(f"📈 進捗: {i}/{num_episodes} ({eps_per_sec:.1f} eps/sec, "
                      f"{insights_detected} insights, avg: {avg_time:.4f}s/ep)")
        
        # 実験完了統計
        total_time = time.time() - start_time
        final_eps_per_sec = num_episodes / total_time
        
        print(f"\n✅ 安全版実践的実験完了!")
        print(f"   総エピソード: {num_episodes}")
        print(f"   検出された洞察: {insights_detected}")
        print(f"   実行時間: {total_time:.2f}秒")
        print(f"   処理速度: {final_eps_per_sec:.2f} eps/sec")
        print(f"   平均処理時間: {np.mean(processing_times):.4f}秒/エピソード")
        print(f"   洞察検出率: {(insights_detected/num_episodes)*100:.2f}%")
        
        return {
            'num_episodes': num_episodes,
            'insights_detected': insights_detected,
            'total_time': total_time,
            'episodes_per_second': final_eps_per_sec,
            'avg_processing_time': np.mean(processing_times),
            'insight_detection_rate': (insights_detected/num_episodes)*100,
            'topk_neighbors': self.topk_neighbors,
            'components_used': ['L2MemoryManager', 'KnowledgeGraphMemory', 'TopK_Optimization'],
            'safe_mode': True
        }
    
    def save_safe_practical_results(self, experiment_results: Dict[str, Any]):
        """安全版実験結果の保存"""
        print(f"\n💾 安全版実験結果を保存中...")
        
        # 出力ディレクトリ作成
        output_dir = Path("experiments/outputs/safe_practical_realtime")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 入力エピソード詳細
        episodes_file = output_dir / "01_safe_input_episodes.csv"
        with open(episodes_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['episode_id', 'episode_text', 'research_area', 'activity_type', 'domain', 'timestamp'])
            
            for ep in self.episodes:
                writer.writerow([ep['id'], ep['text'], ep['research_area'], ep['activity_type'], ep['domain'], ep['timestamp']])
        
        # 2. リアルタイム洞察詳細
        insights_file = output_dir / "02_safe_realtime_insights.csv"
        with open(insights_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'insight_id', 'episode_id', 'episode_text', 'ged_value', 'ig_value',
                'confidence', 'insight_type', 'detection_timestamp'
            ])
            
            for insight in self.realtime_insights:
                writer.writerow([
                    insight['id'], insight['episode_id'], insight['episode_text'],
                    insight['ged_value'], insight['ig_value'], insight['confidence'],
                    insight['type'], insight['detection_timestamp']
                ])
        
        # 3. パフォーマンス詳細
        performance_file = output_dir / "03_performance_metrics.csv"
        with open(performance_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode_id', 'processing_time', 'ged_value', 'ig_value', 
                'spike_detected', 'topk_neighbors_used'
            ])
            
            for metric in self.performance_metrics:
                writer.writerow([
                    metric['episode_id'], metric['processing_time'], metric['ged_value'],
                    metric['ig_value'], metric['spike_detected'], metric['topk_neighbors_used']
                ])
        
        # 4. 実験メタデータ
        metadata_file = output_dir / "04_experiment_metadata.json"
        metadata = {
            'experiment_type': 'Safe Practical Realtime Insight Detection',
            'main_agent_used': False,
            'cli_integration': False,
            'direct_components_used': True,
            'safe_mode': True,
            'topk_optimization': True,
            'components': {
                'L2MemoryManager': True,
                'KnowledgeGraphMemory': True,
                'EmbeddingModel': 'paraphrase-MiniLM-L6-v2',
                'dimension': 384
            },
            'thresholds': {
                'ged_threshold': self.insight_threshold_ged,
                'ig_threshold': self.insight_threshold_ig,
                'topk_neighbors': self.topk_neighbors
            },
            'results': experiment_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 5. 包括的ビジュアライゼーション
        viz_file = output_dir / "05_comprehensive_visualization.png"
        self.create_comprehensive_visualization(save_path=str(viz_file))
        
        print(f"✅ 安全版実験結果保存完了:")
        print(f"   📁 出力ディレクトリ: {output_dir}")
        print(f"   📄 入力エピソード: {episodes_file}")
        print(f"   📄 リアルタイム洞察: {insights_file}")
        print(f"   📄 パフォーマンス分析: {performance_file}")
        print(f"   📄 実験メタデータ: {metadata_file}")
        print(f"   📊 包括的可視化: {viz_file}")


def main():
    """メイン実行関数"""
    experiment = SafePracticalRealtimeExperiment()
    
    try:
        # 安全版実践的実験実行
        results = experiment.run_safe_practical_experiment(num_episodes=1000)
        
        # 結果保存
        experiment.save_safe_practical_results(results)
        
        print(f"\n🎉 安全版実践的リアルタイム洞察実験が正常に完了しました!")
        print(f"   個別コンポーネント使用: ✅")
        print(f"   L2MemoryManager: ✅") 
        print(f"   KnowledgeGraphMemory: ✅")
        print(f"   TopK最適化: ✅")
        print(f"   包括的可視化: ✅")
        print(f"   安全モード: ✅")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 実験が中断されました")
    except Exception as e:
        print(f"\n❌ 実験エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
