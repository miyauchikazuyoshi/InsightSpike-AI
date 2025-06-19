#!/usr/bin/env python3
"""
標準化実験実行スクリプト
====================

同一条件での対照実験を実現する統一的な実験実行ツール
"""

import sys
import argparse
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# InsightSpike-AIコンポーネントを読み込み
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

try:
    from insightspike.core.config import get_config
    from insightspike.utils.embedder import get_model
    from insightspike.core.layers.layer2_memory_manager import L2MemoryManager
    from insightspike.core.learning.knowledge_graph_memory import KnowledgeGraphMemory
except ImportError as e:
    print(f"❌ InsightSpike-AIコンポーネント読み込みエラー: {e}")
    sys.exit(1)


class StandardizedExperiment:
    """標準化実験クラス"""
    
    def __init__(self, session_id: str, experiment_config: Dict[str, Any]):
        self.session_id = session_id
        self.config = experiment_config
        
        # 出力ディレクトリ
        self.outputs_dir = Path("experiments/outputs")
        self.session_dir = self.outputs_dir / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # データディレクトリ
        self.data_dir = Path("data")
        
        # 実験パラメータ
        self.memory_dim = self.config.get("memory_dim", 384)
        self.topk = self.config.get("topk", 10)
        self.ged_threshold = self.config.get("ged_threshold", 0.15)
        self.ig_threshold = self.config.get("ig_threshold", 0.10)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.3)
        
        # 実験データ
        self.experiment_logs = []
        self.insight_logs = []
        self.memory_snapshots = []
        self.topk_logs = []
        
        print(f"📋 標準化実験初期化完了")
        print(f"   セッション: {session_id}")
        print(f"   メモリ次元: {self.memory_dim}")
        print(f"   TopK: {self.topk}")
        print(f"   GED閾値: {self.ged_threshold}")
        print(f"   IG閾値: {self.ig_threshold}")
    
    def initialize_components(self):
        """コンポーネント初期化"""
        self.model = get_model()
        self.memory_manager = L2MemoryManager(dim=self.memory_dim)
        self.knowledge_graph = KnowledgeGraphMemory(
            embedding_dim=self.memory_dim,
            similarity_threshold=self.similarity_threshold
        )
        print("✅ コンポーネント初期化完了")
    
    def generate_episodes(self, num_episodes: int, seed: int = 42, episode_type: str = "experiment") -> List[Dict]:
        """エピソード生成（再現可能）"""
        print(f"📝 {num_episodes}個のエピソードを生成中 (seed={seed}, type={episode_type})...")
        
        import random
        random.seed(seed)
        np.random.seed(seed)
        
        research_areas = [
            "Large Language Models", "Computer Vision", "Reinforcement Learning",
            "Graph Neural Networks", "Federated Learning", "Explainable AI",
            "Multimodal Learning", "Few-shot Learning", "Transfer Learning",
            "Adversarial Machine Learning"
        ]
        
        activity_types = [
            "achieves breakthrough performance on",
            "introduces novel architecture for", 
            "demonstrates significant improvements in",
            "reveals new insights about",
            "establishes new benchmarks for"
        ]
        
        domains = [
            "medical diagnosis", "autonomous systems", "natural language processing",
            "computer vision", "robotics", "cybersecurity", "climate modeling",
            "drug discovery", "financial prediction", "educational technology"
        ]
        
        episodes = []
        for i in range(1, num_episodes + 1):
            research_area = research_areas[(i - 1) % len(research_areas)]
            activity_type = activity_types[(i - 1) % len(activity_types)]
            domain = domains[(i - 1) % len(domains)]
            
            if episode_type == "initial":
                text = f"Initial research in {research_area} {activity_type} {domain}, establishing foundational knowledge for future insights."
            else:
                text = f"Recent research in {research_area} {activity_type} {domain}, showing promising results with practical implications for real-world deployment."
            
            episodes.append({
                'id': i,
                'text': text,
                'research_area': research_area,
                'activity_type': activity_type,
                'domain': domain,
                'type': episode_type,
                'seed': seed,
                'timestamp': datetime.now().isoformat()
            })
        
        print(f"✅ {len(episodes)}個のエピソード生成完了")
        return episodes
    
    def take_memory_snapshot(self, phase: str, episode_num: int) -> Dict[str, Any]:
        """メモリ状態スナップショット取得"""
        try:
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'phase': phase,
                'episode_number': episode_num,
                'total_episodes': len(self.memory_manager.episodes),
                'memory_size_mb': sys.getsizeof(self.memory_manager) / (1024 * 1024),
                'config': {
                    'memory_dim': self.memory_dim,
                    'topk': self.topk,
                    'ged_threshold': self.ged_threshold,
                    'ig_threshold': self.ig_threshold
                }
            }
            
            # メモリ統計
            if len(self.memory_manager.episodes) > 0:
                c_values = [ep.c_value for ep in self.memory_manager.episodes if hasattr(ep, 'c_value')]
                if c_values:
                    snapshot['c_value_stats'] = {
                        'mean': float(np.mean(c_values)),
                        'std': float(np.std(c_values)),
                        'min': float(np.min(c_values)),
                        'max': float(np.max(c_values))
                    }
            
            self.memory_snapshots.append(snapshot)
            return snapshot
            
        except Exception as e:
            print(f"⚠️ スナップショット取得エラー: {e}")
            return {}
    
    def calculate_ged_ig_metrics(self, current_embedding: np.ndarray, episode_num: int) -> tuple[float, float]:
        """GEDとIG値を計算"""
        try:
            if len(self.memory_manager.episodes) < 2:
                return 0.5, 0.0
            
            # 直近のエピソードとの類似度を計算
            prev_episode = self.memory_manager.episodes[-1]
            similarity = np.dot(current_embedding, prev_episode.vec)
            
            # GED: グローバル編集距離（類似度の逆数として近似）
            ged = max(0.0, 1.0 - similarity)
            
            # IG: 情報ゲイン（エピソード数に基づく簡易計算）
            ig = min(0.3, episode_num * 0.001)
            
            return float(ged), float(ig)
            
        except Exception as e:
            print(f"⚠️ GED/IG計算エラー: {e}")
            return 0.5, 0.0
    
    def check_insight_condition(self, ged: float, ig: float, use_random: bool = True) -> bool:
        """洞察条件をチェック"""
        ged_condition = ged > self.ged_threshold
        ig_condition = ig > self.ig_threshold
        
        if use_random:
            # 確率的要素を加える（再現性のため固定シード使用）
            random_factor = np.random.random() < 0.05  # 5%の確率
            return ged_condition and ig_condition or random_factor
        else:
            return ged_condition and ig_condition
    
    def get_topk_similar_episodes(self, current_episode: Dict, embedding: np.ndarray) -> List[Dict]:
        """TopK類似エピソードを取得"""
        try:
            similarities, indices = self.memory_manager.search(embedding, top_k=self.topk)
            
            topk_episodes = []
            for idx, (similarity, episode_idx) in enumerate(zip(similarities, indices)):
                if episode_idx >= len(self.memory_manager.episodes):
                    continue
                    
                stored_episode = self.memory_manager.episodes[episode_idx]
                
                # ドメイン間分析
                current_domain = current_episode.get('domain', 'unknown')
                similar_domain = getattr(stored_episode, 'metadata', {}).get('domain', 'unknown')
                is_cross_domain = current_domain != similar_domain
                
                episode_info = {
                    'rank': idx + 1,
                    'similarity': float(similarity),
                    'episode_id': getattr(stored_episode, 'id', episode_idx),
                    'domain': similar_domain,
                    'research_area': getattr(stored_episode, 'metadata', {}).get('research_area', 'unknown'),
                    'is_cross_domain': is_cross_domain
                }
                
                topk_episodes.append(episode_info)
            
            return topk_episodes
            
        except Exception as e:
            print(f"⚠️ TopK取得エラー: {e}")
            return []
    
    def run_experiment(self, experiment_name: str, num_episodes: int, seed: int = 42) -> Dict[str, Any]:
        """実験実行"""
        print(f"🚀 実験開始: {experiment_name}")
        print(f"   エピソード数: {num_episodes}")
        print(f"   シード値: {seed}")
        print("=" * 60)
        
        start_time = time.time()
        
        # 乱数シード設定
        np.random.seed(seed)
        
        # 初期スナップショット
        self.take_memory_snapshot("experiment_start", 0)
        
        # エピソード生成
        episodes = self.generate_episodes(num_episodes, seed, "experiment")
        
        # 実験実行
        processing_times = []
        
        for i, episode in enumerate(episodes):
            episode_start = time.time()
            
            try:
                # エピソードをベクトル化
                embedding = self.model.encode(episode['text'])
                
                # TopK類似エピソードを取得
                topk_episodes = self.get_topk_similar_episodes(episode, embedding)
                
                # 洞察メトリクス計算
                delta_ged, delta_ig = self.calculate_ged_ig_metrics(embedding, i + 1)
                
                # 洞察条件チェック
                is_insight = self.check_insight_condition(delta_ged, delta_ig)
                
                # ドメイン分析
                cross_domain_count = sum(1 for ep in topk_episodes if ep['is_cross_domain'])
                domain_diversity = len(set(ep['domain'] for ep in topk_episodes))
                
                if is_insight:
                    insight_id = f"{experiment_name}_INS_{episode['id']:04d}_{int(time.time() * 1000) % 10000}"
                    
                    # 洞察タイプ分類
                    if delta_ged > 0.3:
                        insight_type = "Significant_Insight"
                    elif delta_ged > 0.2:
                        insight_type = "Notable_Pattern"  
                    else:
                        insight_type = "Micro_Insight"
                    
                    print(f"🔥 洞察検出: {insight_id} (Episode {episode['id']})")
                    print(f"   ΔGED: {delta_ged:.4f}, ΔIG: {delta_ig:.4f}, Type: {insight_type}")
                    
                    insight_data = {
                        'insight_id': insight_id,
                        'episode_id': episode['id'],
                        'episode_text': episode['text'],
                        'ged_value': delta_ged,
                        'ig_value': delta_ig,
                        'confidence': (delta_ged + delta_ig) / 2,
                        'insight_type': insight_type,
                        'cross_domain_count': cross_domain_count,
                        'domain_diversity': domain_diversity,
                        'current_domain': episode.get('domain', 'unknown'),
                        'current_research_area': episode.get('research_area', 'unknown'),
                        'detection_timestamp': datetime.now().isoformat(),
                        'experiment_name': experiment_name,
                        'seed': seed
                    }
                    
                    self.insight_logs.append(insight_data)
                
                # 詳細ログの記録
                experiment_log = {
                    'episode_id': episode['id'],
                    'episode_text': episode['text'],
                    'domain': episode.get('domain', 'unknown'),
                    'research_area': episode.get('research_area', 'unknown'),
                    'delta_ged': delta_ged,
                    'delta_ig': delta_ig,
                    'is_insight': is_insight,
                    'cross_domain_count': cross_domain_count,
                    'domain_diversity': domain_diversity,
                    'topk_count': len(topk_episodes),
                    'memory_size_before': len(self.memory_manager.episodes),
                    'timestamp': datetime.now().isoformat(),
                    'experiment_name': experiment_name,
                    'seed': seed
                }
                
                self.experiment_logs.append(experiment_log)
                
                # TopKログの記録
                if topk_episodes:
                    topk_log = {
                        'current_episode_id': episode['id'],
                        'current_domain': episode.get('domain', 'unknown'),
                        'current_research_area': episode.get('research_area', 'unknown'),
                        'topk_episodes': topk_episodes,
                        'cross_domain_count': cross_domain_count,
                        'experiment_name': experiment_name,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.topk_logs.append(topk_log)
                
                # メモリに保存
                self.memory_manager.store_episode(
                    text=episode['text'], 
                    c_value=0.2,
                    metadata={
                        'id': episode['id'], 
                        'domain': episode.get('domain', 'unknown'),
                        'research_area': episode.get('research_area', 'unknown'),
                        'experiment_name': experiment_name
                    }
                )
                
                # 処理時間記録
                episode_time = time.time() - episode_start
                processing_times.append(episode_time)
                
                # 定期スナップショット
                if (i + 1) % 50 == 0:
                    self.take_memory_snapshot(f"episode_{i+1}", i + 1)
                
                # 進捗表示
                if (i + 1) % 100 == 0:
                    avg_time = np.mean(processing_times[-100:])
                    eps_per_sec = 1.0 / avg_time if avg_time > 0 else 0
                    print(f"📈 進捗: {i+1}/{num_episodes} ({eps_per_sec:.1f} eps/sec, {len(self.insight_logs)} insights)")
                
            except Exception as e:
                print(f"⚠️ エピソード {episode['id']} 処理エラー: {e}")
                continue
        
        # 実験完了
        total_time = time.time() - start_time
        avg_eps_per_sec = num_episodes / total_time if total_time > 0 else 0
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        
        # 最終スナップショット
        self.take_memory_snapshot("experiment_end", num_episodes)
        
        # 結果サマリー
        results = {
            'experiment_name': experiment_name,
            'session_id': self.session_id,
            'total_episodes': num_episodes,
            'total_insights': len(self.insight_logs),
            'insight_rate': len(self.insight_logs) / num_episodes if num_episodes > 0 else 0,
            'total_time_seconds': total_time,
            'avg_episodes_per_second': avg_eps_per_sec,
            'avg_processing_time': avg_processing_time,
            'seed': seed,
            'config': {
                'memory_dim': self.memory_dim,
                'topk': self.topk,
                'ged_threshold': self.ged_threshold,
                'ig_threshold': self.ig_threshold,
                'similarity_threshold': self.similarity_threshold
            },
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n✅ 実験完了!")
        print(f"   総エピソード: {num_episodes}")
        print(f"   検出された洞察: {len(self.insight_logs)}")
        print(f"   実行時間: {total_time:.2f}秒")
        print(f"   処理速度: {avg_eps_per_sec:.2f} eps/sec")
        print(f"   洞察検出率: {len(self.insight_logs)/num_episodes*100:.2f}%")
        
        return results
    
    def save_results(self, experiment_name: str, results: Dict[str, Any]):
        """結果保存"""
        print("💾 実験結果を保存中...")
        
        # 実験固有の出力ディレクトリ
        exp_output_dir = self.session_dir / experiment_name
        exp_output_dir.mkdir(exist_ok=True)
        
        # 1. 入力エピソードCSV
        if self.experiment_logs:
            episodes_data = []
            for log in self.experiment_logs:
                episodes_data.append({
                    'episode_id': log['episode_id'],
                    'episode_text': log['episode_text'],
                    'domain': log['domain'],
                    'research_area': log['research_area'],
                    'experiment_name': log['experiment_name'],
                    'seed': log['seed'],
                    'timestamp': log['timestamp']
                })
            
            episodes_df = pd.DataFrame(episodes_data)
            episodes_df.to_csv(exp_output_dir / "01_input_episodes.csv", index=False)
        
        # 2. 洞察検出結果CSV
        if self.insight_logs:
            insights_df = pd.DataFrame(self.insight_logs)
            insights_df.to_csv(exp_output_dir / "02_insights.csv", index=False)
        
        # 3. 詳細実験ログCSV
        if self.experiment_logs:
            experiment_df = pd.DataFrame(self.experiment_logs)
            experiment_df.to_csv(exp_output_dir / "03_experiment_logs.csv", index=False)
        
        # 4. TopK分析CSV
        if self.topk_logs:
            topk_df = pd.DataFrame(self.topk_logs)
            topk_df.to_csv(exp_output_dir / "04_topk_analysis.csv", index=False)
        
        # 5. メモリスナップショットCSV
        if self.memory_snapshots:
            snapshots_df = pd.DataFrame(self.memory_snapshots)
            snapshots_df.to_csv(exp_output_dir / "05_memory_snapshots.csv", index=False)
        
        # 6. 実験結果サマリーJSON
        with open(exp_output_dir / "06_experiment_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 実験結果保存完了:")
        print(f"   📁 出力ディレクトリ: {exp_output_dir}")
        print(f"   📄 結果ファイル数: 6個")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="標準化実験実行スクリプト")
    parser.add_argument("session_id", help="実験セッションID")
    parser.add_argument("experiment_name", help="実験名")
    parser.add_argument("--episodes", type=int, default=500, help="エピソード数 (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="ランダムシード (default: 42)")
    parser.add_argument("--memory-dim", type=int, default=384, help="メモリ次元 (default: 384)")
    parser.add_argument("--topk", type=int, default=10, help="TopK近傍数 (default: 10)")
    parser.add_argument("--ged-threshold", type=float, default=0.15, help="GED閾値 (default: 0.15)")
    parser.add_argument("--ig-threshold", type=float, default=0.10, help="IG閾値 (default: 0.10)")
    parser.add_argument("--similarity-threshold", type=float, default=0.3, help="類似度閾値 (default: 0.3)")
    
    args = parser.parse_args()
    
    # 実験設定
    experiment_config = {
        "memory_dim": args.memory_dim,
        "topk": args.topk,
        "ged_threshold": args.ged_threshold,
        "ig_threshold": args.ig_threshold,
        "similarity_threshold": args.similarity_threshold
    }
    
    try:
        # 実験実行
        experiment = StandardizedExperiment(args.session_id, experiment_config)
        experiment.initialize_components()
        
        results = experiment.run_experiment(args.experiment_name, args.episodes, args.seed)
        experiment.save_results(args.experiment_name, results)
        
        print("\n🎉 標準化実験が正常に完了しました!")
        
    except Exception as e:
        print(f"❌ 実験実行エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
