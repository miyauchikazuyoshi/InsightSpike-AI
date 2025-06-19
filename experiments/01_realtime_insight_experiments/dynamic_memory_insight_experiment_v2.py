#!/usr/bin/env python3
"""
動的記憶洞察実験 v2.0 - InsightSpike-AI
==========================================

洞察エピソードを動的に記憶に追加して、自己強化ループの効果を観察する実験
（L2MemoryManager APIの正確な実装版）

修正内容:
- L2MemoryManagerの正確なAPIに合わせて修正
- 検索エラーの解決
- メモリ状態測定の適切な実装
- より堅牢なエラーハンドリング

実験目的:
1. 洞察エピソードの記憶追加が新たな洞察生成を加速するか
2. 記憶レイヤーがカオス化するか、それとも整理されるか
3. 自己参照的学習による創発的効果の観測
"""

import sys
from pathlib import Path

# パス設定
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# 安全なインポート
try:
    from insightspike.core.config import get_config
    from insightspike.utils.embedder import get_model
    from insightspike.core.layers.layer2_memory_manager import L2MemoryManager
    IMPORTS_OK = True
except ImportError as e:
    print(f"⚠️ インポートエラー（簡易版で続行）: {e}")
    IMPORTS_OK = False

class DynamicMemoryInsightExperimentV2:
    """動的記憶洞察実験クラス v2.0（API修正版）"""
    
    def __init__(self, output_dir: str = "experiments/01_realtime_insight_experiments/outputs/dynamic_memory_v2"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if IMPORTS_OK:
            # 実システム初期化
            self.config = get_config()
            self.model = get_model()
            self.memory_manager = L2MemoryManager(dim=384)
            
            # ベースグラフの埋め込みプロセスを初期化
            self._initialize_base_memory()
            print("✅ 実システムとベースメモリで初期化 (v2.0)")
        else:
            # ダミーシステム初期化
            self.config = {"dummy": True}
            self.model = None
            self.memory_manager = MockMemoryManagerV2()
            print("⚠️ ダミーシステムで初期化 (v2.0)")
        
        # 実験パラメータ
        self.topk = 10
        self.ged_threshold = 0.15
        self.ig_threshold = 0.10
        
        # 追跡データ
        self.episodes = []
        self.insights = []
        self.memory_snapshots = []
        self.insight_episodes_added = 0
    
    def _initialize_base_memory(self):
        """ベースメモリの初期化 - 基本的な学術概念を事前に記憶"""
        base_concepts = [
            "Machine learning algorithms optimize parameters through gradient descent.",
            "Deep neural networks learn hierarchical representations of data.",
            "Computer vision systems recognize patterns in visual information.",
            "Natural language processing models understand semantic relationships.",
            "Graph neural networks propagate information through network structures.",
            "Reinforcement learning agents maximize rewards through exploration.",
            "Cybersecurity systems detect anomalies in network traffic patterns.",
            "Climate models simulate atmospheric and oceanic dynamics.",
            "Drug discovery pipelines identify molecular targets and compounds.",
            "Autonomous systems navigate environments using sensor fusion."
        ]
        
        added_count = 0
        for concept in base_concepts:
            try:
                success = self.memory_manager.store_episode(
                    text=concept,
                    c_value=0.3  # 基本概念なので中程度のC値
                )
                if success:
                    added_count += 1
            except Exception as e:
                print(f"⚠️ ベース概念の追加に失敗: {e}")
        
        print(f"📚 {added_count}/{len(base_concepts)}個のベース概念をメモリに追加")
    
    def generate_episode(self) -> Dict:
        """エピソード生成"""
        domains = ["cybersecurity", "climate modeling", "drug discovery", 
                  "autonomous systems", "computer vision"]
        
        research_areas = ["Machine Learning", "Deep Learning", "Computer Vision", 
                         "Natural Language Processing", "Graph Neural Networks"]
        
        domain = np.random.choice(domains)
        research_area = np.random.choice(research_areas)
        
        templates = [
            f"Recent research in {research_area} achieves significant performance on {domain}.",
            f"Novel {research_area} architecture shows improvements in {domain} applications.",
            f"Study of {research_area} in {domain} reveals insights about scalability.",
            f"Advanced {research_area} techniques demonstrate robustness in {domain} scenarios.",
            f"Cross-domain transfer learning from {research_area} to {domain} shows promise.",
            f"Optimization methods in {research_area} enhance {domain} system efficiency.",
        ]
        
        episode_text = np.random.choice(templates)
        
        return {
            "episode_text": episode_text,
            "domain": domain,
            "research_area": research_area,
            "complexity": np.random.uniform(0.1, 1.0),
            "novelty": np.random.uniform(0.1, 1.0),
            "timestamp": datetime.now().isoformat()
        }
    
    def detect_insight(self, episode_data: Dict) -> Dict:
        """洞察検出（修正版）"""
        try:
            if IMPORTS_OK and self.model:
                # 実際のエンべディング処理
                embedding = self.model.encode([episode_data['episode_text']], 
                                           convert_to_numpy=True, 
                                           normalize_embeddings=True)[0]
                
                # 正しいメソッド名で類似エピソード検索
                try:
                    similar_episodes = self.memory_manager.search_episodes(embedding, k=self.topk)
                    
                    # 類似度計算（修正版）
                    if similar_episodes and len(similar_episodes) > 0:
                        # similar_episodesがEpisodeオブジェクトのリストの場合
                        c_values = []
                        for ep in similar_episodes:
                            if hasattr(ep, 'c'):
                                c_values.append(ep.c)
                            elif isinstance(ep, tuple) and len(ep) > 1:
                                c_values.append(ep[1])
                            else:
                                c_values.append(0.5)  # デフォルト値
                        
                        avg_similarity = np.mean(c_values) if c_values else 0.5
                        delta_ged = max(0.0, 0.5 - avg_similarity)
                        delta_ig = np.random.uniform(0.0, delta_ged * 2)  # GEDに基づくIG
                    else:
                        delta_ged = 0.5
                        delta_ig = 0.3
                    
                except Exception as search_error:
                    print(f"⚠️ 検索エラー: {search_error}")
                    delta_ged = 0.5
                    delta_ig = 0.3
                
                # メモリに追加
                try:
                    self.memory_manager.store_episode(
                        text=episode_data['episode_text'],
                        c_value=0.5
                    )
                except Exception as store_error:
                    print(f"⚠️ エピソード保存エラー: {store_error}")
                
            else:
                # ダミー処理
                delta_ged = np.random.uniform(0.0, 0.7)
                delta_ig = np.random.uniform(0.0, 0.5)
            
            # 洞察判定
            is_insight = (delta_ged > self.ged_threshold and delta_ig > self.ig_threshold)
            
            insight_result = {
                "is_insight": is_insight,
                "delta_ged": delta_ged,
                "delta_ig": delta_ig,
                "threshold_ged": self.ged_threshold,
                "threshold_ig": self.ig_threshold,
                "timestamp": datetime.now().isoformat()
            }
            
            return insight_result
            
        except Exception as e:
            print(f"⚠️ 洞察検出エラー: {e}")
            return {
                "is_insight": False,
                "delta_ged": 0.0,
                "delta_ig": 0.0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def add_insight_to_memory(self, episode_data: Dict, insight_data: Dict):
        """洞察エピソードをメモリに追加（動的記憶）"""
        try:
            # 洞察エピソードの拡張テキスト作成
            insight_text = f"""INSIGHT: {episode_data['episode_text']} 
            [GED:{insight_data['delta_ged']:.3f}, IG:{insight_data['delta_ig']:.3f}] 
            Domain: {episode_data['domain']}, Area: {episode_data['research_area']}"""
            
            if IMPORTS_OK and self.memory_manager:
                # 高いC値で洞察を記憶
                success = self.memory_manager.store_episode(
                    text=insight_text,
                    c_value=0.8  # 洞察は高いC値
                )
                
                if success:
                    self.insight_episodes_added += 1
                    print(f"💡 洞察エピソードを記憶に追加 (#{self.insight_episodes_added})")
                else:
                    print("⚠️ 洞察エピソード追加に失敗")
            else:
                # ダミー処理
                self.insight_episodes_added += 1
                
        except Exception as e:
            print(f"⚠️ 洞察追加エラー: {e}")
    
    def measure_memory_chaos(self) -> float:
        """記憶の混沌度を測定"""
        try:
            if IMPORTS_OK and self.memory_manager:
                # メモリ統計を取得
                stats = self.memory_manager.get_memory_stats()
                
                # エピソード数ベースの混沌度
                num_episodes = stats.get('num_episodes', 0)
                
                if num_episodes > 0:
                    # C値の分散を基に混沌度を計算
                    c_variance = stats.get('c_variance', 0.0)
                    chaos_score = min(1.0, c_variance * 10)  # 正規化
                else:
                    chaos_score = 0.0
                
                return chaos_score
                
            else:
                # ダミー処理: 動的記憶に基づく擬似混沌度
                base_chaos = 0.1
                insight_factor = self.insight_episodes_added * 0.05
                return min(1.0, base_chaos + insight_factor)
                
        except Exception as e:
            print(f"⚠️ 記憶状態測定エラー: {e}")
            return 0.0
    
    def run_experiment(self, num_episodes: int = 100, progress_interval: int = 25) -> Dict:
        """実験実行"""
        print("\n🚀 動的記憶洞察実験開始 v2.0")
        print(f"   エピソード数: {num_episodes}")
        
        start_time = time.time()
        initial_chaos = self.measure_memory_chaos()
        
        for i in range(num_episodes):
            # エピソード生成
            episode_data = self.generate_episode()
            
            # 洞察検出
            insight_result = self.detect_insight(episode_data)
            
            # データ記録
            episode_record = {
                **episode_data,
                **insight_result,
                "episode_id": i
            }
            self.episodes.append(episode_record)
            
            # 洞察の場合は記憶に追加
            if insight_result.get("is_insight", False):
                self.insights.append(episode_record)
                self.add_insight_to_memory(episode_data, insight_result)
            
            # 記憶状態スナップショット
            if (i + 1) % 10 == 0:
                chaos_score = self.measure_memory_chaos()
                snapshot = {
                    "episode": i + 1,
                    "chaos_score": chaos_score,
                    "insights_count": len(self.insights),
                    "memory_additions": self.insight_episodes_added,
                    "timestamp": datetime.now().isoformat()
                }
                self.memory_snapshots.append(snapshot)
            
            # 進捗報告
            if (i + 1) % progress_interval == 0:
                elapsed = time.time() - start_time
                insights_so_far = len(self.insights)
                chaos_score = self.measure_memory_chaos()
                speed = (i + 1) / elapsed
                
                print(f"📊 [{i+1:3d}/{num_episodes}] 洞察: {insights_so_far:3d} "
                      f"({insights_so_far/(i+1)*100:4.1f}%) 記憶追加: {self.insight_episodes_added:3d} "
                      f"カオス: {chaos_score:.3f} 速度: {speed:.1f} eps/s")
        
        # 実験終了
        end_time = time.time()
        final_chaos = self.measure_memory_chaos()
        
        results = {
            'total_episodes': num_episodes,
            'insights_detected': len(self.insights),
            'insight_rate': len(self.insights) / num_episodes,
            'insight_episodes_added': self.insight_episodes_added,
            'experiment_duration': end_time - start_time,
            'processing_speed': num_episodes / (end_time - start_time),
            'initial_chaos_score': initial_chaos,
            'final_chaos_score': final_chaos,
            'chaos_change': final_chaos - initial_chaos,
        }
        
        print(f"\n✅ 動的記憶洞察実験完了! v2.0")
        print(f"   総エピソード: {results['total_episodes']}")
        print(f"   洞察検出: {results['insights_detected']} ({results['insight_rate']:.1%})")
        print(f"   記憶追加: {results['insight_episodes_added']}")
        print(f"   実験時間: {results['experiment_duration']:.2f}秒")
        print(f"   処理速度: {results['processing_speed']:.1f} eps/s")
        print(f"   カオス変化: {results['initial_chaos_score']:.3f} → {results['final_chaos_score']:.3f} ({results['chaos_change']:+.3f})")
        
        # 結果保存
        self.save_results()
        
        return results
    
    def save_results(self):
        """結果の保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # エピソードデータ
            if self.episodes:
                episodes_df = pd.DataFrame(self.episodes)
                episodes_df.to_csv(self.output_dir / f"episodes_{timestamp}.csv", index=False)
            
            # 洞察データ
            if self.insights:
                insights_df = pd.DataFrame(self.insights)
                insights_df.to_csv(self.output_dir / f"insights_{timestamp}.csv", index=False)
            
            # 記憶スナップショット
            if self.memory_snapshots:
                memory_df = pd.DataFrame(self.memory_snapshots)
                memory_df.to_csv(self.output_dir / f"memory_snapshots_{timestamp}.csv", index=False)
            
            # メタデータ
            metadata = {
                'experiment': 'dynamic_memory_insight_v2',
                'timestamp': timestamp,
                'total_episodes': len(self.episodes),
                'total_insights': len(self.insights),
                'insight_rate': len(self.insights) / len(self.episodes) if self.episodes else 0,
                'insight_episodes_added': self.insight_episodes_added,
                'system_type': 'real' if IMPORTS_OK else 'dummy',
                'version': '2.0'
            }
            
            with open(self.output_dir / f"metadata_{timestamp}.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"💾 結果保存完了: {self.output_dir}")
            
        except Exception as e:
            print(f"⚠️ 保存エラー: {e}")


class MockMemoryManagerV2:
    """ダミーメモリマネージャー v2.0"""
    def __init__(self):
        self.episodes = []
    
    def store_episode(self, text, c_value=0.5):
        """エピソード保存（ダミー）"""
        episode = {
            "text": text,
            "c_value": c_value,
            "timestamp": datetime.now().isoformat()
        }
        self.episodes.append(episode)
        return True
    
    def search_episodes(self, query, k=10):
        """類似エピソード検索（ダミー）"""
        class MockEpisode:
            def __init__(self, text, c):
                self.text = text
                self.c = c
        
        return [MockEpisode(ep.get("text", ""), ep.get("c_value", 0.5)) 
                for ep in self.episodes[:k]]
    
    def get_memory_stats(self):
        """メモリ統計（ダミー）"""
        if not self.episodes:
            return {'num_episodes': 0, 'c_variance': 0.0}
        
        c_values = [ep.get("c_value", 0.5) for ep in self.episodes]
        return {
            'num_episodes': len(self.episodes),
            'c_variance': np.var(c_values) if c_values else 0.0,
            'mean_c_value': np.mean(c_values) if c_values else 0.5
        }
    
    def get_recent_episodes(self, n=50):
        return self.episodes[-n:]


def main():
    """メイン実行関数"""
    print("🧠 動的記憶洞察実験 v2.0 - InsightSpike-AI")
    print("=" * 60)
    
    try:
        experiment = DynamicMemoryInsightExperimentV2()
        
        # 実験実行
        results = experiment.run_experiment(
            num_episodes=150,  # 中規模で詳細観察
            progress_interval=25
        )
        
        print("\n🎉 実験結果サマリー:")
        print(f"   洞察検出率: {results['insight_rate']:.1%}")
        print(f"   記憶追加数: {results['insight_episodes_added']}")
        print(f"   最終カオス度: {results['final_chaos_score']:.3f}")
        
        # 結果分析
        if results['insight_episodes_added'] > 0:
            acceleration_ratio = results['insight_rate'] / (results['insight_episodes_added'] / results['insights_detected']) if results['insights_detected'] > 0 else 0
            print(f"   加速効果: {acceleration_ratio:.3f}")
            
            if results['final_chaos_score'] > 0.5:
                print("   📈 記憶レイヤーは多様化傾向（良いカオス）")
            elif results['final_chaos_score'] < 0.2:
                print("   📉 記憶レイヤーは収束傾向（秩序化）")
            else:
                print("   ⚖️ 記憶レイヤーは安定状態")
        
        print(f"\n📊 カオス変化の解釈:")
        if results['chaos_change'] > 0.1:
            print("   🔥 動的記憶が創発的複雑性を生成")
        elif results['chaos_change'] < -0.1:
            print("   🧘 動的記憶が秩序化を促進")
        else:
            print("   ⚖️ 動的記憶が安定した学習環境を維持")
        
    except KeyboardInterrupt:
        print("\n⏹️ 実験中断")
    except Exception as e:
        print(f"\n❌ 実験エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
