#!/usr/bin/env python3
"""
動的記憶洞察実験 - InsightSpike-AI (簡略版)
============================================

洞察エピソードを動的に記憶に追加して、自己強化ループの効果を観察する実験

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

class DynamicMemoryInsightExperiment:
    """動的記憶洞察実験クラス（簡略版）"""
    
    def __init__(self, output_dir: str = "experiments/01_realtime_insight_experiments/outputs/dynamic_memory"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if IMPORTS_OK:
            # 実システム初期化
            self.config = get_config()
            self.model = get_model()
            self.memory_manager = L2MemoryManager(dim=384)
            
            # ベースグラフの埋め込みプロセスを初期化
            self._initialize_base_memory()
            print("✅ 実システムとベースメモリで初期化")
        else:
            # ダミーシステム初期化
            self.config = {"dummy": True}
            self.model = None
            self.memory_manager = MockMemoryManager()
            print("⚠️ ダミーシステムで初期化")
        
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
        
        for concept in base_concepts:
            try:
                self.memory_manager.store_episode(
                    text=concept,
                    c_value=0.3,  # 基本概念なので中程度のC値
                    metadata={"type": "base_concept", "domain": "general"}
                )
            except Exception as e:
                print(f"⚠️ ベース概念の追加に失敗: {e}")
        
        print(f"📚 {len(base_concepts)}個のベース概念をメモリに追加")
        
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
        ]
        
        episode_text = np.random.choice(templates)
        
        return {
            "episode_text": episode_text,
            "domain": domain,
            "research_area": research_area,
            "complexity": np.random.uniform(0.1, 1.0),
            "novelty": np.random.uniform(0.1, 1.0)
        }
    
    def detect_insight(self, episode_data: Dict) -> Dict:
        """洞察検出（簡易版）"""
        try:
            if IMPORTS_OK and self.model:
                # 実際のエンべディング処理
                embedding = self.model.encode([episode_data['episode_text']], 
                                           convert_to_numpy=True, 
                                           normalize_embeddings=True)[0]
                
                # 正しいメソッド名で類似エピソード検索
                similar_episodes = self.memory_manager.search_episodes(embedding, k=self.topk)
                
                # 簡易メトリクス計算
                if similar_episodes:
                    avg_similarity = np.mean([episode.c for episode in similar_episodes])
                    delta_ged = max(0.0, 0.5 - avg_similarity)
                    delta_ig = np.random.uniform(0.0, 0.5)  # 簡易IG
                else:
                    delta_ged = 0.5
                    delta_ig = 0.3
                
                # メモリに追加（正しいメソッド名）
                self.memory_manager.store_episode(
                    text=episode_data['episode_text'],
                    c_value=0.5,
                    metadata=episode_data
                )
            else:
                # ダミー処理
                delta_ged = np.random.uniform(0.0, 0.6)
                delta_ig = np.random.uniform(0.0, 0.5)
                self.memory_manager.add_episode(episode_data)
            
            # 洞察判定
            has_insight = (delta_ged >= self.ged_threshold and delta_ig >= self.ig_threshold)
            confidence = min(delta_ged + delta_ig, 1.0)
            
            return {
                'has_insight': has_insight,
                'ged': delta_ged,
                'ig': delta_ig,
                'confidence': confidence,
                'explanation': f"GED: {delta_ged:.3f}, IG: {delta_ig:.3f}"
            }
            
        except Exception as e:
            print(f"⚠️ 洞察検出エラー: {e}")
            return {'has_insight': False, 'ged': 0.0, 'ig': 0.0, 'confidence': 0.0}
    
    def add_insight_to_memory(self, episode_data: Dict, insight_data: Dict):
        """洞察エピソードを動的に記憶に追加"""
        try:
            # 洞察マーク付きエピソード
            insight_episode = {
                **episode_data,
                'is_insight': True,
                'insight_timestamp': datetime.now().isoformat(),
                'ged_value': insight_data.get('ged', 0),
                'ig_value': insight_data.get('ig', 0),
                'confidence': insight_data.get('confidence', 0)
            }
            
            if IMPORTS_OK and self.model:
                # 実際の処理
                insight_text = f"[INSIGHT] {episode_data['episode_text']}"
                embedding = self.model.encode([insight_text])[0]
                self.memory_manager.add_episode(
                    text=insight_text,
                    vector=embedding,
                    metadata=insight_episode
                )
            else:
                # ダミー処理
                self.memory_manager.add_episode(insight_episode)
            
            self.insight_episodes_added += 1
            print(f"💡 洞察エピソード記憶追加 (総数: {self.insight_episodes_added})")
            
        except Exception as e:
            print(f"⚠️ 洞察追加エラー: {e}")
    
    def measure_memory_state(self) -> Dict:
        """記憶状態の測定"""
        try:
            memory_size = self.memory_manager.get_memory_size()
            recent_episodes = self.memory_manager.get_recent_episodes(50)
            
            # 多様性計算
            domains = set()
            research_areas = set()
            insight_count = 0
            
            for ep in recent_episodes:
                metadata = getattr(ep, 'metadata', ep) if hasattr(ep, 'metadata') else ep
                domains.add(metadata.get('domain', 'unknown'))
                research_areas.add(metadata.get('research_area', 'unknown'))
                if metadata.get('is_insight', False):
                    insight_count += 1
            
            domain_diversity = len(domains) / max(len(recent_episodes), 1)
            research_diversity = len(research_areas) / max(len(recent_episodes), 1)
            insight_density = insight_count / max(len(recent_episodes), 1)
            chaos_score = domain_diversity * research_diversity * (1 + insight_density)
            
            return {
                'memory_size': memory_size,
                'domain_diversity': domain_diversity,
                'research_diversity': research_diversity,
                'insight_density': insight_density,
                'chaos_score': chaos_score
            }
            
        except Exception as e:
            print(f"⚠️ 記憶状態測定エラー: {e}")
            return {'memory_size': 0, 'chaos_score': 0}
    
    def run_experiment(self, num_episodes: int = 200, progress_interval: int = 25):
        """動的記憶洞察実験実行"""
        print(f"\n🚀 動的記憶洞察実験開始")
        print(f"   エピソード数: {num_episodes}")
        
        start_time = time.time()
        insights_detected = 0
        
        for i in range(num_episodes):
            try:
                # エピソード生成・処理
                episode_data = self.generate_episode()
                episode_data['episode_id'] = i
                episode_data['timestamp'] = datetime.now().isoformat()
                
                self.episodes.append(episode_data)
                
                # 洞察検出
                result = self.detect_insight(episode_data)
                
                if result['has_insight']:
                    insights_detected += 1
                    insight_data = {
                        'episode_id': i,
                        'timestamp': datetime.now().isoformat(),
                        **result
                    }
                    self.insights.append(insight_data)
                    
                    # 洞察エピソードを記憶に追加（実験のメイン機能）
                    self.add_insight_to_memory(episode_data, insight_data)
                    
                    # 記憶状態スナップショット
                    memory_state = self.measure_memory_state()
                    memory_state['episode_id'] = i
                    memory_state['insight_count'] = insights_detected
                    self.memory_snapshots.append(memory_state)
                
                # 進捗表示
                if (i + 1) % progress_interval == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    latest_chaos = self.memory_snapshots[-1]['chaos_score'] if self.memory_snapshots else 0
                    
                    print(f"📊 [{i+1:3d}/{num_episodes}] "
                          f"洞察: {insights_detected:3d} ({insights_detected/(i+1)*100:.1f}%) "
                          f"記憶追加: {self.insight_episodes_added:3d} "
                          f"カオス: {latest_chaos:.3f} "
                          f"速度: {rate:.1f} eps/s")
                    
            except Exception as e:
                print(f"⚠️ エピソード {i} エラー: {e}")
                continue
        
        # 実験完了
        total_time = time.time() - start_time
        final_rate = num_episodes / total_time
        
        print(f"\n✅ 動的記憶洞察実験完了!")
        print(f"   総エピソード: {num_episodes}")
        print(f"   洞察検出: {insights_detected} ({insights_detected/num_episodes*100:.1f}%)")
        print(f"   記憶追加: {self.insight_episodes_added}")
        print(f"   実験時間: {total_time:.2f}秒")
        print(f"   処理速度: {final_rate:.1f} eps/s")
        
        if self.memory_snapshots:
            initial_chaos = self.memory_snapshots[0]['chaos_score'] if len(self.memory_snapshots) > 1 else 0
            final_chaos = self.memory_snapshots[-1]['chaos_score']
            print(f"   カオス変化: {initial_chaos:.3f} → {final_chaos:.3f} ({final_chaos-initial_chaos:+.3f})")
        
        # 結果保存
        self.save_results()
        
        return {
            'insights_detected': insights_detected,
            'insight_rate': insights_detected / num_episodes,
            'insight_episodes_added': self.insight_episodes_added,
            'total_time': total_time,
            'final_chaos_score': self.memory_snapshots[-1]['chaos_score'] if self.memory_snapshots else 0
        }
    
    def save_results(self):
        """実験結果保存"""
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
                'experiment': 'dynamic_memory_insight',
                'timestamp': timestamp,
                'total_episodes': len(self.episodes),
                'total_insights': len(self.insights),
                'insight_rate': len(self.insights) / len(self.episodes) if self.episodes else 0,
                'insight_episodes_added': self.insight_episodes_added,
                'system_type': 'real' if IMPORTS_OK else 'dummy'
            }
            
            with open(self.output_dir / f"metadata_{timestamp}.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"💾 結果保存完了: {self.output_dir}")
            
        except Exception as e:
            print(f"⚠️ 保存エラー: {e}")


class MockMemoryManager:
    """ダミーメモリマネージャー"""
    def __init__(self):
        self.episodes = []
    
    def store_episode(self, text, c_value=0.5, metadata=None):
        """エピソード保存（ダミー）"""
        episode = {
            "text": text,
            "c_value": c_value,
            "metadata": metadata or {}
        }
        self.episodes.append(episode)
    
    def search_episodes(self, query, k=10):
        """類似エピソード検索（ダミー）"""
        class MockEpisode:
            def __init__(self, text, c):
                self.text = text
                self.c = c
        
        return [MockEpisode(ep.get("text", ""), ep.get("c_value", 0.5)) 
                for ep in self.episodes[:k]]
    
    def get_memory_size(self):
        return len(self.episodes)
    
    def get_recent_episodes(self, n=50):
        return self.episodes[-n:]


def main():
    """メイン実行関数"""
    print("🧠 動的記憶洞察実験 - InsightSpike-AI")
    print("=" * 50)
    
    try:
        experiment = DynamicMemoryInsightExperiment()
        
        # 実験実行
        results = experiment.run_experiment(
            num_episodes=200,  # 短めで様子見
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
        
    except KeyboardInterrupt:
        print("\n⚠️ 実験中断")
    except Exception as e:
        print(f"\n❌ 実験エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
