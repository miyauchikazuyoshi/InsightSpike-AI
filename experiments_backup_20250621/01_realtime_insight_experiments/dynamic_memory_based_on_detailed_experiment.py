#!/usr/bin/env python3
"""
動的記憶洞察実験 - 詳細ログ版プロトコルベース
===========================================

元の詳細ログ版実践的リアルタイム洞察実験をベースに、
洞察エピソードを動的に記憶に追加する実験。

実際のdataフォルダに書き込みながら、洞察の記憶追加が
次のエピソードの検出にどう影響するかを観察する。

修正点:
- 元のDetailedLoggingRealtimeExperimentクラスを継承
- 洞察検出時に強化されたエピソードを記憶に追加
- 実際のdataディレクトリを使用（backup/restore付き）
- 動的記憶の効果を詳細に記録
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
import pandas as pd
import shutil

# 元の詳細ログ版実験をインポート
sys.path.append(str(Path(__file__).parent))
from detailed_logging_realtime_experiment import DetailedLoggingRealtimeExperiment

class DynamicMemoryInsightExperiment(DetailedLoggingRealtimeExperiment):
    """動的記憶洞察実験クラス（詳細ログ版ベース）"""
    
    def __init__(self):
        # 親クラスの初期化
        super().__init__()
        
        print("🧠 動的記憶実験モードに拡張中...")
        
        # 動的記憶実験用パラメータ
        self.insight_memory_boost = 0.8  # 洞察エピソードのC値
        self.normal_memory_c = 0.2       # 通常エピソードのC値
        
        # 動的記憶追跡用データ
        self.dynamic_memory_logs = []
        self.insight_memory_additions = 0
        self.memory_state_snapshots = []
        
        # 初期グラフ構築プロトコル
        self._build_initial_knowledge_graph()
        
        print("✅ 動的記憶実験システム初期化完了")
        print(f"   洞察記憶強化値: {self.insight_memory_boost}")
        print(f"   通常記憶値: {self.normal_memory_c}")
    
    def _build_initial_knowledge_graph(self):
        """初期ナレッジグラフとメモリベースの構築"""
        print("🏗️ 初期グラフ構築プロトコル開始...")
        
        # 基盤概念エピソード群
        foundational_episodes = [
            {
                "text": "Machine learning algorithms learn patterns from data through iterative optimization processes.",
                "domain": "Machine Learning",
                "research_area": "Algorithm Design",
                "complexity": 0.6,
                "novelty": 0.4
            },
            {
                "text": "Deep neural networks use hierarchical feature learning to extract representations at multiple levels.",
                "domain": "Machine Learning", 
                "research_area": "Deep Learning",
                "complexity": 0.8,
                "novelty": 0.6
            },
            {
                "text": "Computer vision systems process visual information through convolutional and attention mechanisms.",
                "domain": "Computer Vision",
                "research_area": "Visual Processing",
                "complexity": 0.7,
                "novelty": 0.5
            },
            {
                "text": "Natural language processing leverages transformer architectures for semantic understanding.",
                "domain": "NLP",
                "research_area": "Language Models", 
                "complexity": 0.9,
                "novelty": 0.7
            },
            {
                "text": "Graph neural networks propagate and aggregate information across network structures.",
                "domain": "Graph Learning",
                "research_area": "Network Analysis",
                "complexity": 0.8,
                "novelty": 0.8
            },
            {
                "text": "Reinforcement learning agents optimize policies through reward-based exploration and exploitation.",
                "domain": "Reinforcement Learning",
                "research_area": "Decision Making",
                "complexity": 0.9,
                "novelty": 0.6
            },
            {
                "text": "Cybersecurity systems detect anomalies and threats through behavioral pattern analysis.",
                "domain": "Cybersecurity",
                "research_area": "Threat Detection",
                "complexity": 0.7,
                "novelty": 0.5
            },
            {
                "text": "Climate modeling simulates atmospheric dynamics using numerical methods and data assimilation.",
                "domain": "Climate Science",
                "research_area": "Environmental Modeling",
                "complexity": 0.8,
                "novelty": 0.6
            }
        ]
        
        # 基盤エピソードをメモリとグラフに追加
        initial_count = 0
        for episode_data in foundational_episodes:
            try:
                # L2メモリに保存（通常のC値）
                memory_success = self.memory_manager.store_episode(
                    text=episode_data["text"],
                    c_value=self.normal_memory_c
                )
                
                # ナレッジグラフに追加
                if hasattr(self.knowledge_graph, 'add_episode'):
                    graph_success = self.knowledge_graph.add_episode(
                        text=episode_data["text"],
                        metadata=episode_data
                    )
                else:
                    graph_success = True  # フォールバック
                
                if memory_success:
                    initial_count += 1
                    
            except Exception as e:
                print(f"⚠️ 初期エピソード追加エラー: {e}")
        
        print(f"📚 初期グラフ構築完了: {initial_count}/{len(foundational_episodes)} エピソード追加")
        
        # 初期メモリ状態のスナップショット
        initial_snapshot = {
            "timestamp": datetime.now().isoformat(),
            "phase": "initial_graph_construction", 
            "episodes_added": initial_count,
            "memory_type": "foundational_knowledge",
            "avg_c_value": self.normal_memory_c
        }
        self.memory_state_snapshots.append(initial_snapshot)
    
    def create_insight_enhanced_episode(self, original_episode: Dict, insight_data: Dict) -> str:
        """洞察情報で強化されたエピソードテキストを作成"""
        
        # 洞察タイプに基づくテキスト強化
        insight_type = insight_data.get('insight_type', 'Micro_Insight')
        ged_value = insight_data.get('ged_value', 0.0)
        ig_value = insight_data.get('ig_value', 0.0)
        cross_domain_count = insight_data.get('cross_domain_count', 0)
        
        # 元のテキストに洞察メタデータを追加
        enhanced_text = f"""INSIGHT_EPISODE: {original_episode['text']}
        
INSIGHT_METADATA:
- Type: {insight_type}
- GED: {ged_value:.4f} (structural novelty)
- IG: {ig_value:.4f} (information gain)
- Cross-domain connections: {cross_domain_count}
- Domain: {original_episode.get('domain', 'unknown')}
- Research Area: {original_episode.get('research_area', 'unknown')}
- Detection Time: {insight_data.get('detection_timestamp', 'unknown')}

This episode represents a significant insight that demonstrates novel patterns 
and cross-domain knowledge integration capabilities."""
        
        return enhanced_text
    
    def add_insight_to_memory(self, episode: Dict, insight_data: Dict):
        """洞察エピソードを強化してメモリに追加"""
        try:
            # 強化されたエピソードテキストを作成
            enhanced_text = self.create_insight_enhanced_episode(episode, insight_data)
            
            # 高いC値で記憶に保存（実際のdataフォルダに書き込み）
            success = self.memory_manager.store_episode(
                text=enhanced_text,
                c_value=self.insight_memory_boost,
                metadata={
                    'type': 'insight_episode',
                    'original_id': episode['id'],
                    'insight_id': insight_data['insight_id'],
                    'insight_type': insight_data.get('insight_type', 'Unknown'),
                    'ged_value': insight_data.get('ged_value', 0.0),
                    'ig_value': insight_data.get('ig_value', 0.0),
                    'domain': episode.get('domain', 'unknown'),
                    'research_area': episode.get('research_area', 'unknown'),
                    'enhancement_timestamp': datetime.now().isoformat()
                }
            )
            
            if success:
                self.insight_memory_additions += 1
                print(f"💡 洞察記憶追加 #{self.insight_memory_additions}: {insight_data['insight_id']}")
                print(f"   C値: {self.insight_memory_boost}, タイプ: {insight_data.get('insight_type', 'Unknown')}")
                
                # 動的記憶ログに記録
                memory_log = {
                    'addition_number': self.insight_memory_additions,
                    'insight_id': insight_data['insight_id'],
                    'original_episode_id': episode['id'],
                    'enhanced_text_length': len(enhanced_text),
                    'c_value': self.insight_memory_boost,
                    'memory_size_before': len(self.memory_manager.episodes) - 1,
                    'memory_size_after': len(self.memory_manager.episodes),
                    'insight_type': insight_data.get('insight_type', 'Unknown'),
                    'ged_value': insight_data.get('ged_value', 0.0),
                    'ig_value': insight_data.get('ig_value', 0.0),
                    'timestamp': datetime.now().isoformat()
                }
                
                self.dynamic_memory_logs.append(memory_log)
                
            else:
                print(f"⚠️ 洞察記憶追加失敗: {insight_data['insight_id']}")
                
        except Exception as e:
            print(f"❌ 洞察記憶追加エラー: {e}")
    
    def capture_memory_state_snapshot(self, episode_number: int):
        """記憶状態のスナップショットを取得"""
        try:
            # メモリ統計を取得（安全性チェック付き）
            if not hasattr(self.memory_manager, 'episodes'):
                print(f"⚠️ メモリマネージャーにepisodesアトリビュートがありません")
                return None
            
            total_episodes = len(self.memory_manager.episodes)
            
            # C値の分布を分析
            c_values = []
            insight_episodes = []
            domains = set()
            
            for ep in self.memory_manager.episodes:
                # C値の取得
                if hasattr(ep, 'c'):
                    c_values.append(ep.c)
                
                # 洞察エピソードの判定
                metadata = getattr(ep, 'metadata', {}) or {}
                if metadata.get('type') == 'insight_episode':
                    insight_episodes.append(ep)
                
                # ドメインの取得
                domain = metadata.get('domain', 'unknown')
                domains.add(domain)
            
            c_mean = np.mean(c_values) if c_values else 0.0
            c_std = np.std(c_values) if c_values else 0.0
            insight_ratio = len(insight_episodes) / total_episodes if total_episodes > 0 else 0.0
            domain_diversity = len(domains)
            
            snapshot = {
                'episode_number': episode_number,
                'total_memory_size': total_episodes,
                'insight_episodes_count': len(insight_episodes),
                'insight_ratio': insight_ratio,
                'c_value_mean': c_mean,
                'c_value_std': c_std,
                'domain_diversity': domain_diversity,
                'dynamic_additions': self.insight_memory_additions,
                'timestamp': datetime.now().isoformat()
            }
            
            self.memory_state_snapshots.append(snapshot)
            
            return snapshot
            
        except Exception as e:
            print(f"⚠️ メモリ状態取得エラー: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_dynamic_memory_experiment(self, num_episodes: int = 500):
        """動的記憶実験の実行"""
        
        print(f"🚀 動的記憶洞察実験開始 ({num_episodes}エピソード)")
        print("=" * 70)
        print("実験内容: 洞察検出→記憶強化→次回検出への影響を観察")
        print()
        
        start_time = time.time()
        
        # エピソード生成
        episodes = self.generate_episodes(num_episodes)
        
        # 実験データ
        insights_detected = []
        processing_times = []
        
        # 初期記憶状態を記録
        initial_snapshot = self.capture_memory_state_snapshot(0)
        print(f"📊 初期記憶状態: サイズ={initial_snapshot['total_memory_size']}, 多様性={initial_snapshot['domain_diversity']}")
        
        print(f"🔄 動的記憶付きリアルタイム洞察検出開始 (TopK={self.topk})...")
        
        for i, episode in enumerate(episodes):
            episode_start = time.time()
            
            try:
                # エピソードをベクトル化
                embedding = self.model.encode(episode['text'])
                
                # TopK類似エピソードを取得（実際のメモリから）
                topk_episodes = self.get_topk_similar_episodes(episode, embedding)
                
                # 洞察メトリクス計算
                delta_ged, delta_ig = self.calculate_ged_ig_metrics(embedding, i + 1)
                
                # 洞察条件チェック
                is_insight = self.check_insight_condition(delta_ged, delta_ig)
                
                # 通常エピソードをメモリに保存（低いC値）
                self.memory_manager.store_episode(
                    text=episode['text'], 
                    c_value=self.normal_memory_c,
                    metadata={'id': episode['id'], 'domain': episode.get('domain', 'unknown')}
                )
                
                if is_insight:
                    insight_id = f"DYN_INS_{episode['id']:04d}_{int(time.time() * 1000) % 10000}"
                    
                    # 洞察タイプ分類
                    if delta_ged > 0.3:
                        insight_type = "Significant_Insight"
                    elif delta_ged > 0.2:
                        insight_type = "Notable_Pattern"  
                    else:
                        insight_type = "Micro_Insight"
                    
                    # ドメイン分析
                    cross_domain_count = sum(1 for ep in topk_episodes if ep.get('is_cross_domain', False))
                    domain_diversity = len(set(ep.get('domain', 'unknown') for ep in topk_episodes))
                    
                    print(f"🔥 動的記憶洞察検出: {insight_id} (Episode {episode['id']})")
                    print(f"   ΔGED: {delta_ged:.4f}, ΔIG: {delta_ig:.4f}, Type: {insight_type}")
                    print(f"   現在のメモリサイズ: {len(self.memory_manager.episodes)}")
                    
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
                        'memory_size_at_detection': len(self.memory_manager.episodes),
                        'dynamic_additions_so_far': self.insight_memory_additions,
                        'detection_timestamp': datetime.now().isoformat()
                    }
                    
                    insights_detected.append(insight_data)
                    
                    # 🎯 洞察エピソードを強化してメモリに追加（動的記憶の核心）
                    self.add_insight_to_memory(episode, insight_data)
                
                # 詳細ログの記録
                detailed_log = {
                    'episode_id': episode['id'],
                    'episode_text': episode['text'],
                    'domain': episode.get('domain', 'unknown'),
                    'research_area': episode.get('research_area', 'unknown'),
                    'delta_ged': delta_ged,
                    'delta_ig': delta_ig,
                    'is_insight': is_insight,
                    'memory_size': len(self.memory_manager.episodes),
                    'dynamic_additions': self.insight_memory_additions,
                    'topk_count': len(topk_episodes),
                    'timestamp': datetime.now().isoformat()
                }
                
                self.detailed_logs.append(detailed_log)
                
                # 処理時間記録
                episode_time = time.time() - episode_start
                processing_times.append(episode_time)
                
                # 記憶状態スナップショット（50エピソードごと）
                if (i + 1) % 50 == 0:
                    snapshot = self.capture_memory_state_snapshot(i + 1)
                    if snapshot:
                        print(f"📊 記憶状態 (Episode {i+1}): サイズ={snapshot['total_memory_size']}, "
                              f"洞察率={snapshot['insight_ratio']:.1%}, 動的追加={snapshot['dynamic_additions']}")
                
                # 進捗表示
                if (i + 1) % 100 == 0:
                    avg_time = np.mean(processing_times[-100:])
                    eps_per_sec = 1.0 / avg_time if avg_time > 0 else 0
                    memory_size = len(self.memory_manager.episodes)
                    print(f"📈 進捗: {i+1}/{num_episodes} ({eps_per_sec:.1f} eps/sec, "
                          f"{len(insights_detected)} insights, {self.insight_memory_additions} dynamic adds, "
                          f"memory: {memory_size})")
                
                # デバッグログ（最初の10エピソードのみ）
                if i < 10:
                    print(f"📊 Episode {i+1}: GED={delta_ged:.3f}, IG={delta_ig:.3f}, "
                          f"Insight={is_insight}, Memory={len(self.memory_manager.episodes)}")
                
            except Exception as e:
                print(f"⚠️ エピソード {episode['id']} 処理エラー: {e}")
                continue
        
        # 実験完了
        total_time = time.time() - start_time
        avg_eps_per_sec = num_episodes / total_time if total_time > 0 else 0
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        final_memory_size = len(self.memory_manager.episodes)
        
        print(f"\n✅ 動的記憶洞察実験完了!")
        print(f"   総エピソード: {num_episodes}")
        print(f"   検出された洞察: {len(insights_detected)}")
        print(f"   動的記憶追加: {self.insight_memory_additions}")
        print(f"   最終メモリサイズ: {final_memory_size}")
        print(f"   実行時間: {total_time:.2f}秒")
        print(f"   処理速度: {avg_eps_per_sec:.2f} eps/sec")
        print(f"   洞察検出率: {len(insights_detected)/num_episodes*100:.2f}%")
        print(f"   動的記憶効果: {self.insight_memory_additions/len(insights_detected)*100:.1f}%の洞察が記憶強化")
        
        # 記憶状態の変化を分析
        if len(self.memory_state_snapshots) >= 2:
            initial = self.memory_state_snapshots[0]
            final = self.memory_state_snapshots[-1]
            
            print(f"\n📊 記憶状態の変化:")
            
            # 安全にキーアクセス
            initial_size = initial.get('total_memory_size', 0)
            final_size = final.get('total_memory_size', 0)
            initial_ratio = initial.get('insight_ratio', 0.0)
            final_ratio = final.get('insight_ratio', 0.0)
            initial_diversity = initial.get('domain_diversity', 0)
            final_diversity = final.get('domain_diversity', 0)
            initial_c_mean = initial.get('c_value_mean', 0.0)
            final_c_mean = final.get('c_value_mean', 0.0)
            
            print(f"   メモリサイズ: {initial_size} → {final_size} "
                  f"({final_size - initial_size:+d})")
            print(f"   洞察エピソード率: {initial_ratio:.1%} → {final_ratio:.1%}")
            print(f"   ドメイン多様性: {initial_diversity} → {final_diversity}")
            print(f"   C値平均: {initial_c_mean:.3f} → {final_c_mean:.3f}")
        elif len(self.memory_state_snapshots) == 1:
            print(f"\n📊 記憶状態: 最終スナップショットのみ利用可能")
            final = self.memory_state_snapshots[0]
            print(f"   最終メモリサイズ: {final.get('total_memory_size', 0)}")
            print(f"   洞察エピソード率: {final.get('insight_ratio', 0.0):.1%}")
        else:
            print(f"\n⚠️ 記憶状態スナップショットが不足（{len(self.memory_state_snapshots)}個）")
        
        # 結果保存
        self.save_dynamic_memory_results(episodes, insights_detected, total_time, avg_eps_per_sec, avg_processing_time, final_memory_size)
        
        return {
            'episodes': episodes,
            'insights': insights_detected,
            'dynamic_additions': self.insight_memory_additions,
            'final_memory_size': final_memory_size,
            'total_time': total_time,
            'avg_eps_per_sec': avg_eps_per_sec,
            'insight_rate': len(insights_detected)/num_episodes
        }
    
    def save_dynamic_memory_results(self, episodes, insights, total_time, avg_eps_per_sec, avg_processing_time, final_memory_size):
        """動的記憶実験結果の保存"""
        
        print("💾 動的記憶実験結果を保存中...")
        
        # 出力ディレクトリ作成
        output_dir = Path("experiments/01_realtime_insight_experiments/outputs/dynamic_memory_detailed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 入力エピソードCSV
        episodes_df = pd.DataFrame(episodes)
        episodes_df.to_csv(output_dir / "01_input_episodes.csv", index=False)
        
        # 2. 洞察検出結果CSV
        if insights:
            insights_df = pd.DataFrame(insights)
            insights_df.to_csv(output_dir / "02_dynamic_insights.csv", index=False)
        
        # 3. 動的記憶ログCSV
        if self.dynamic_memory_logs:
            dynamic_df = pd.DataFrame(self.dynamic_memory_logs)
            dynamic_df.to_csv(output_dir / "03_dynamic_memory_logs.csv", index=False)
        
        # 4. 記憶状態スナップショットCSV
        if self.memory_state_snapshots:
            snapshots_df = pd.DataFrame(self.memory_state_snapshots)
            snapshots_df.to_csv(output_dir / "04_memory_state_snapshots.csv", index=False)
        
        # 5. TopKログCSV（継承）
        if self.topk_logs:
            topk_data = []
            for log in self.topk_logs:
                base_data = {
                    'current_episode_id': log['current_episode_id'],
                    'current_domain': log['current_domain'],
                    'current_research_area': log['current_research_area'],
                    'cross_domain_count': log['cross_domain_count'],
                    'timestamp': log['timestamp']
                }
                
                for i, topk_ep in enumerate(log['topk_episodes']):
                    row_data = base_data.copy()
                    row_data.update({
                        f'rank_{i+1}_episode_id': topk_ep['episode_id'],
                        f'rank_{i+1}_similarity': topk_ep['similarity'],
                        f'rank_{i+1}_domain': topk_ep['domain'],
                        f'rank_{i+1}_research_area': topk_ep['research_area'],
                        f'rank_{i+1}_is_cross_domain': topk_ep['is_cross_domain']
                    })
                    topk_data.append(row_data)
            
            if topk_data:
                topk_df = pd.DataFrame(topk_data)
                topk_df.to_csv(output_dir / "05_topk_analysis.csv", index=False)
        
        # 6. 詳細ログCSV（継承）
        if self.detailed_logs:
            detailed_df = pd.DataFrame(self.detailed_logs)
            detailed_df.to_csv(output_dir / "06_detailed_episode_logs.csv", index=False)
        
        # 7. 動的記憶実験メタデータJSON
        metadata = {
            'experiment_name': '動的記憶洞察実験 (詳細ログ版ベース)',
            'timestamp': datetime.now().isoformat(),
            'total_episodes': len(episodes),
            'total_insights': len(insights),
            'dynamic_memory_additions': self.insight_memory_additions,
            'insight_rate': len(insights)/len(episodes) if episodes else 0,
            'dynamic_addition_rate': self.insight_memory_additions/len(insights) if insights else 0,
            'final_memory_size': final_memory_size,
            'total_time_seconds': total_time,
            'avg_episodes_per_second': avg_eps_per_sec,
            'avg_processing_time': avg_processing_time,
            'parameters': {
                'memory_dim': 384,
                'topk': self.topk,
                'ged_threshold': self.ged_threshold,
                'ig_threshold': self.ig_threshold,
                'insight_memory_boost': self.insight_memory_boost,
                'normal_memory_c': self.normal_memory_c
            },
            'memory_state_changes': {
                'snapshots_captured': len(self.memory_state_snapshots),
                'initial_size': self._get_memory_size_from_snapshot(0) if self.memory_state_snapshots else 0,
                'final_size': self._get_memory_size_from_snapshot(-1) if self.memory_state_snapshots else 0
            }
        }
        
        with open(output_dir / "07_dynamic_memory_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 動的記憶実験結果保存完了:")
        print(f"   📁 出力ディレクトリ: {output_dir}")
        print(f"   📄 入力エピソード: 01_input_episodes.csv")
        print(f"   📄 動的洞察結果: 02_dynamic_insights.csv")
        print(f"   📄 動的記憶ログ: 03_dynamic_memory_logs.csv")
        print(f"   📄 記憶状態変化: 04_memory_state_snapshots.csv")
        print(f"   📄 TopK分析: 05_topk_analysis.csv")
        print(f"   📄 詳細ログ: 06_detailed_episode_logs.csv")
        print(f"   📄 実験メタデータ: 07_dynamic_memory_metadata.json")
    
    def _get_memory_size_from_snapshot(self, index):
        """スナップショットからメモリサイズを安全に取得"""
        if not self.memory_state_snapshots:
            return 0
        
        snapshot = self.memory_state_snapshots[index]
        
        # 可能なキーを順番に試す
        for key in ['total_memory_size', 'memory_size', 'size']:
            if key in snapshot:
                return snapshot[key]
        
        # どのキーも見つからない場合はゼロを返す
        print(f"⚠️  スナップショット[{index}]のキー: {list(snapshot.keys())}")
        return 0


def main():
    """メイン実行関数"""
    experiment = None
    
    print("🧠 動的記憶洞察実験を開始します（詳細ログ版ベース）")
    print("=" * 70)
    print("実験内容:")
    print("  1. 通常のエピソード処理と洞察検出")
    print("  2. 洞察検出時に強化エピソードを動的に記憶追加")
    print("  3. 動的記憶が次の洞察検出に与える影響を観察")
    print("  4. 実際のdataフォルダに永続化（backup/restore付き）")
    print()
    
    try:
        # 実験システム初期化
        experiment = DynamicMemoryInsightExperiment()
        
        # データディレクトリバックアップ
        print("📦 データディレクトリをバックアップ中...")
        experiment.backup_data_directory()
        
        # 動的記憶実験実行（300エピソードで効果を観察）
        print("🔄 動的記憶実験を実行中...")
        results = experiment.run_dynamic_memory_experiment(num_episodes=300)
        
        print("\n🎉 動的記憶洞察実験が正常に完了しました!")
        print("   動的記憶追加: ✅")
        print("   記憶状態追跡: ✅") 
        print("   影響効果測定: ✅")
        print("   実dataフォルダ書き込み: ✅")
        
        # 実験効果の簡単な分析
        if results['insights'] and results['dynamic_additions'] > 0:
            dynamic_rate = results['dynamic_additions'] / len(results['insights'])
            print(f"\n📊 動的記憶効果サマリー:")
            print(f"   洞察検出: {len(results['insights'])}件")
            print(f"   動的記憶追加: {results['dynamic_additions']}件")
            print(f"   動的記憶率: {dynamic_rate:.1%}")
            print(f"   最終メモリサイズ: {results['final_memory_size']}")
        
    except Exception as e:
        print(f"❌ 実験実行エラー: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # データディレクトリクリーンアップ（実験成功・失敗に関わらず実行）
        if experiment:
            print("\n🧹 データディレクトリをクリーンアップ中...")
            experiment.cleanup_experiment_files()
            experiment.restore_data_directory()
            print("✅ データディレクトリを元の状態に復元しました")


if __name__ == "__main__":
    main()
