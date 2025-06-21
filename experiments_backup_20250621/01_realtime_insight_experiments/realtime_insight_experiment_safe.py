#!/usr/bin/env python3
"""
実践的リアルタイム洞察検出実験 - セグメンテーションフォルト解決版
=======================================================

SafeMainAgentを使用して、毎エピソード洞察検出とグラフビジュアライゼーションを実装
"""

import sys
import os
import json
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# InsightSpike-AIのパスを追加
sys.path.append(str(Path(__file__).parent.parent / "src"))

import logging
from safe_main_agent_test import SafeMainAgent

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeInsightDetector:
    """リアルタイム洞察検出器"""
    
    def __init__(self):
        self.agent = SafeMainAgent()
        self.insight_events = []
        self.episode_history = []
        self.similarity_timeline = []
        self.graph_growth_data = []
        
        # 洞察検出パラメータ
        self.similarity_threshold = 0.7  # 高類似度閾値
        self.novelty_threshold = 0.3     # 新規性閾値
        self.insight_window = 5          # 直前5エピソードで評価
        
    def initialize(self) -> bool:
        """検出器を初期化"""
        logger.info("🚀 Initializing RealTimeInsightDetector...")
        return self.agent.initialize()
    
    def calculate_episode_novelty(self, new_episode: str) -> float:
        """新しいエピソードの新規性を計算"""
        if len(self.episode_history) == 0:
            return 1.0  # 最初のエピソードは完全に新規
        
        # 直近のエピソードとの類似度を計算
        recent_episodes = self.episode_history[-self.insight_window:]
        
        max_similarity = 0.0
        for past_episode in recent_episodes:
            # 簡易類似度計算（語彙重複ベース）
            new_words = set(new_episode.lower().split())
            past_words = set(past_episode['text'].lower().split())
            
            if len(new_words | past_words) > 0:
                similarity = len(new_words & past_words) / len(new_words | past_words)
                max_similarity = max(max_similarity, similarity)
        
        return 1.0 - max_similarity  # 新規性 = 1 - 最大類似度
    
    def detect_insight_spike(self, episode_id: int, novelty: float, memory_stats: Dict) -> Optional[Dict]:
        """洞察スパイクを検出"""
        try:
            # 洞察検出条件
            high_novelty = novelty > self.novelty_threshold
            memory_growth = memory_stats.get('total_episodes', 0) >= episode_id
            
            # スパイク強度を計算
            spike_strength = novelty
            if len(self.similarity_timeline) > 0:
                avg_novelty = np.mean([s['novelty'] for s in self.similarity_timeline[-10:]])
                spike_strength = novelty / (avg_novelty + 0.1)
            
            # 洞察検出
            insight_detected = high_novelty and spike_strength > 1.5
            
            if insight_detected:
                insight_event = {
                    'insight_id': f"RT_INS_{episode_id:04d}",
                    'episode_id': episode_id,
                    'novelty_score': novelty,
                    'spike_strength': spike_strength,
                    'detection_timestamp': datetime.now().isoformat(),
                    'insight_type': self._classify_insight_type(novelty, spike_strength),
                    'memory_state': memory_stats.copy()
                }
                
                logger.info(f"🔥 Insight detected: {insight_event['insight_id']} (novelty: {novelty:.3f})")
                return insight_event
            
            return None
            
        except Exception as e:
            logger.error(f"Insight detection failed: {e}")
            return None
    
    def _classify_insight_type(self, novelty: float, spike_strength: float) -> str:
        """洞察タイプを分類"""
        if novelty > 0.8 and spike_strength > 3.0:
            return "Breakthrough_Insight"
        elif novelty > 0.6 and spike_strength > 2.5:
            return "Major_Discovery"
        elif novelty > 0.4 and spike_strength > 2.0:
            return "Conceptual_Shift"
        else:
            return "Minor_Innovation"
    
    def process_episode(self, episode_text: str, episode_id: int) -> Dict[str, Any]:
        """単一エピソードを処理"""
        try:
            start_time = time.time()
            
            # 1. 新規性計算
            novelty = self.calculate_episode_novelty(episode_text)
            
            # 2. エピソード保存
            success = self.agent.store_episode(episode_text, c_value=0.5)
            
            # 3. メモリ統計取得
            memory_stats = self.agent.get_memory_stats()
            
            # 4. 洞察検出
            insight_event = self.detect_insight_spike(episode_id, novelty, memory_stats)
            if insight_event:
                self.insight_events.append(insight_event)
            
            # 5. 処理時間計算
            processing_time = time.time() - start_time
            
            # 6. データ記録
            episode_data = {
                'episode_id': episode_id,
                'text': episode_text,
                'novelty': novelty,
                'processing_time': processing_time,
                'storage_success': success,
                'timestamp': datetime.now().isoformat()
            }
            
            self.episode_history.append(episode_data)
            
            # 7. 類似度タイムライン更新
            self.similarity_timeline.append({
                'episode_id': episode_id,
                'novelty': novelty,
                'memory_size': memory_stats.get('total_episodes', 0),
                'timestamp': datetime.now().timestamp()
            })
            
            # 8. グラフ成長データ更新
            self.graph_growth_data.append({
                'episode_id': episode_id,
                'total_episodes': memory_stats.get('total_episodes', 0),
                'dimension': memory_stats.get('dimension', 384),
                'index_trained': memory_stats.get('index_trained', False),
                'insight_detected': insight_event is not None
            })
            
            return {
                'success': success,
                'novelty': novelty,
                'processing_time': processing_time,
                'insight_detected': insight_event is not None,
                'memory_stats': memory_stats
            }
            
        except Exception as e:
            logger.error(f"Episode processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_experiment(self, num_episodes: int = 1000):
        """リアルタイム洞察検出実験を実行"""
        logger.info(f"🚀 Starting real-time insight detection experiment ({num_episodes} episodes)")
        
        start_time = time.time()
        
        # エピソード生成とリアルタイム処理
        for i in range(1, num_episodes + 1):
            episode_text = self._generate_episode(i)
            result = self.process_episode(episode_text, i)
            
            # 進捗表示
            if i % 100 == 0:
                elapsed = time.time() - start_time
                eps_per_sec = i / elapsed
                logger.info(f"📈 Progress: {i}/{num_episodes} ({eps_per_sec:.1f} eps/sec, {len(self.insight_events)} insights)")
        
        # 実験完了
        total_time = time.time() - start_time
        logger.info(f"✅ Experiment completed in {total_time:.2f}s")
        logger.info(f"   Total insights detected: {len(self.insight_events)}")
        logger.info(f"   Average processing speed: {num_episodes/total_time:.2f} eps/sec")
        
        return {
            'total_episodes': num_episodes,
            'total_insights': len(self.insight_events),
            'total_time': total_time,
            'avg_eps_per_sec': num_episodes / total_time
        }
    
    def _generate_episode(self, episode_id: int) -> str:
        """エピソードを生成"""
        # 10のベーストピック
        topics = [
            "AI healthcare", "ML training", "Deep learning", "NLP interaction",
            "Computer vision", "Predictive analytics", "Data science", 
            "Neural networks", "Automation", "Personalized medicine"
        ]
        
        # 修正とバリエーション
        modifications = [
            "advanced algorithms", "large datasets", "real-time processing",
            "improved accuracy", "cost reduction", "enhanced security",
            "cloud integration", "mobile optimization", "user experience",
            "scalability", "performance", "innovation", "automation",
            "intelligence", "efficiency", "reliability", "flexibility"
        ]
        
        # ランダムな組み合わせで新規性を注入
        topic = topics[episode_id % len(topics)]
        mod = modifications[(episode_id * 3) % len(modifications)]
        
        # 時々完全に新しいパターンを注入（洞察誘発）
        if episode_id % 137 == 0:  # 素数間隔で新パターン
            return f"Revolutionary breakthrough in {topic}: {mod} enables unprecedented capabilities through quantum-enhanced methodologies."
        elif episode_id % 73 == 0:
            return f"Paradigm shift discovered: {topic} integration with {mod} creates emergent properties beyond traditional approaches."
        else:
            return f"{topic.title()} systems can leverage {mod} for enhanced performance and user satisfaction."
    
    def visualize_results(self):
        """結果をビジュアライゼーション"""
        logger.info("📊 Generating visualizations...")
        
        # 出力ディレクトリ作成
        output_dir = Path("experiments/outputs/realtime_insights")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 新規性タイムライン
        plt.figure(figsize=(12, 6))
        episodes = [s['episode_id'] for s in self.similarity_timeline]
        novelties = [s['novelty'] for s in self.similarity_timeline]
        
        plt.subplot(2, 1, 1)
        plt.plot(episodes, novelties, 'b-', alpha=0.7, linewidth=1)
        plt.axhline(y=self.novelty_threshold, color='r', linestyle='--', label='Novelty Threshold')
        
        # 洞察イベントをハイライト
        for insight in self.insight_events:
            plt.axvline(x=insight['episode_id'], color='red', alpha=0.8, linewidth=2)
        
        plt.title('Episode Novelty Timeline')
        plt.xlabel('Episode ID')
        plt.ylabel('Novelty Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. メモリ成長
        plt.subplot(2, 1, 2)
        episodes = [g['episode_id'] for g in self.graph_growth_data]
        memory_sizes = [g['total_episodes'] for g in self.graph_growth_data]
        
        plt.plot(episodes, memory_sizes, 'g-', linewidth=2, label='Memory Size')
        
        # 洞察イベントをマーク
        insight_episodes = [i['episode_id'] for i in self.insight_events]
        insight_memory_sizes = [g['total_episodes'] for g in self.graph_growth_data 
                               if g['episode_id'] in insight_episodes]
        
        plt.scatter(insight_episodes, insight_memory_sizes, 
                   color='red', s=100, marker='*', label='Insights Detected', zorder=5)
        
        plt.title('Memory Growth & Insight Detection')
        plt.xlabel('Episode ID')
        plt.ylabel('Total Episodes in Memory')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "realtime_insights_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 洞察分布ヒートマップ
        if len(self.insight_events) > 0:
            plt.figure(figsize=(10, 6))
            
            # 洞察タイプの分布
            insight_types = [i['insight_type'] for i in self.insight_events]
            unique_types = list(set(insight_types))
            type_counts = [insight_types.count(t) for t in unique_types]
            
            plt.bar(unique_types, type_counts, color='skyblue')
            plt.title('Distribution of Insight Types')
            plt.xlabel('Insight Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "insight_type_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"📊 Visualizations saved to: {output_dir}")
    
    def save_detailed_summary(self):
        """詳細サマリを保存"""
        logger.info("💾 Saving detailed summary...")
        
        output_dir = Path("experiments/outputs/realtime_insights")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 入力エピソード詳細
        episodes_file = output_dir / "01_input_episodes_realtime.csv"
        with open(episodes_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['episode_id', 'episode_text', 'novelty_score', 'processing_time', 'timestamp'])
            
            for ep in self.episode_history:
                writer.writerow([
                    ep['episode_id'], ep['text'], ep['novelty'], 
                    ep['processing_time'], ep['timestamp']
                ])
        
        # 2. リアルタイム洞察イベント
        insights_file = output_dir / "02_realtime_insights_detailed.csv"
        with open(insights_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'insight_id', 'episode_id', 'insight_type', 'novelty_score', 
                'spike_strength', 'detection_timestamp', 'memory_size'
            ])
            
            for insight in self.insight_events:
                writer.writerow([
                    insight['insight_id'], insight['episode_id'], insight['insight_type'],
                    insight['novelty_score'], insight['spike_strength'], 
                    insight['detection_timestamp'], insight['memory_state'].get('total_episodes', 0)
                ])
        
        # 3. 完全詳細JSON
        full_details_file = output_dir / "03_realtime_experiment_full_details.json"
        full_data = {
            'experiment_metadata': {
                'experiment_type': 'realtime_insight_detection',
                'total_episodes': len(self.episode_history),
                'total_insights': len(self.insight_events),
                'novelty_threshold': self.novelty_threshold,
                'insight_window': self.insight_window,
                'generation_timestamp': datetime.now().isoformat()
            },
            'episode_history': self.episode_history,
            'insight_events': self.insight_events,
            'similarity_timeline': self.similarity_timeline,
            'graph_growth_data': self.graph_growth_data
        }
        
        with open(full_details_file, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)
        
        # 4. サマリ統計
        summary_file = output_dir / "04_experiment_summary.json"
        summary = {
            'total_episodes_processed': len(self.episode_history),
            'total_insights_detected': len(self.insight_events),
            'insight_detection_rate': len(self.insight_events) / len(self.episode_history) if self.episode_history else 0,
            'average_novelty': np.mean([ep['novelty'] for ep in self.episode_history]) if self.episode_history else 0,
            'average_processing_time': np.mean([ep['processing_time'] for ep in self.episode_history]) if self.episode_history else 0,
            'insight_types_detected': list(set([i['insight_type'] for i in self.insight_events])),
            'peak_novelty_episode': max(self.episode_history, key=lambda x: x['novelty'])['episode_id'] if self.episode_history else None
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Detailed summary saved:")
        logger.info(f"   📄 Episodes: {episodes_file}")
        logger.info(f"   📄 Insights: {insights_file}")  
        logger.info(f"   📄 Full details: {full_details_file}")
        logger.info(f"   📄 Summary: {summary_file}")


def main():
    """メイン実行関数"""
    detector = RealTimeInsightDetector()
    
    try:
        # 初期化
        if not detector.initialize():
            logger.error("Failed to initialize detector")
            return
        
        # 実験実行
        results = detector.run_experiment(num_episodes=1000)
        
        # 結果保存とビジュアライゼーション
        detector.save_detailed_summary()
        detector.visualize_results()
        
        logger.info("🎉 Real-time insight detection experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
