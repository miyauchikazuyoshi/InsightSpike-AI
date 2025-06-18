#!/usr/bin/env python3
"""
データ同期のテスト
=================

MainAgentのepisode追加とグラフ更新の同期をテストします。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging
from insightspike.core.agents.main_agent import MainAgent

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_sync():
    """データ同期機能のテスト"""
    print("🧪 Testing data synchronization functionality")
    
    # MainAgent初期化
    print("1. Initializing MainAgent...")
    agent = MainAgent()
    if not agent.initialize():
        print("❌ Failed to initialize MainAgent")
        return False
    print("✅ MainAgent initialized successfully")
    
    # 初期状態確認
    print("\n2. Checking initial state...")
    initial_state = agent.get_memory_graph_state()
    print(f"   Memory episodes: {initial_state['memory'].get('num_episodes', 0)}")
    print(f"   Graph nodes: {initial_state['graph'].get('num_nodes', 0)}")
    print(f"   Synchronized: {initial_state.get('synchronized', False)}")
    
    # Episode追加テスト
    print("\n3. Adding test episodes...")
    test_episodes = [
        "Machine learning involves pattern recognition in data.",
        "Neural networks are inspired by biological brain structures.",
        "Deep learning uses multiple layers for complex representations.",
        "Reinforcement learning learns through trial and error.",
        "Supervised learning uses labeled training data."
    ]
    
    for i, episode_text in enumerate(test_episodes):
        print(f"   Adding episode {i+1}: {episode_text[:50]}...")
        result = agent.add_episode_with_graph_update(episode_text, c_value=0.5)
        
        if result['success']:
            print(f"     ✅ Episode {result['episode_idx']} added successfully")
            
            # ΔGED/ΔIG確認
            if result['graph_analysis']:
                metrics = result['graph_analysis'].get('metrics', {})
                delta_ged = metrics.get('delta_ged', 0.0)
                delta_ig = metrics.get('delta_ig', 0.0)
                print(f"     ΔGED: {delta_ged:.4f}, ΔIG: {delta_ig:.4f}")
            else:
                print("     ⚠️  No graph analysis returned")
        else:
            print(f"     ❌ Episode addition failed: {result.get('error', 'Unknown error')}")
    
    # 最終状態確認
    print("\n4. Checking final state...")
    final_state = agent.get_memory_graph_state()
    print(f"   Memory episodes: {final_state['memory'].get('num_episodes', 0)}")
    print(f"   Graph nodes: {final_state['graph'].get('num_nodes', 0)}")
    print(f"   Synchronized: {final_state.get('synchronized', False)}")
    
    # データ保存テスト
    print("\n5. Testing data persistence...")
    saved = agent.save_state()
    if saved:
        print("✅ Data saved successfully")
    else:
        print("❌ Data save failed")
    
    # データ読み込みテスト
    print("\n6. Testing data loading...")
    new_agent = MainAgent()
    new_agent.initialize()
    loaded = new_agent.load_state()
    if loaded:
        print("✅ Data loaded successfully")
        loaded_state = new_agent.get_memory_graph_state()
        print(f"   Loaded episodes: {loaded_state['memory'].get('num_episodes', 0)}")
        print(f"   Loaded graph nodes: {loaded_state['graph'].get('num_nodes', 0)}")
    else:
        print("❌ Data load failed")
    
    print("\n🎉 Data synchronization test completed!")
    return True

if __name__ == "__main__":
    test_data_sync()
