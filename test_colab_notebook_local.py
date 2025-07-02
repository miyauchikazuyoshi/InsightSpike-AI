#!/usr/bin/env python3
"""
ローカルでColab実験ノートブックの内容をテストするスクリプト
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from pathlib import Path

# プロジェクトのsrcディレクトリをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

def test_basic_imports():
    """基本的なインポートをテスト"""
    print("🎯 基本パッケージのテスト:")
    print(f"   NumPy: {np.__version__}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   Pandas: {pd.__version__}")
    print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    return True

def test_insightspike_imports():
    """InsightSpikeのインポートをテスト"""
    try:
        from insightspike.core.rag_system import SimpleRAGSystem
        print("✅ InsightSpike-AI: SimpleRAGSystem インポート成功")
        
        # システムの初期化テスト
        system = SimpleRAGSystem()
        print("✅ InsightSpike-AI: システム初期化成功")
        return True
        
    except Exception as e:
        print(f"❌ InsightSpike-AI インポートエラー: {e}")
        return False

def create_fallback_classes():
    """フォールバック用の基本クラスを作成"""
    print("🔧 フォールバック: 基本的な実験クラスを作成")
    
    class SimpleGridWorld:
        def __init__(self, size=8, num_obstacles=5):
            self.size = size
            self.grid = np.zeros((size, size))
            
            # ランダムに障害物を配置
            obstacles = np.random.choice(size*size, num_obstacles, replace=False)
            for obs in obstacles:
                row, col = divmod(obs, size)
                self.grid[row, col] = -1
            
            # スタートとゴールを設定
            self.start_pos = (0, 0)
            self.goal_pos = (size-1, size-1)
            self.grid[self.goal_pos] = 1
            self.current_pos = self.start_pos
            
            self.state_space_size = size * size
            self.action_space_size = 4  # 上下左右
            
        def reset(self):
            self.current_pos = self.start_pos
            return self.current_pos
            
        def step(self, action):
            # 簡単な移動ロジック
            row, col = self.current_pos
            
            if action == 0:  # 上
                row = max(0, row - 1)
            elif action == 1:  # 下
                row = min(self.size - 1, row + 1)
            elif action == 2:  # 左
                col = max(0, col - 1)
            elif action == 3:  # 右
                col = min(self.size - 1, col + 1)
            
            new_pos = (row, col)
            
            # 障害物チェック
            if self.grid[new_pos] == -1:
                new_pos = self.current_pos  # 移動せず
            
            self.current_pos = new_pos
            
            # 報酬の計算
            if new_pos == self.goal_pos:
                reward = 100.0
                done = True
            else:
                reward = -1.0
                done = False
                
            return new_pos, reward, done, {}
    
    class IntrinsicMotivationAgent:
        def __init__(self, state_size, action_size, use_ged=True, use_ig=True):
            self.state_size = state_size
            self.action_size = action_size
            self.use_ged = use_ged
            self.use_ig = use_ig
            self.q_table = np.random.random((state_size, action_size)) * 0.1
            self.learning_rate = 0.1
            self.discount_factor = 0.95
            self.epsilon = 1.0
            self.epsilon_decay = 0.995
            self.epsilon_min = 0.01
            
        def act(self, state):
            if isinstance(state, tuple):
                state_idx = state[0] * int(np.sqrt(self.state_size)) + state[1]  # グリッド位置をインデックスに変換
            else:
                state_idx = state
            
            # インデックスの範囲チェック
            state_idx = min(state_idx, self.state_size - 1)
                
            if np.random.random() < self.epsilon:
                return np.random.randint(self.action_size)
            else:
                return np.argmax(self.q_table[state_idx])
                
        def update_q_table(self, state, action, reward, next_state):
            if isinstance(state, tuple):
                state_idx = state[0] * int(np.sqrt(self.state_size)) + state[1]
            else:
                state_idx = state
                
            if isinstance(next_state, tuple):
                next_state_idx = next_state[0] * int(np.sqrt(self.state_size)) + next_state[1]
            else:
                next_state_idx = next_state
            
            # インデックスの範囲チェック
            state_idx = min(state_idx, self.state_size - 1)
            next_state_idx = min(next_state_idx, self.state_size - 1)
            
            # 内発的報酬の計算
            intrinsic_reward = 0.0
            if self.use_ged:
                intrinsic_reward += np.random.random() * 0.1  # 簡単なGEDシミュレーション
            if self.use_ig:
                intrinsic_reward += np.random.random() * 0.1   # 簡単なIGシミュレーション
            
            total_reward = reward + intrinsic_reward
            
            # Q-learning更新
            current_q = self.q_table[state_idx, action]
            max_future_q = np.max(self.q_table[next_state_idx])
            new_q = current_q + self.learning_rate * (total_reward + self.discount_factor * max_future_q - current_q)
            self.q_table[state_idx, action] = new_q
            
            # epsilon減衰
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    
    return SimpleGridWorld, IntrinsicMotivationAgent

def run_simple_experiment():
    """簡単な実験を実行してテスト"""
    print("\n🧪 簡単な実験の実行:")
    
    # フォールバッククラスを作成
    SimpleGridWorld, IntrinsicMotivationAgent = create_fallback_classes()
    
    # 環境とエージェントを作成
    env = SimpleGridWorld(size=6, num_obstacles=3)
    
    configs = [
        {"name": "Full (ΔGED × ΔIG)", "use_ged": True, "use_ig": True},
        {"name": "No GED (ΔIG only)", "use_ged": False, "use_ig": True},
        {"name": "No IG (ΔGED only)", "use_ged": True, "use_ig": False},
        {"name": "Baseline (No intrinsic)", "use_ged": False, "use_ig": False}
    ]
    
    results = {}
    
    for config in configs:
        agent = IntrinsicMotivationAgent(
            state_size=env.state_space_size,
            action_size=env.action_space_size,
            use_ged=config["use_ged"],
            use_ig=config["use_ig"]
        )
        
        # 短いエピソードを実行
        episodes = 50
        success_count = 0
        episode_lengths = []
        
        for episode in range(episodes):
            state = env.reset()
            episode_length = 0
            
            for step in range(100):  # 最大100ステップ
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.update_q_table(state, action, reward, next_state)
                
                state = next_state
                episode_length += 1
                
                if done:
                    success_count += 1
                    break
            
            episode_lengths.append(episode_length)
        
        success_rate = success_count / episodes
        avg_episode_length = np.mean(episode_lengths)
        
        results[config["name"]] = {
            "success_rate": success_rate,
            "avg_episode_length": avg_episode_length
        }
        
        print(f"   {config['name']}: 成功率 {success_rate:.3f}, 平均エピソード長 {avg_episode_length:.1f}")
    
    return results

def create_visualization(results):
    """結果の可視化"""
    print("\n📈 結果の可視化:")
    
    configs = list(results.keys())
    success_rates = [results[config]["success_rate"] for config in configs]
    episode_lengths = [results[config]["avg_episode_length"] for config in configs]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 成功率のプロット
    bars1 = ax1.bar(range(len(configs)), success_rates, 
                    color=sns.color_palette("husl", len(configs)), alpha=0.7)
    ax1.set_title('Success Rates by Configuration')
    ax1.set_ylabel('Success Rate')
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels([c.replace(" (", "\n(") for c in configs], fontsize=9)
    
    # 値をバーの上に表示
    for i, rate in enumerate(success_rates):
        ax1.text(i, rate + 0.01, f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # エピソード長のプロット
    bars2 = ax2.bar(range(len(configs)), episode_lengths,
                    color=sns.color_palette("husl", len(configs)), alpha=0.7)
    ax2.set_title('Average Episode Length by Configuration')
    ax2.set_ylabel('Episode Length')
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels([c.replace(" (", "\n(") for c in configs], fontsize=9)
    
    # 値をバーの上に表示
    for i, length in enumerate(episode_lengths):
        ax2.text(i, length + 1, f'{length:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('colab_experiment_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ 結果をcolab_experiment_test_results.pngに保存")

def main():
    """メイン実行関数"""
    print("🚀 Colab実験ノートブックのローカルテスト開始")
    print("=" * 50)
    
    # 基本インポートテスト
    test_basic_imports()
    
    # InsightSpikeインポートテスト
    insightspike_available = test_insightspike_imports()
    
    if not insightspike_available:
        print("⚠️  InsightSpike-AIモジュールが利用できないため、フォールバックを使用")
    
    # 簡単な実験を実行
    results = run_simple_experiment()
    
    # 結果の可視化
    create_visualization(results)
    
    print("\n🎉 テスト完了!")
    print("📊 結果サマリー:")
    for config, result in results.items():
        print(f"   {config}: 成功率 {result['success_rate']:.3f}")
    
    print("\n💡 実際のColab実験では、より詳細な内発的報酬計算とより多くのエピソードが実行されます")

if __name__ == "__main__":
    # ランダムシードを設定
    np.random.seed(42)
    torch.manual_seed(42)
    
    main()