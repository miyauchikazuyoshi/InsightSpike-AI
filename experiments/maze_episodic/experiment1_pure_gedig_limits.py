#!/usr/bin/env python3
"""
実験1: 純粋geDIG実装の限界確認
================================

正確なgeDIG定義に基づく実装の成功率と失敗パターンを分析
"""

from true_gedig_flow_navigator import TrueGeDIGFlowNavigator
from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path
import numpy as np
import time

def test_single_maze(size, seed=42, max_steps=None, verbose=True):
    """単一迷路でのテスト"""
    if max_steps is None:
        max_steps = size * size * 2
    
    maze = create_complex_maze(size, seed=seed)
    nav = TrueGeDIGFlowNavigator(maze)
    
    start_time = time.time()
    
    # ナビゲーション
    step = 0
    path_history = []
    
    while step < max_steps and nav.position != nav.goal_pos:
        direction, reached_goal = nav.navigate_step()
        path_history.append(nav.position)
        
        # 循環検出
        if len(path_history) >= 10:
            recent = path_history[-10:]
            if recent[:5] == recent[5:]:
                if verbose:
                    print(f"  循環検出: {recent[:5]}")
                return False, step, "cycle", time.time() - start_time
        
        step += 1
        if reached_goal:
            break
    
    success = nav.position == nav.goal_pos
    elapsed = time.time() - start_time
    
    if success:
        return True, step, "success", elapsed
    else:
        return False, step, "timeout", elapsed

def run_experiment():
    """実験実行"""
    print("="*70)
    print("実験1: 純粋geDIG実装の限界確認")
    print("="*70)
    print("\n正確なgeDIG定義に基づく実装:")
    print("- 6次元エピソード記憶")
    print("- 初期記憶（4方向+ゴール）")
    print("- ドーナツ検索（内径0.0, 外径1.5）")
    print("- geDIG最小化による記憶選定")
    print("- 洞察エピソード形成と方向正規化")
    
    # テストサイズ
    sizes = [5, 10, 15, 20]
    trials_per_size = 3
    
    results = {}
    
    for size in sizes:
        print(f"\n\n{size}×{size} 迷路でのテスト:")
        print("-" * 40)
        
        successes = 0
        total_steps = []
        failure_reasons = []
        
        for trial in range(trials_per_size):
            print(f"\n試行 {trial+1}/{trials_per_size}:")
            
            success, steps, reason, elapsed = test_single_maze(
                size, seed=42+trial, verbose=True
            )
            
            if success:
                successes += 1
                total_steps.append(steps)
                print(f"  ✓ 成功: {steps}ステップ ({elapsed:.1f}秒)")
            else:
                failure_reasons.append(reason)
                print(f"  ✗ 失敗: {reason} ({steps}ステップ, {elapsed:.1f}秒)")
        
        # 統計
        success_rate = successes / trials_per_size
        avg_steps = np.mean(total_steps) if total_steps else 0
        
        results[size] = {
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'failures': failure_reasons
        }
        
        print(f"\n{size}×{size} まとめ:")
        print(f"  成功率: {success_rate*100:.0f}% ({successes}/{trials_per_size})")
        if total_steps:
            print(f"  平均ステップ数: {avg_steps:.0f}")
        print(f"  失敗理由: {failure_reasons}")
    
    # 全体まとめ
    print("\n\n" + "="*70)
    print("実験結果まとめ")
    print("="*70)
    
    print(f"\n{'Size':<10} {'Success Rate':<15} {'Avg Steps':<15} {'Failure Pattern'}")
    print("-" * 60)
    
    for size, result in results.items():
        failures = ', '.join(result['failures']) if result['failures'] else 'N/A'
        print(f"{size}×{size:<5} {result['success_rate']*100:>6.0f}%        "
              f"{result['avg_steps']:>8.0f}        {failures}")
    
    # 分析
    print("\n\n分析:")
    print("1. 純粋なgeDIG最小化の問題点:")
    print("   - ゴールエピソードのIG（情報獲得）が大きすぎる")
    print("   - 常にゴールエピソードが選択される")
    print("   - 実際の移動可能性を考慮していない")
    
    print("\n2. 失敗パターン:")
    for size, result in results.items():
        if result['failures']:
            if 'cycle' in result['failures']:
                print(f"   - {size}×{size}: 循環（同じ場所を繰り返す）")
            if 'timeout' in result['failures']:
                print(f"   - {size}×{size}: タイムアウト（探索が不十分）")
    
    print("\n3. 結論:")
    print("   - 現在の実装では壁情報を適切に学習できない")
    print("   - geDIG計算のIG部分を改善する必要がある")
    print("   - またはエピソード表現を工夫する必要がある")

if __name__ == "__main__":
    run_experiment()