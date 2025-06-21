#!/usr/bin/env python3
"""
動的記憶の影響分析スクリプト
洞察エピソードが後続エピソードの検出に与える影響を分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def load_experiment_data():
    """実験データを読み込み"""
    base_path = Path("experiments/01_realtime_insight_experiments/outputs/dynamic_memory_detailed")
    
    # 各CSVファイルを読み込み
    episodes = pd.read_csv(base_path / "01_input_episodes.csv")
    insights = pd.read_csv(base_path / "02_dynamic_insights.csv")
    memory_logs = pd.read_csv(base_path / "03_dynamic_memory_logs.csv")
    detailed_logs = pd.read_csv(base_path / "06_detailed_episode_logs.csv")
    
    # メタデータを読み込み
    with open(base_path / "07_dynamic_memory_metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return episodes, insights, memory_logs, detailed_logs, metadata

def analyze_insight_impact():
    """洞察エピソードの影響を分析"""
    print("🔍 動的記憶の影響分析を開始...")
    
    episodes, insights, memory_logs, detailed_logs, metadata = load_experiment_data()
    
    print(f"📊 実験データ概要:")
    print(f"   総エピソード数: {len(episodes)}")
    print(f"   洞察検出数: {len(insights)}")
    print(f"   洞察検出率: {len(insights)/len(episodes)*100:.1f}%")
    
    # 洞察エピソードのIDを抽出
    insight_episode_ids = set(insights['episode_id'].values)
    
    # 詳細ログを分析
    detailed_logs['is_insight'] = detailed_logs['episode_id'].isin(insight_episode_ids)
    detailed_logs['episode_id'] = detailed_logs['episode_id'].astype(int)
    detailed_logs = detailed_logs.sort_values('episode_id')
    
    # 各エピソードで動的に追加された洞察の累積数を計算
    dynamic_additions_cumsum = []
    current_additions = 0
    
    for _, row in detailed_logs.iterrows():
        if row['is_insight']:
            current_additions += 1
        dynamic_additions_cumsum.append(current_additions)
    
    detailed_logs['cumulative_dynamic_additions'] = dynamic_additions_cumsum
    
    # 時系列での洞察検出率の変化を分析
    window_size = 50  # 50エピソードのウィンドウで分析
    
    insight_rates = []
    memory_sizes = []
    dynamic_additions = []
    window_centers = []
    
    for i in range(window_size//2, len(detailed_logs) - window_size//2):
        window_start = i - window_size//2
        window_end = i + window_size//2
        
        window_data = detailed_logs.iloc[window_start:window_end]
        insight_rate = window_data['is_insight'].mean() * 100
        avg_memory_size = window_data['memory_size'].mean()
        avg_dynamic = window_data['cumulative_dynamic_additions'].mean()
        
        insight_rates.append(insight_rate)
        memory_sizes.append(avg_memory_size)
        dynamic_additions.append(avg_dynamic)
        window_centers.append(i)
    
    # GED値の分布を時系列で分析
    print("\n📈 時系列でのGED値分布分析:")
    
    # 洞察エピソードと非洞察エピソードのGED値を比較
    insight_episodes = detailed_logs[detailed_logs['is_insight']]
    non_insight_episodes = detailed_logs[~detailed_logs['is_insight']]
    
    print(f"   洞察エピソードの平均ΔGED: {insight_episodes['delta_ged'].mean():.3f}")
    print(f"   非洞察エピソードの平均ΔGED: {non_insight_episodes['delta_ged'].mean():.3f}")
    
    # 動的記憶追加前後での非洞察エピソードのGED値変化
    print("\n🎯 動的記憶の影響分析:")
    
    # 各洞察エピソードの直後のエピソードを分析
    post_insight_analysis = []
    
    for insight_id in insight_episode_ids:
        # 洞察エピソードの直後の5エピソードを分析
        post_episodes = detailed_logs[
            (detailed_logs['episode_id'] > insight_id) & 
            (detailed_logs['episode_id'] <= insight_id + 5) &
            (~detailed_logs['is_insight'])
        ]
        
        if len(post_episodes) > 0:
            avg_ged = post_episodes['delta_ged'].mean()
            post_insight_analysis.append({
                'insight_episode_id': insight_id,
                'post_insight_avg_ged': avg_ged,
                'post_insight_count': len(post_episodes)
            })
    
    post_insight_df = pd.DataFrame(post_insight_analysis)
    
    if len(post_insight_df) > 0:
        print(f"   洞察直後エピソードの平均GED: {post_insight_df['post_insight_avg_ged'].mean():.3f}")
        
        # 全体の非洞察エピソードと比較
        overall_non_insight_ged = non_insight_episodes['delta_ged'].mean()
        post_insight_ged = post_insight_df['post_insight_avg_ged'].mean()
        
        print(f"   全体非洞察エピソード平均ΔGED: {overall_non_insight_ged:.3f}")
        print(f"   洞察直後エピソード平均ΔGED: {post_insight_ged:.3f}")
        print(f"   差分: {post_insight_ged - overall_non_insight_ged:.3f}")
        
        if post_insight_ged < overall_non_insight_ged:
            print("   ✅ 洞察エピソードが「予習効果」を持っていることを確認！")
            print("      洞察直後のエピソードはより類似度が高く、ΔGED値が低い")
        else:
            print("   ❌ 明確な予習効果は観測されず")
    
    # メモリサイズと洞察検出率の関係
    print("\n📊 メモリサイズと洞察検出の関係:")
    
    # エピソードを4つの期間に分割して分析
    total_episodes = len(detailed_logs)
    quarter_size = total_episodes // 4
    
    for i in range(4):
        start_idx = i * quarter_size
        end_idx = (i + 1) * quarter_size if i < 3 else total_episodes
        
        quarter_data = detailed_logs.iloc[start_idx:end_idx]
        quarter_insight_rate = quarter_data['is_insight'].mean() * 100
        quarter_avg_memory = quarter_data['memory_size'].mean()
        quarter_avg_ged = quarter_data['delta_ged'].mean()
        
        print(f"   第{i+1}四半期 (エピソード {start_idx+1}-{end_idx}):")
        print(f"     洞察検出率: {quarter_insight_rate:.1f}%")
        print(f"     平均メモリサイズ: {quarter_avg_memory:.0f}")
        print(f"     平均ΔGED値: {quarter_avg_ged:.3f}")
    
    # 可視化データを準備
    visualization_data = {
        'window_centers': window_centers,
        'insight_rates': insight_rates,
        'memory_sizes': memory_sizes,
        'dynamic_additions': dynamic_additions,
        'detailed_logs': detailed_logs,
        'insight_episodes': insight_episodes,
        'non_insight_episodes': non_insight_episodes
    }
    
    return visualization_data

def create_visualizations(data):
    """分析結果の可視化"""
    print("\n📊 可視化を作成中...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('動的記憶洞察実験の影響分析', fontsize=16, fontweight='bold')
    
    # 1. 洞察検出率の時系列変化
    ax1 = axes[0, 0]
    ax1.plot(data['window_centers'], data['insight_rates'], 'b-', linewidth=2, label='洞察検出率')
    ax1.set_xlabel('エピソード番号')
    ax1.set_ylabel('洞察検出率 (%)')
    ax1.set_title('洞察検出率の時系列変化')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. メモリサイズの時系列変化
    ax2 = axes[0, 1]
    ax2.plot(data['window_centers'], data['memory_sizes'], 'g-', linewidth=2, label='メモリサイズ')
    ax2.set_xlabel('エピソード番号')
    ax2.set_ylabel('メモリサイズ')
    ax2.set_title('メモリサイズの時系列変化')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. GED値の分布比較
    ax3 = axes[1, 0]
    ax3.hist(data['insight_episodes']['delta_ged'], bins=30, alpha=0.7, label='洞察エピソード', color='red')
    ax3.hist(data['non_insight_episodes']['delta_ged'], bins=30, alpha=0.7, label='非洞察エピソード', color='blue')
    ax3.set_xlabel('ΔGED値')
    ax3.set_ylabel('頻度')
    ax3.set_title('ΔGED値の分布比較')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 洞察検出率とメモリサイズの関係
    ax4 = axes[1, 1]
    scatter = ax4.scatter(data['memory_sizes'], data['insight_rates'], 
                         c=data['dynamic_additions'], cmap='viridis', alpha=0.7)
    ax4.set_xlabel('メモリサイズ')
    ax4.set_ylabel('洞察検出率 (%)')
    ax4.set_title('メモリサイズと洞察検出率の関係')
    ax4.grid(True, alpha=0.3)
    
    # カラーバーを追加
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('累積動的追加数')
    
    plt.tight_layout()
    
    # 画像を保存
    output_path = Path("experiments/01_realtime_insight_experiments/outputs/dynamic_memory_detailed")
    plt.savefig(output_path / "08_dynamic_memory_impact_analysis.png", dpi=300, bbox_inches='tight')
    print(f"   可視化を保存: {output_path / '08_dynamic_memory_impact_analysis.png'}")
    
    plt.show()

def main():
    """メイン分析関数"""
    print("🧠 動的記憶の影響分析")
    print("=" * 60)
    
    try:
        # データ分析
        visualization_data = analyze_insight_impact()
        
        # 可視化
        create_visualizations(visualization_data)
        
        print("\n✅ 分析完了!")
        print("\n📋 分析結果サマリー:")
        print("   1. 洞察エピソードが後続エピソードに与える「予習効果」を検証")
        print("   2. 動的記憶によるメモリサイズの増加と洞察検出率の関係を分析")
        print("   3. GED値の分布変化から類似度の影響を評価")
        print("   4. 時系列での洞察検出パターンの変化を可視化")
        
    except Exception as e:
        print(f"❌ 分析中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
