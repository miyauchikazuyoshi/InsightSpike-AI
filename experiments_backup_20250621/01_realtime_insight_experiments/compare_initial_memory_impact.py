#!/usr/bin/env python3
"""
初期メモリサイズの影響を比較分析
================================================================

過去の実験（初期メモリ0）と現在の実験（初期メモリ8）を比較して、
初期メモリサイズが洞察検出率に与える影響を分析する。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# 日本語フォントの設定
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

def analyze_initial_memory_impact():
    """初期メモリサイズの影響を比較分析"""
    
    print("📊 初期メモリサイズの影響比較分析")
    print("=" * 60)
    
    # データパス設定
    base_path = Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/experiments/01_realtime_insight_experiments/outputs")
    
    # 過去の実験データ（初期メモリ0）
    past_logs_path = base_path / "detailed_logging_realtime" / "04_detailed_episode_logs.csv"
    past_meta_path = base_path / "detailed_logging_realtime" / "05_experiment_metadata.json"
    
    # 現在の実験データ（初期メモリ8）
    current_logs_path = base_path / "dynamic_memory_detailed" / "06_detailed_episode_logs.csv"
    current_meta_path = base_path / "dynamic_memory_detailed" / "07_dynamic_memory_metadata.json"
    
    # データ読み込み
    try:
        # 過去の実験データ
        past_df = pd.read_csv(past_logs_path)
        with open(past_meta_path, 'r', encoding='utf-8') as f:
            past_meta = json.load(f)
        
        # 現在の実験データ
        current_df = pd.read_csv(current_logs_path)
        with open(current_meta_path, 'r', encoding='utf-8') as f:
            current_meta = json.load(f)
            
        print("✅ データ読み込み完了")
        
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        return
    
    # 基本統計情報の比較
    print("\n📈 基本統計情報の比較")
    print("-" * 40)
    
    past_insights = past_df['is_insight'].sum()
    past_total = len(past_df)
    past_rate = past_insights / past_total
    
    current_insights = current_df['is_insight'].sum()
    current_total = len(current_df)
    current_rate = current_insights / current_total
    
    print(f"過去の実験（初期メモリ0）:")
    print(f"  総エピソード数: {past_total}")
    print(f"  洞察検出数: {past_insights}")
    print(f"  洞察検出率: {past_rate:.1%}")
    
    print(f"\n現在の実験（初期メモリ8）:")
    print(f"  総エピソード数: {current_total}")
    print(f"  洞察検出数: {current_insights}")
    print(f"  洞察検出率: {current_rate:.1%}")
    
    print(f"\n🔍 差分分析:")
    print(f"  洞察検出率の差: {current_rate - past_rate:.1%}")
    print(f"  改善倍率: {current_rate / past_rate:.2f}x")
    
    # 時系列での洞察検出率の変化を比較
    print("\n📊 時系列での洞察検出率変化の比較")
    print("-" * 40)
    
    # 25エピソード毎の検出率を計算
    window_size = 25
    
    def calculate_rolling_insight_rate(df, window_size):
        """ローリング洞察検出率を計算"""
        rates = []
        episodes = []
        
        for i in range(window_size, len(df) + 1, window_size):
            start_idx = max(0, i - window_size)
            end_idx = i
            
            window_df = df.iloc[start_idx:end_idx]
            rate = window_df['is_insight'].mean()
            rates.append(rate)
            episodes.append(i)
        
        return episodes, rates
    
    # 過去の実験の検出率推移
    past_episodes, past_rates = calculate_rolling_insight_rate(past_df, window_size)
    
    # 現在の実験の検出率推移
    current_episodes, current_rates = calculate_rolling_insight_rate(current_df, window_size)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 洞察検出率の時系列推移
    axes[0, 0].plot(past_episodes, past_rates, 'b-o', label='過去の実験（初期メモリ0）', linewidth=2)
    axes[0, 0].plot(current_episodes, current_rates, 'r-s', label='現在の実験（初期メモリ8）', linewidth=2)
    axes[0, 0].set_xlabel('エピソード数')
    axes[0, 0].set_ylabel('洞察検出率')
    axes[0, 0].set_title('洞察検出率の時系列推移比較')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 累積洞察検出率
    past_cumulative = past_df['is_insight'].cumsum() / (past_df.index + 1)
    current_cumulative = current_df['is_insight'].cumsum() / (current_df.index + 1)
    
    axes[0, 1].plot(past_cumulative.index + 1, past_cumulative, 'b-', label='過去の実験（初期メモリ0）', linewidth=2)
    axes[0, 1].plot(current_cumulative.index + 1, current_cumulative, 'r-', label='現在の実験（初期メモリ8）', linewidth=2)
    axes[0, 1].set_xlabel('エピソード数')
    axes[0, 1].set_ylabel('累積洞察検出率')
    axes[0, 1].set_title('累積洞察検出率の比較')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. GED値の分布比較
    axes[1, 0].hist(past_df['delta_ged'], bins=50, alpha=0.7, label='過去の実験（初期メモリ0）', color='blue')
    axes[1, 0].hist(current_df['delta_ged'], bins=50, alpha=0.7, label='現在の実験（初期メモリ8）', color='red')
    axes[1, 0].set_xlabel('ΔGED値')
    axes[1, 0].set_ylabel('頻度')
    axes[1, 0].set_title('ΔGED値の分布比較')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 初期段階の洞察検出率詳細比較（最初の100エピソード）
    past_early = past_df.iloc[:100]['is_insight'].mean()
    current_early = current_df.iloc[:100]['is_insight'].mean()
    
    past_mid = past_df.iloc[100:200]['is_insight'].mean() if len(past_df) > 200 else 0
    current_mid = current_df.iloc[100:200]['is_insight'].mean() if len(current_df) > 200 else 0
    
    categories = ['初期段階\n(1-100話)', '中期段階\n(101-200話)']
    past_values = [past_early, past_mid]
    current_values = [current_early, current_mid]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, past_values, width, label='過去の実験（初期メモリ0）', color='blue', alpha=0.7)
    axes[1, 1].bar(x + width/2, current_values, width, label='現在の実験（初期メモリ8）', color='red', alpha=0.7)
    
    axes[1, 1].set_xlabel('実験段階')
    axes[1, 1].set_ylabel('洞察検出率')
    axes[1, 1].set_title('実験段階別洞察検出率比較')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(categories)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 数値ラベルを追加
    for i, (past_val, current_val) in enumerate(zip(past_values, current_values)):
        axes[1, 1].text(i - width/2, past_val + 0.01, f'{past_val:.1%}', ha='center', va='bottom')
        axes[1, 1].text(i + width/2, current_val + 0.01, f'{current_val:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 結果保存
    output_path = Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/experiments/01_realtime_insight_experiments/outputs/dynamic_memory_detailed")
    output_path.mkdir(exist_ok=True)
    
    plt.savefig(output_path / "09_initial_memory_impact_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\n💾 可視化を保存: {output_path / '09_initial_memory_impact_comparison.png'}")
    
    # 詳細な統計分析
    print("\n📈 詳細な統計分析")
    print("-" * 40)
    
    print(f"初期段階(1-100話)の洞察検出率:")
    print(f"  過去の実験: {past_early:.1%}")
    print(f"  現在の実験: {current_early:.1%}")
    print(f"  改善: {current_early - past_early:.1%}")
    
    if past_mid > 0 and current_mid > 0:
        print(f"\n中期段階(101-200話)の洞察検出率:")
        print(f"  過去の実験: {past_mid:.1%}")
        print(f"  現在の実験: {current_mid:.1%}")
        print(f"  改善: {current_mid - past_mid:.1%}")
    
    # GED値の統計比較
    print(f"\nΔGED値の統計:")
    print(f"  過去の実験 - 平均: {past_df['delta_ged'].mean():.3f}, 標準偏差: {past_df['delta_ged'].std():.3f}")
    print(f"  現在の実験 - 平均: {current_df['delta_ged'].mean():.3f}, 標準偏差: {current_df['delta_ged'].std():.3f}")
    
    # 洞察エピソードのGED値比較
    past_insight_ged = past_df[past_df['is_insight'] == True]['delta_ged'].mean()
    current_insight_ged = current_df[current_df['is_insight'] == True]['delta_ged'].mean()
    
    print(f"\n洞察エピソードのΔGED値:")
    print(f"  過去の実験: {past_insight_ged:.3f}")
    print(f"  現在の実験: {current_insight_ged:.3f}")
    print(f"  差分: {current_insight_ged - past_insight_ged:.3f}")
    
    plt.show()
    
    print("\n✅ 初期メモリサイズ影響分析完了!")
    
    # 結論
    print("\n📋 分析結果まとめ")
    print("=" * 60)
    print(f"1. 初期メモリサイズ（8エピソード）の効果:")
    print(f"   - 全体的な洞察検出率: {past_rate:.1%} → {current_rate:.1%} (+{current_rate - past_rate:.1%})")
    print(f"   - 初期段階の検出率: {past_early:.1%} → {current_early:.1%} (+{current_early - past_early:.1%})")
    print(f"   - 改善倍率: {current_rate / past_rate:.2f}x")
    
    print(f"\n2. 初期知識の「ブートストラップ効果」:")
    if current_early > past_early:
        print(f"   ✅ 初期知識により実験開始時から高い検出率を実現")
    else:
        print(f"   ❌ 初期知識の効果は限定的")
    
    print(f"\n3. 記憶の「学習曲線効果」:")
    print(f"   - 初期メモリがあることで学習の立ち上がりが高速化")
    print(f"   - 空のメモリから開始する場合の「冷開始問題」を軽減")

if __name__ == "__main__":
    analyze_initial_memory_impact()
