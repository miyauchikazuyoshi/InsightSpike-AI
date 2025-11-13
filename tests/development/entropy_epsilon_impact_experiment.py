#!/usr/bin/env python3
"""
数値実験: epsilon バグの影響を検証

問題のコード:
    p = p + 1e-10  # ❌ 確率分布を破壊
    H = -sum(p * log2(p))

正しいコード:
    H = -sum(p * log2(p + 1e-10))  # ✅ log の中だけ
"""

import numpy as np
import sys

def entropy_buggy(p):
    """バグのある実装"""
    p_modified = p + 1e-10
    return -np.sum(p_modified * np.log2(p_modified))

def entropy_correct(p):
    """正しい実装"""
    return -np.sum(p * np.log2(p + 1e-10))

def entropy_correct_masked(p):
    """正しい実装（マスク版）"""
    mask = p > 0
    if not np.any(mask):
        return 0.0
    return -np.sum(p[mask] * np.log2(p[mask]))

# テストケース
print("=" * 70)
print("epsilon バグの数値的影響分析")
print("=" * 70)

# ケース1: 均等分布（最大エントロピー）
print("\n[ケース1] 均等分布（5要素）")
p1 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
print(f"確率分布: {p1}")
print(f"合計: {p1.sum():.10f}")

h1_buggy = entropy_buggy(p1)
h1_correct = entropy_correct(p1)
h1_masked = entropy_correct_masked(p1)
print(f"\nバグ版エントロピー:   {h1_buggy:.10f} bits")
print(f"正しい版（epsilon）:  {h1_correct:.10f} bits")
print(f"正しい版（mask）:     {h1_masked:.10f} bits")
print(f"理論値 log2(5):       {np.log2(5):.10f} bits")
print(f"誤差（バグ版）:       {abs(h1_buggy - np.log2(5)):.10e} bits")
print(f"誤差（正しい版）:     {abs(h1_correct - np.log2(5)):.10e} bits")
print(f"相対誤差（バグ版）:   {abs(h1_buggy - np.log2(5)) / np.log2(5) * 100:.6f}%")

# ケース2: 偏った分布
print("\n" + "=" * 70)
print("[ケース2] 偏った分布（一つが支配的）")
p2 = np.array([0.9, 0.05, 0.03, 0.015, 0.005])
print(f"確率分布: {p2}")
print(f"合計: {p2.sum():.10f}")

h2_buggy = entropy_buggy(p2)
h2_correct = entropy_correct(p2)
h2_masked = entropy_correct_masked(p2)
h2_theory = -np.sum(p2 * np.log2(p2))  # 理論値
print(f"\nバグ版エントロピー:   {h2_buggy:.10f} bits")
print(f"正しい版（epsilon）:  {h2_correct:.10f} bits")
print(f"正しい版（mask）:     {h2_masked:.10f} bits")
print(f"理論値:               {h2_theory:.10f} bits")
print(f"誤差（バグ版）:       {abs(h2_buggy - h2_theory):.10e} bits")
print(f"誤差（正しい版）:     {abs(h2_correct - h2_theory):.10e} bits")
print(f"相対誤差（バグ版）:   {abs(h2_buggy - h2_theory) / h2_theory * 100:.6f}%")

# ケース3: ゼロを含む分布（Graph Attention で起こりうる）
print("\n" + "=" * 70)
print("[ケース3] ゼロを含む分布（現実的）")
p3 = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
print(f"確率分布: {p3}")
print(f"合計: {p3.sum():.10f}")

h3_buggy = entropy_buggy(p3)
h3_correct = entropy_correct(p3)
h3_masked = entropy_correct_masked(p3)
h3_theory = -np.sum(p3[p3 > 0] * np.log2(p3[p3 > 0]))  # 理論値
print(f"\nバグ版エントロピー:   {h3_buggy:.10f} bits")
print(f"正しい版（epsilon）:  {h3_correct:.10f} bits")
print(f"正しい版（mask）:     {h3_masked:.10f} bits")
print(f"理論値:               {h3_theory:.10f} bits")
print(f"誤差（バグ版）:       {abs(h3_buggy - h3_theory):.10e} bits")
print(f"誤差（正しい版）:     {abs(h3_correct - h3_theory):.10e} bits")
print(f"相対誤差（バグ版）:   {abs(h3_buggy - h3_theory) / h3_theory * 100:.6f}%")

# ケース4: 確率の公理違反チェック
print("\n" + "=" * 70)
print("[ケース4] 確率の公理違反チェック")
p4 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
p4_buggy = p4 + 1e-10
print(f"元の確率分布: {p4}")
print(f"バグ版修正後: {p4_buggy}")
print(f"\n確率の公理チェック:")
print(f"  元の合計: {p4.sum():.15f} (= 1.0 ✅)")
print(f"  バグ版合計: {p4_buggy.sum():.15f} (> 1.0 ❌)")
print(f"  公理違反量: {p4_buggy.sum() - 1.0:.15e}")

# ケース5: 多数の要素（Graph Attention の現実的ケース）
print("\n" + "=" * 70)
print("[ケース5] 多数の要素（50ノード、現実的）")
n = 50
p5 = np.random.dirichlet(np.ones(n))  # ランダムな確率分布
print(f"要素数: {n}")
print(f"分布サンプル: [{p5[0]:.4f}, {p5[1]:.4f}, {p5[2]:.4f}, ...]")
print(f"合計: {p5.sum():.15f}")

h5_buggy = entropy_buggy(p5)
h5_correct = entropy_correct(p5)
h5_masked = entropy_correct_masked(p5)
h5_theory = -np.sum(p5 * np.log2(p5 + 1e-10))
print(f"\nバグ版エントロピー:   {h5_buggy:.10f} bits")
print(f"正しい版（epsilon）:  {h5_correct:.10f} bits")
print(f"正しい版（mask）:     {h5_masked:.10f} bits")
print(f"誤差（バグ版）:       {abs(h5_buggy - h5_correct):.10e} bits")
print(f"相対誤差（バグ版）:   {abs(h5_buggy - h5_correct) / h5_correct * 100:.6f}%")

# 情報利得への影響
print("\n" + "=" * 70)
print("[ケース6] 情報利得への影響（実験で重要）")
p_before = np.array([0.25, 0.25, 0.25, 0.25])
p_after = np.array([0.7, 0.2, 0.05, 0.05])

h_before_buggy = entropy_buggy(p_before)
h_after_buggy = entropy_buggy(p_after)
ig_buggy = h_before_buggy - h_after_buggy

h_before_correct = entropy_correct(p_before)
h_after_correct = entropy_correct(p_after)
ig_correct = h_before_correct - h_after_correct

print(f"Before分布: {p_before} (高エントロピー)")
print(f"After分布:  {p_after} (低エントロピー)")
print(f"\n情報利得（バグ版）:   {ig_buggy:.10f} bits")
print(f"情報利得（正しい版）: {ig_correct:.10f} bits")
print(f"誤差:                 {abs(ig_buggy - ig_correct):.10e} bits")
print(f"相対誤差:             {abs(ig_buggy - ig_correct) / ig_correct * 100:.6f}%")

# 総合評価
print("\n" + "=" * 70)
print("総合評価")
print("=" * 70)
print(f"\nepsilon = 1e-10 のバグによる影響:")
print(f"  - 確率の公理: 必ず違反する（Σp > 1.0）")
print(f"  - 数値的誤差: 平均 ~1e-9 bits 程度")
print(f"  - 相対誤差: 多くの場合 < 0.01%")
print(f"\n結論:")
print(f"  理論的: ❌ 完全に間違い（確率分布ではない）")
print(f"  実用的: △ 数値的影響は小さいが、蓄積する可能性")
print(f"  実験的: ⚠️ 厳密な検証には不適切")
print(f"\n推奨: 必ず修正すべき（理論的正しさのため）")
