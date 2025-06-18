#!/usr/bin/env python3
"""
ベクトルからテキストへの近似逆変換実験
=====================================

埋め込みベクトルから元のテキストやセマンティックに近いテキストを
近似的に復元する手法を実験します。
"""

import numpy as np
import json
from typing import List, Tuple, Optional
from pathlib import Path
import sys
import os

# InsightSpike-AIのパスを追加
sys.path.append(str(Path(__file__).parent.parent / "src"))

from insightspike.utils.embedder import get_model


class VectorToTextApproximator:
    """ベクトルからテキストへの近似変換器"""
    
    def __init__(self):
        self.model = get_model()
        self.reference_texts = []
        self.reference_vectors = []
        
    def build_reference_database(self, texts: List[str]):
        """参照テキストデータベースを構築"""
        print(f"📚 参照データベース構築中... ({len(texts)}件)")
        self.reference_texts = texts
        self.reference_vectors = self.model.encode(texts)
        print(f"✅ 参照データベース構築完了")
        
    def find_nearest_text(self, target_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """最近傍テキストを検索"""
        if len(self.reference_vectors) == 0:
            return []
            
        # コサイン類似度計算
        similarities = self.reference_vectors @ target_vector
        
        # Top-K取得
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((self.reference_texts[idx], similarities[idx]))
            
        return results
    
    def interpolate_meanings(self, vector1: np.ndarray, vector2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """2つのベクトル間を補間して新しい意味を生成"""
        interpolated = alpha * vector1 + (1 - alpha) * vector2
        # 正規化
        return interpolated / np.linalg.norm(interpolated)
    
    def semantic_arithmetic(self, base_vector: np.ndarray, add_vector: np.ndarray, 
                          subtract_vector: Optional[np.ndarray] = None) -> np.ndarray:
        """セマンティック演算（例: King - Man + Woman = Queen的な）"""
        result = base_vector + add_vector
        if subtract_vector is not None:
            result = result - subtract_vector
        # 正規化
        return result / np.linalg.norm(result)


def run_vector_to_text_experiment():
    """ベクトル→テキスト変換実験を実行"""
    
    print("🧪 ベクトル→テキスト近似変換実験")
    print("=" * 60)
    
    approximator = VectorToTextApproximator()
    
    # 1. InsightSpike-AIの実際のエピソードを参照データベースとして使用
    print("\n📖 1. 参照データベースの準備")
    
    # CSVから実際のエピソードテキストを読み込み
    try:
        import pandas as pd
        episodes_df = pd.read_csv("outputs/csv_summaries/input_episodes.csv")
        reference_texts = episodes_df['episode_text'].tolist()[:100]  # 最初の100件を使用
        print(f"✅ InsightSpike-AIエピソード {len(reference_texts)}件を参照データベースに追加")
    except Exception as e:
        print(f"❌ CSV読み込みエラー: {e}")
        # フォールバック: 基本的なテキスト集合
        reference_texts = [
            "AI can revolutionize healthcare diagnostics",
            "Machine learning models require high-quality data",
            "Deep learning excels at pattern recognition",
            "Natural language processing enables human-computer interaction",
            "Computer vision systems analyze medical images",
            "基礎概念の学習が重要です",
            "概念間の関係性を理解する",
            "知識の体系化と統合",
            "専門知識の獲得プロセス",
            "継続的学習による改善"
        ]
    
    approximator.build_reference_database(reference_texts)
    
    # 2. テストケース: 元のテキストからベクトルを生成し、逆変換を試す
    print("\n🎯 2. 逆変換テスト")
    
    test_cases = [
        "AI can revolutionize healthcare diagnostics",
        "基礎概念の学習が重要です",
        "Machine learning models require data"
    ]
    
    for i, original_text in enumerate(test_cases, 1):
        print(f"\n--- テストケース {i} ---")
        print(f"🔤 元のテキスト: \"{original_text}\"")
        
        # テキスト → ベクトル
        vector = approximator.model.encode([original_text])[0]
        print(f"🔢 ベクトル化: {vector.shape} 次元")
        
        # ベクトル → 近似テキスト
        nearest_texts = approximator.find_nearest_text(vector, top_k=3)
        
        print(f"🔍 近似復元結果:")
        for j, (text, similarity) in enumerate(nearest_texts, 1):
            print(f"  {j}. (類似度: {similarity:.4f}) \"{text}\"")
    
    # 3. セマンティック演算実験
    print(f"\n🧮 3. セマンティック演算実験")
    
    # ベクトル演算の例
    healthcare_vec = approximator.model.encode(["healthcare diagnostics"])[0]
    ai_vec = approximator.model.encode(["artificial intelligence"])[0]
    learning_vec = approximator.model.encode(["machine learning"])[0]
    
    # AI + Healthcare の意味
    ai_healthcare = approximator.semantic_arithmetic(ai_vec, healthcare_vec)
    print(f"\n🔬 AI + Healthcare の意味:")
    nearest = approximator.find_nearest_text(ai_healthcare, top_k=3)
    for j, (text, sim) in enumerate(nearest, 1):
        print(f"  {j}. (類似度: {sim:.4f}) \"{text}\"")
    
    # 4. 補間実験
    print(f"\n🔀 4. 意味補間実験")
    
    ai_vec = approximator.model.encode(["artificial intelligence"])[0]
    health_vec = approximator.model.encode(["healthcare"])[0]
    
    for alpha in [0.2, 0.5, 0.8]:
        interpolated = approximator.interpolate_meanings(ai_vec, health_vec, alpha)
        print(f"\nAI({alpha:.1f}) + Healthcare({1-alpha:.1f}):")
        nearest = approximator.find_nearest_text(interpolated, top_k=2)
        for j, (text, sim) in enumerate(nearest, 1):
            print(f"  {j}. (類似度: {sim:.4f}) \"{text}\"")
    
    # 5. 結論
    print(f"\n📊 5. 実験結論")
    print("=" * 40)
    print("✅ 可能な近似変換:")
    print("  • 最近傍検索による意味的に近いテキストの発見")
    print("  • セマンティック演算による新しい概念の生成")
    print("  • ベクトル補間による意味の段階的変化")
    print()
    print("❌ 不可能な完全逆変換:")
    print("  • 元の正確なテキストの復元")
    print("  • 語彙や文法の完全な復元")
    print("  • 固有名詞や数値の正確な復元")
    print()
    print("💡 InsightSpike-AIでの応用:")
    print("  • 洞察の概念的説明生成")
    print("  • 類似概念の発見と提示")
    print("  • 概念間の関係性の視覚化")


if __name__ == "__main__":
    run_vector_to_text_experiment()
