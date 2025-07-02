#!/usr/bin/env python3
"""
ローカルでColab実験をテストするためのスクリプト
"""

import subprocess
import sys
import os

def test_colab_experiment_locally():
    """Colab実験をローカル環境でテスト"""
    
    print("🧪 Colab実験のローカルテストを開始します...")
    
    # 1. 現在の環境を確認
    print("\n1️⃣ 現在の環境確認:")
    print(f"Python: {sys.version}")
    print(f"作業ディレクトリ: {os.getcwd()}")
    
    # 2. 必要なパッケージの確認
    print("\n2️⃣ 必要なパッケージの確認:")
    required_packages = [
        "numpy",
        "torch", 
        "transformers",
        "sentence-transformers",
        "faiss-cpu",
        "matplotlib",
        "seaborn",
        "pandas",
        "scikit-learn"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}: インストール済み")
        except ImportError:
            print(f"❌ {package}: 未インストール")
            missing_packages.append(package)
    
    # 3. InsightSpike-AIモジュールの確認
    print("\n3️⃣ InsightSpike-AIモジュールの確認:")
    try:
        import insightspike
        print("✅ InsightSpike-AI: インポート成功")
        print(f"   パス: {insightspike.__file__}")
    except ImportError:
        print("❌ InsightSpike-AI: インポート失敗")
        print("   PYTHONPATH設定が必要かもしれません")
        
    # 4. Colab実験ノートブックの存在確認
    print("\n4️⃣ Colab実験ノートブックの確認:")
    notebook_path = "experiments/colab_experiments/foundational_experiment/foundational_experiment_colab.ipynb"
    if os.path.exists(notebook_path):
        print(f"✅ ノートブック存在: {notebook_path}")
    else:
        print(f"❌ ノートブック不在: {notebook_path}")
        
    # 5. テスト実行の提案
    print("\n5️⃣ テスト実行方法:")
    print("以下のコマンドでJupyterを起動してノートブックを実行できます:")
    print(f"jupyter notebook {notebook_path}")
    
    if missing_packages:
        print("\n⚠️ 不足パッケージのインストール:")
        print(f"pip install {' '.join(missing_packages)}")
        
    # 6. 簡単な動作確認
    print("\n6️⃣ 簡単な動作確認:")
    try:
        from src.insightspike.core.system import InsightSpikeSystem
        system = InsightSpikeSystem()
        print("✅ InsightSpikeSystem: 初期化成功")
    except Exception as e:
        print(f"❌ InsightSpikeSystem: 初期化失敗 - {e}")

if __name__ == "__main__":
    test_colab_experiment_locally()