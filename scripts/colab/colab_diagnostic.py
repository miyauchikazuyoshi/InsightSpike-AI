#!/usr/bin/env python3
"""
Colab環境用の診断・修復スクリプト
Google Colabでの動作問題を診断し、自動修復を試みます。
"""

import sys
import os
import subprocess
from pathlib import Path

def check_environment():
    """環境の診断"""
    print("🔍 Environment Diagnosis")
    print("=" * 50)
    
    # Python version
    print(f"Python version: {sys.version}")
    
    # CUDA availability
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not found")
    
    # Critical dependencies
    dependencies = [
        "faiss", "sentence_transformers", "transformers", 
        "torch_geometric", "networkx", "sklearn", "matplotlib"
    ]
    
    missing = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep}")
            missing.append(dep)
    
    return missing

def fix_missing_dependencies(missing):
    """不足している依存関係を修復"""
    if not missing:
        print("✅ All dependencies are available!")
        return
    
    print(f"\n🔧 Fixing {len(missing)} missing dependencies...")
    
    # 依存関係のマッピング
    package_map = {
        "faiss": "faiss-gpu",
        "sklearn": "scikit-learn",
        "torch_geometric": "torch-geometric",
    }
    
    for dep in missing:
        package = package_map.get(dep, dep)
        print(f"Installing {package}...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-q", package
            ], check=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def check_data_files():
    """データファイルの存在確認"""
    print("\n📊 Data Files Check")
    print("=" * 50)
    
    data_paths = [
        "data/raw/test_sentences.txt",
        "data/processed/episodes",
        "data/embedding/",
        "data/graph_pyg.pt"
    ]
    
    missing_data = []
    for path in data_paths:
        if Path(path).exists():
            print(f"✅ {path}")
        else:
            print(f"❌ {path}")
            missing_data.append(path)
    
    return missing_data

def create_test_data():
    """テスト用データの作成"""
    print("\n🔧 Creating test data...")
    
    # テスト用の文章データを作成
    test_sentences = [
        "Artificial intelligence is the simulation of human intelligence in machines.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information.",
        "Quantum computing uses quantum mechanics for computation.",
        "Quantum entanglement is a phenomenon where particles become correlated.",
        "The universe contains billions of galaxies.",
        "Stars are formed from clouds of gas and dust.",
        "Black holes are regions where gravity is so strong that nothing can escape."
    ]
    
    # ディレクトリ作成
    os.makedirs("data/raw", exist_ok=True)
    
    # テストファイル作成
    with open("data/raw/test_sentences.txt", "w", encoding="utf-8") as f:
        for sentence in test_sentences:
            f.write(sentence + "\n")
    
    print("✅ Test data created in data/raw/test_sentences.txt")

def run_quick_test():
    """クイックテスト実行"""
    print("\n🚀 Running Quick Test")
    print("=" * 50)
    
    try:
        # PYTHONPATHを設定
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        
        # 簡単なインポートテスト
        result = subprocess.run([
            sys.executable, "-c", 
            "import sys; sys.path.append('src'); from insightspike.config import Config; print('✅ InsightSpike import successful')"
        ], env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"❌ Import test failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Quick test failed: {e}")

def main():
    """メイン関数"""
    print("🧠 InsightSpike-AI Colab Diagnostic Tool")
    print("=" * 60)
    
    # 1. 環境診断
    missing = check_environment()
    
    # 2. 不足依存関係の修復
    if missing:
        fix_missing_dependencies(missing)
    
    # 3. データファイル確認
    missing_data = check_data_files()
    
    # 4. テストデータ作成
    if "data/raw/test_sentences.txt" in missing_data:
        create_test_data()
    
    # 5. クイックテスト
    run_quick_test()
    
    print("\n🎉 Diagnostic complete!")
    print("\nNext steps:")
    print("1. If data is missing: run 'python scripts/databake.py'")
    print("2. Build memory: 'PYTHONPATH=src python -m insightspike.cli embed --path data/raw/test_sentences.txt'")
    print("3. Run PoC: 'PYTHONPATH=src python scripts/run_poc.py'")

if __name__ == "__main__":
    main()
