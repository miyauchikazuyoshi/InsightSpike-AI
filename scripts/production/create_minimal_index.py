#!/usr/bin/env python3
"""
Quick FAISS Index Creator for InsightSpike-AI
===========================================

Creates a minimal FAISS index for testing purposes.
"""

import os
import sys
import numpy as np
from pathlib import Path

def create_minimal_faiss_index():
    print("🔧 Creating minimal FAISS index...")
    
    # CI環境の検出を強化
    is_ci = any([
        os.getenv('CI') == 'true',
        os.getenv('GITHUB_ACTIONS') == 'true',
        os.getenv('RUNNER_OS'),
        'runner' in os.getcwd().lower(),
        'github' in os.getcwd().lower()
    ])
    
    # プロジェクトルートの正しい設定
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # scripts/production -> scripts -> root
    data_dir = project_root / 'data'
    
    # データディレクトリを確実に作成
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if is_ci:
        print("🤖 CI環境を検出 - ダミーインデックスを作成...")
        # ダミーファイルを作成
        dummy_index_path = data_dir / 'index.faiss'
        dummy_index_path.write_bytes(b'dummy_faiss_index_for_ci_testing')
        print(f"✅ ダミーインデックスを作成: {dummy_index_path}")
        return True
    
    # 本物のFAISSインデックス作成（ローカル環境のみ）
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        
        print("📝 Generating embeddings...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        test_sentences = [
            "The aurora borealis is caused by charged particles from the sun.",
            "Quantum entanglement is a phenomenon in quantum physics.",
            "Artificial intelligence uses machine learning algorithms."
        ]
        
        embeddings = model.encode(test_sentences)
        
        # FAISSインデックス作成
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        # インデックス保存
        index_path = data_dir / 'index.faiss'
        faiss.write_index(index, str(index_path))
        
        print(f"✅ FAISSインデックスを作成: {index_path}")
        return True
        
    except Exception as e:
        print(f"❌ FAISSインデックス作成に失敗: {e}")
        return False

if __name__ == "__main__":
    success = create_minimal_faiss_index()
    sys.exit(0 if success else 1)
