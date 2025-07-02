#!/usr/bin/env python3
"""
大規模データセット収集スクリプト
===============================

複数のHuggingFaceデータセットから100-200問を収集して
InsightSpike-AIの真の性能を評価する
"""

from datasets import load_dataset
from pathlib import Path
import json
import time
from typing import Dict, List, Any

def download_large_datasets():
    """複数データセットから大規模サンプルを収集"""
    
    # データディレクトリ作成
    data_dir = Path("data/large_huggingface_datasets")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("🌐 大規模データセット収集開始...")
    print("=" * 60)
    
    datasets_info = {}
    total_samples = 0
    
    # 1. SQuAD（より多く）
    print("\n📚 Step 1: SQuAD (100 samples)...")
    try:
        start_time = time.time()
        squad_dataset = load_dataset("squad", split="validation[:100]")
        download_time = time.time() - start_time
        
        squad_path = data_dir / "squad_100"
        squad_dataset.save_to_disk(str(squad_path))
        
        print(f"✅ SQuAD: {len(squad_dataset)} samples ({download_time:.1f}s)")
        
        datasets_info["squad_100"] = {
            "path": str(squad_path),
            "samples": len(squad_dataset),
            "download_time": download_time,
            "status": "success"
        }
        total_samples += len(squad_dataset)
        
    except Exception as e:
        print(f"❌ SQuAD failed: {e}")
        datasets_info["squad_100"] = {"status": "failed", "error": str(e)}
    
    # 2. MS MARCO（より多く）
    print("\n🔍 Step 2: MS MARCO (50 samples)...")
    try:
        start_time = time.time()
        marco_dataset = load_dataset("ms_marco", "v1.1", split="validation[:50]")
        download_time = time.time() - start_time
        
        marco_path = data_dir / "ms_marco_50"
        marco_dataset.save_to_disk(str(marco_path))
        
        print(f"✅ MS MARCO: {len(marco_dataset)} samples ({download_time:.1f}s)")
        
        datasets_info["ms_marco_50"] = {
            "path": str(marco_path),
            "samples": len(marco_dataset),
            "download_time": download_time,
            "status": "success"
        }
        total_samples += len(marco_dataset)
        
    except Exception as e:
        print(f"❌ MS MARCO failed: {e}")
        datasets_info["ms_marco_50"] = {"status": "failed", "error": str(e)}
    
    # 3. CoQA（対話型QA）
    print("\n💬 Step 3: CoQA (30 samples)...")
    try:
        start_time = time.time()
        coqa_dataset = load_dataset("coqa", split="validation[:30]")
        download_time = time.time() - start_time
        
        coqa_path = data_dir / "coqa_30"
        coqa_dataset.save_to_disk(str(coqa_path))
        
        print(f"✅ CoQA: {len(coqa_dataset)} samples ({download_time:.1f}s)")
        
        datasets_info["coqa_30"] = {
            "path": str(coqa_path),
            "samples": len(coqa_dataset),
            "download_time": download_time,
            "status": "success"
        }
        total_samples += len(coqa_dataset)
        
    except Exception as e:
        print(f"❌ CoQA failed: {e}")
        datasets_info["coqa_30"] = {"status": "failed", "error": str(e)}
    
    # 4. QuAC（文脈QA）
    print("\n📖 Step 4: QuAC (20 samples)...")
    try:
        start_time = time.time()
        quac_dataset = load_dataset("quac", split="validation[:20]")
        download_time = time.time() - start_time
        
        quac_path = data_dir / "quac_20"
        quac_dataset.save_to_disk(str(quac_path))
        
        print(f"✅ QuAC: {len(quac_dataset)} samples ({download_time:.1f}s)")
        
        datasets_info["quac_20"] = {
            "path": str(quac_path),
            "samples": len(quac_dataset),
            "download_time": download_time,
            "status": "success"
        }
        total_samples += len(quac_dataset)
        
    except Exception as e:
        print(f"❌ QuAC failed: {e}")
        datasets_info["quac_20"] = {"status": "failed", "error": str(e)}
    
    # 5. DROP（数値推論）
    print("\n🔢 Step 5: DROP (20 samples)...")
    try:
        start_time = time.time()
        drop_dataset = load_dataset("drop", split="validation[:20]")
        download_time = time.time() - start_time
        
        drop_path = data_dir / "drop_20"
        drop_dataset.save_to_disk(str(drop_path))
        
        print(f"✅ DROP: {len(drop_dataset)} samples ({download_time:.1f}s)")
        
        datasets_info["drop_20"] = {
            "path": str(drop_path),
            "samples": len(drop_dataset),
            "download_time": download_time,
            "status": "success"
        }
        total_samples += len(drop_dataset)
        
    except Exception as e:
        print(f"❌ DROP failed: {e}")
        datasets_info["drop_20"] = {"status": "failed", "error": str(e)}
    
    # ダウンロード情報保存
    info_path = data_dir / "large_download_info.json"
    with open(info_path, 'w') as f:
        json.dump(datasets_info, f, indent=2)
    
    # サマリ表示
    print(f"\n📊 大規模データセット収集完了!")
    print("=" * 60)
    successful = [k for k, v in datasets_info.items() if v.get("status") == "success"]
    print(f"✅ 成功: {len(successful)} datasets")
    print(f"📝 総サンプル数: {total_samples}")
    print(f"💾 保存先: {data_dir}")
    
    for dataset_name in successful:
        info = datasets_info[dataset_name]
        print(f"   📚 {dataset_name}: {info['samples']} samples ({info['download_time']:.1f}s)")
    
    if total_samples >= 100:
        print(f"\n🎯 目標達成! {total_samples}問で大規模RAG実験準備完了!")
    else:
        print(f"\n⚠️ サンプル数不足: {total_samples}問（目標: 100問以上）")
    
    return datasets_info, total_samples

if __name__ == "__main__":
    download_large_datasets()