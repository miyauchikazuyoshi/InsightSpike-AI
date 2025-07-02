#!/usr/bin/env python3
"""
Step-by-step dataset download for RAG experiment
===============================================
"""

from datasets import load_dataset
from pathlib import Path
import json
import time

def download_and_save_datasets():
    """Download datasets step by step with proper error handling"""
    
    # Create directory
    data_dir = Path("data/huggingface_datasets")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("🌐 Starting dataset downloads...")
    
    datasets_info = {}
    
    # 1. Start with SQuAD (smaller and more reliable)
    print("\n📚 Step 1: Downloading SQuAD dataset (30 samples)...")
    try:
        start_time = time.time()
        squad_dataset = load_dataset("squad", split="validation[:30]")
        download_time = time.time() - start_time
        
        # Save dataset
        squad_path = data_dir / "squad_30"
        squad_dataset.save_to_disk(str(squad_path))
        
        print(f"✅ SQuAD downloaded successfully!")
        print(f"   📊 Samples: {len(squad_dataset)}")
        print(f"   ⏰ Time: {download_time:.1f}s")
        print(f"   💾 Saved to: {squad_path}")
        
        datasets_info["squad"] = {
            "path": str(squad_path),
            "samples": len(squad_dataset),
            "download_time": download_time,
            "status": "success"
        }
        
    except Exception as e:
        print(f"❌ SQuAD download failed: {e}")
        datasets_info["squad"] = {"status": "failed", "error": str(e)}
    
    # 2. Try a smaller Natural Questions sample first
    print("\n📚 Step 2: Downloading Natural Questions (10 samples first)...")
    try:
        start_time = time.time()
        # Start with just 10 samples to test
        nq_small = load_dataset("natural_questions", split="validation[:10]")
        download_time = time.time() - start_time
        
        nq_small_path = data_dir / "natural_questions_10"
        nq_small.save_to_disk(str(nq_small_path))
        
        print(f"✅ Natural Questions (small) downloaded successfully!")
        print(f"   📊 Samples: {len(nq_small)}")
        print(f"   ⏰ Time: {download_time:.1f}s")
        print(f"   💾 Saved to: {nq_small_path}")
        
        datasets_info["natural_questions_small"] = {
            "path": str(nq_small_path),
            "samples": len(nq_small),
            "download_time": download_time,
            "status": "success"
        }
        
        # If small version works, try larger version
        print("\n📚 Step 3: Downloading Natural Questions (50 samples)...")
        start_time = time.time()
        nq_dataset = load_dataset("natural_questions", split="validation[:50]")
        download_time = time.time() - start_time
        
        nq_path = data_dir / "natural_questions_50"
        nq_dataset.save_to_disk(str(nq_path))
        
        print(f"✅ Natural Questions (full) downloaded successfully!")
        print(f"   📊 Samples: {len(nq_dataset)}")
        print(f"   ⏰ Time: {download_time:.1f}s")
        print(f"   💾 Saved to: {nq_path}")
        
        datasets_info["natural_questions"] = {
            "path": str(nq_path),
            "samples": len(nq_dataset),
            "download_time": download_time,
            "status": "success"
        }
        
    except Exception as e:
        print(f"❌ Natural Questions download failed: {e}")
        datasets_info["natural_questions"] = {"status": "failed", "error": str(e)}
    
    # 3. Try MS MARCO as alternative
    print("\n📚 Step 4: Downloading MS MARCO dataset (20 samples)...")
    try:
        start_time = time.time()
        marco_dataset = load_dataset("ms_marco", "v1.1", split="validation[:20]")
        download_time = time.time() - start_time
        
        marco_path = data_dir / "ms_marco_20"
        marco_dataset.save_to_disk(str(marco_path))
        
        print(f"✅ MS MARCO downloaded successfully!")
        print(f"   📊 Samples: {len(marco_dataset)}")
        print(f"   ⏰ Time: {download_time:.1f}s")
        print(f"   💾 Saved to: {marco_path}")
        
        datasets_info["ms_marco"] = {
            "path": str(marco_path),
            "samples": len(marco_dataset),
            "download_time": download_time,
            "status": "success"
        }
        
    except Exception as e:
        print(f"❌ MS MARCO download failed: {e}")
        print("🔄 Trying alternative: XNLI...")
        
        try:
            start_time = time.time()
            xnli_dataset = load_dataset("xnli", split="validation[:20]")
            download_time = time.time() - start_time
            
            xnli_path = data_dir / "xnli_20"
            xnli_dataset.save_to_disk(str(xnli_path))
            
            print(f"✅ XNLI downloaded successfully!")
            print(f"   📊 Samples: {len(xnli_dataset)}")
            print(f"   ⏰ Time: {download_time:.1f}s")
            print(f"   💾 Saved to: {xnli_path}")
            
            datasets_info["xnli"] = {
                "path": str(xnli_path),
                "samples": len(xnli_dataset),
                "download_time": download_time,
                "status": "success"
            }
            
        except Exception as e2:
            print(f"❌ XNLI also failed: {e2}")
            datasets_info["alternatives"] = {"status": "failed", "error": str(e2)}
    
    # Save download info
    info_path = data_dir / "download_info.json"
    with open(info_path, 'w') as f:
        json.dump(datasets_info, f, indent=2)
    
    # Summary
    print(f"\n📊 Download Summary:")
    print("=" * 50)
    successful_downloads = [k for k, v in datasets_info.items() if v.get("status") == "success"]
    total_samples = sum(v.get("samples", 0) for v in datasets_info.values() if v.get("status") == "success")
    
    print(f"✅ Successful downloads: {len(successful_downloads)}")
    print(f"📝 Total samples: {total_samples}")
    print(f"💾 Saved to: {data_dir}")
    print(f"📋 Download info: {info_path}")
    
    for dataset_name in successful_downloads:
        info = datasets_info[dataset_name]
        print(f"   📚 {dataset_name}: {info['samples']} samples ({info['download_time']:.1f}s)")
    
    return datasets_info

if __name__ == "__main__":
    download_and_save_datasets()