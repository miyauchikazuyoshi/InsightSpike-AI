#!/usr/bin/env python3
"""
1000問レベル超大規模データセット収集
================================

複数のHuggingFaceデータセットから1000問を効率的に収集して
InsightSpike-AIの真の実力を評価する
"""

from datasets import load_dataset
from pathlib import Path
import json
import time
from typing import Dict, List, Any
import concurrent.futures
from threading import Lock

def download_dataset_batch(dataset_config):
    """データセットを並列ダウンロード"""
    name, config, split, samples = dataset_config
    
    try:
        print(f"📚 Downloading {name} ({samples} samples)...")
        start_time = time.time()
        
        if config:
            dataset = load_dataset(name, config, split=split)
        else:
            dataset = load_dataset(name, split=split)
        
        download_time = time.time() - start_time
        
        return {
            "name": name,
            "dataset": dataset,
            "samples": len(dataset),
            "download_time": download_time,
            "status": "success"
        }
    except Exception as e:
        print(f"❌ {name} failed: {e}")
        return {
            "name": name,
            "status": "failed",
            "error": str(e)
        }

def download_1000_questions():
    """1000問の超大規模データセット収集"""
    
    # データディレクトリ作成
    data_dir = Path("data/mega_huggingface_datasets")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("🌐 1000問レベル超大規模データセット収集開始!")
    print("=" * 70)
    
    # データセット設定（並列ダウンロード用）
    dataset_configs = [
        # メインデータセット（大容量）
        ("squad", None, "validation[:300]", 300),
        ("squad", None, "train[:200]", 200),
        ("ms_marco", "v1.1", "validation[:150]", 150),
        
        # 多様なQAデータセット
        ("natural_questions", None, "validation[:100]", 100),  # 再チャレンジ
        ("coqa", None, "validation[:80]", 80),
        ("drop", None, "validation[:50]", 50),
        
        # 新しいデータセット
        ("hotpot_qa", "fullwiki", "validation[:60]", 60),
        ("xnli", None, "validation[:50]", 50),
        ("boolq", None, "validation[:50]", 50),
        
        # 軽量データセット（確実性重視）
        ("piqa", None, "validation[:40]", 40),
        ("winogrande", "winogrande_xl", "validation[:30]", 30),
        ("hellaswag", None, "validation[:30]", 30),
        ("arc", "ARC-Easy", "validation[:30]", 30),
        ("arc", "ARC-Challenge", "validation[:20]", 20),
        ("commonsense_qa", None, "validation[:20]", 20),
    ]
    
    print(f"🎯 目標: {sum(config[3] for config in dataset_configs)} 問")
    print(f"📊 データセット数: {len(dataset_configs)} 種類")
    
    # 並列ダウンロード実行
    print("\n🚀 並列ダウンロード開始...")
    
    datasets_info = {}
    total_samples = 0
    lock = Lock()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # 全データセットを並列処理で開始
        future_to_config = {
            executor.submit(download_dataset_batch, config): config 
            for config in dataset_configs
        }
        
        for future in concurrent.futures.as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                
                with lock:
                    dataset_name = result["name"]
                    if result["status"] == "success":
                        # データセット保存
                        save_path = data_dir / f"{dataset_name}_{result['samples']}"
                        result["dataset"].save_to_disk(str(save_path))
                        
                        datasets_info[f"{dataset_name}_{result['samples']}"] = {
                            "path": str(save_path),
                            "samples": result["samples"],
                            "download_time": result["download_time"],
                            "status": "success"
                        }
                        
                        total_samples += result["samples"]
                        print(f"✅ {dataset_name}: {result['samples']} samples ({result['download_time']:.1f}s)")
                    else:
                        datasets_info[f"{dataset_name}_failed"] = {
                            "status": "failed",
                            "error": result["error"]
                        }
                        
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
    
    # ダウンロード情報保存
    info_path = data_dir / "mega_download_info.json"
    with open(info_path, 'w') as f:
        json.dump(datasets_info, f, indent=2)
    
    # サマリ表示
    print(f"\n📊 超大規模データセット収集完了!")
    print("=" * 70)
    successful_downloads = [k for k, v in datasets_info.items() if v.get("status") == "success"]
    
    print(f"✅ 成功: {len(successful_downloads)} datasets")
    print(f"📝 総サンプル数: {total_samples}")
    print(f"💾 保存先: {data_dir}")
    print(f"📋 ダウンロード情報: {info_path}")
    
    # 詳細リスト
    print(f"\n📚 ダウンロードされたデータセット:")
    for dataset_name in successful_downloads:
        info = datasets_info[dataset_name]
        print(f"   📖 {dataset_name}: {info['samples']} samples ({info['download_time']:.1f}s)")
    
    if total_samples >= 1000:
        print(f"\n🎯 目標達成! {total_samples}問で超大規模RAG実験準備完了!")
    elif total_samples >= 500:
        print(f"\n⚡ 部分達成! {total_samples}問で大規模RAG実験可能!")
    else:
        print(f"\n⚠️ サンプル数不足: {total_samples}問（目標: 1000問以上）")
    
    # 失敗したデータセット情報
    failed_downloads = [k for k, v in datasets_info.items() if v.get("status") == "failed"]
    if failed_downloads:
        print(f"\n❌ 失敗したデータセット ({len(failed_downloads)}):")
        for dataset_name in failed_downloads:
            error = datasets_info[dataset_name].get("error", "Unknown error")
            print(f"   ❌ {dataset_name}: {error}")
    
    return datasets_info, total_samples

if __name__ == "__main__":
    download_1000_questions()