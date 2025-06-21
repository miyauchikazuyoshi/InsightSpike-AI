"""
ベンチマークデータセット管理ユーティリティ

各実験で使用するデータセットの統一管理・ダウンロード・前処理機能
"""

import os
import json
import hashlib
import requests
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)


class DatasetManager:
    """統一データセット管理クラス"""
    
    def __init__(self, data_dir: str = "./data/benchmark_datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.data_dir / "datasets_metadata.json"
        self.datasets = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """データセットメタデータの読み込み"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """データセットメタデータの保存"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.datasets, f, indent=2, ensure_ascii=False)
    
    def register_dataset(self, name: str, source: str, description: str, 
                        size: int, checksum: str = None) -> bool:
        """データセットの登録"""
        self.datasets[name] = {
            "source": source,
            "description": description,
            "size": size,
            "checksum": checksum,
            "downloaded": False,
            "processed": False
        }
        self._save_metadata()
        logger.info(f"Dataset '{name}' registered successfully")
        return True
    
    def download_dataset(self, name: str, force_redownload: bool = False) -> bool:
        """データセットのダウンロード"""
        if name not in self.datasets:
            logger.error(f"Dataset '{name}' not registered")
            return False
        
        dataset_info = self.datasets[name]
        local_path = self.data_dir / name
        
        if local_path.exists() and not force_redownload:
            logger.info(f"Dataset '{name}' already downloaded")
            return True
        
        try:
            response = requests.get(dataset_info["source"], stream=True)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # チェックサム検証
            if dataset_info.get("checksum"):
                if not self._verify_checksum(local_path, dataset_info["checksum"]):
                    logger.error(f"Checksum verification failed for '{name}'")
                    local_path.unlink()
                    return False
            
            self.datasets[name]["downloaded"] = True
            self._save_metadata()
            logger.info(f"Dataset '{name}' downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download dataset '{name}': {e}")
            return False
    
    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """ファイルのチェックサム検証"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest() == expected_checksum
    
    def get_dataset_path(self, name: str) -> Optional[Path]:
        """データセットのローカルパス取得"""
        if name in self.datasets and self.datasets[name]["downloaded"]:
            return self.data_dir / name
        return None


class BenchmarkLoader:
    """標準ベンチマークデータセット読み込みクラス"""
    
    def __init__(self):
        self.supported_datasets = {
            "20newsgroups": self._load_20newsgroups,
            "ms_marco": self._load_ms_marco,
            "natural_questions": self._load_natural_questions,
            "hotpot_qa": self._load_hotpot_qa,
            "beir": self._load_beir_dataset,
            "maze_benchmark": self._generate_maze_benchmark
        }
    
    def load_benchmark(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        """ベンチマークデータセットの読み込み"""
        if dataset_name not in self.supported_datasets:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        logger.info(f"Loading benchmark dataset: {dataset_name}")
        return self.supported_datasets[dataset_name](**kwargs)
    
    def _load_20newsgroups(self, subset: str = "all") -> Dict[str, Any]:
        """20 Newsgroupsデータセット読み込み"""
        data = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'))
        return {
            "documents": data.data,
            "labels": data.target,
            "target_names": data.target_names,
            "type": "classification",
            "size": len(data.data)
        }
    
    def _load_ms_marco(self, split: str = "train") -> Dict[str, Any]:
        """MS MARCO QAデータセット読み込み"""
        try:
            dataset = load_dataset("ms_marco", "v1.1", split=split)
            return {
                "queries": dataset["query"],
                "passages": dataset["passages"],
                "answers": dataset["answers"],
                "type": "qa",
                "size": len(dataset)
            }
        except Exception as e:
            logger.warning(f"Failed to load MS MARCO: {e}")
            return self._generate_synthetic_qa_data(1000)
    
    def _load_natural_questions(self, split: str = "train") -> Dict[str, Any]:
        """Natural Questionsデータセット読み込み"""
        try:
            dataset = load_dataset("natural_questions", split=split)
            return {
                "questions": [q["text"] for q in dataset["question"]],
                "contexts": [c["text"] for c in dataset["document"]["html"]],
                "type": "qa",
                "size": len(dataset)
            }
        except Exception as e:
            logger.warning(f"Failed to load Natural Questions: {e}")
            return self._generate_synthetic_qa_data(1000)
    
    def _load_hotpot_qa(self, split: str = "train") -> Dict[str, Any]:
        """HotpotQAデータセット読み込み"""
        try:
            dataset = load_dataset("hotpot_qa", "fullwiki", split=split)
            return {
                "questions": dataset["question"],
                "contexts": dataset["context"],
                "answers": dataset["answer"],
                "type": "multi_hop_qa",
                "size": len(dataset)
            }
        except Exception as e:
            logger.warning(f"Failed to load HotpotQA: {e}")
            return self._generate_synthetic_qa_data(1000)
    
    def _load_beir_dataset(self, dataset_name: str = "scifact") -> Dict[str, Any]:
        """BEIRベンチマークデータセット読み込み"""
        try:
            from beir import util
            from beir.datasets.data_loader import GenericDataLoader
            
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
            data_path = util.download_and_unzip(url, "datasets")
            corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
            
            return {
                "corpus": corpus,
                "queries": queries,
                "qrels": qrels,
                "type": "retrieval",
                "size": len(corpus)
            }
        except Exception as e:
            logger.warning(f"Failed to load BEIR dataset: {e}")
            return self._generate_synthetic_retrieval_data(1000)
    
    def _generate_maze_benchmark(self, grid_sizes: List[int] = [10, 20, 50, 100],
                                complexity_levels: List[str] = ["simple", "medium", "complex"]) -> Dict[str, Any]:
        """迷路ベンチマークデータ生成"""
        mazes = []
        
        for size in grid_sizes:
            for complexity in complexity_levels:
                maze_data = self._generate_single_maze(size, complexity)
                mazes.append(maze_data)
        
        return {
            "mazes": mazes,
            "type": "pathfinding",
            "size": len(mazes),
            "evaluation_metrics": ["path_length", "computation_time", "memory_usage", "optimality"]
        }
    
    def _generate_single_maze(self, size: int, complexity: str) -> Dict[str, Any]:
        """単一迷路データ生成"""
        np.random.seed(42)  # 再現性のため
        
        # 迷路生成（0: 通路, 1: 壁）
        maze = np.zeros((size, size), dtype=int)
        
        # 複雑度に応じた壁の配置
        wall_density = {"simple": 0.2, "medium": 0.35, "complex": 0.5}[complexity]
        
        for i in range(size):
            for j in range(size):
                if np.random.random() < wall_density:
                    maze[i, j] = 1
        
        # スタート・ゴール地点確保
        maze[0, 0] = 0  # スタート
        maze[size-1, size-1] = 0  # ゴール
        
        return {
            "grid": maze.tolist(),
            "size": size,
            "complexity": complexity,
            "start": (0, 0),
            "goal": (size-1, size-1),
            "wall_density": wall_density
        }
    
    def _generate_synthetic_qa_data(self, size: int) -> Dict[str, Any]:
        """合成QAデータ生成（フォールバック用）"""
        questions = [f"What is the answer to question {i}?" for i in range(size)]
        contexts = [f"This is context document {i} containing relevant information." for i in range(size)]
        answers = [f"Answer {i}" for i in range(size)]
        
        return {
            "questions": questions,
            "contexts": contexts,
            "answers": answers,
            "type": "synthetic_qa",
            "size": size
        }
    
    def _generate_synthetic_retrieval_data(self, size: int) -> Dict[str, Any]:
        """合成検索データ生成（フォールバック用）"""
        corpus = {f"doc_{i}": f"Document {i} content for retrieval testing." for i in range(size)}
        queries = {f"query_{i}": f"Query {i} for testing" for i in range(size//10)}
        qrels = {f"query_{i}": {f"doc_{i*10}": 1} for i in range(size//10)}
        
        return {
            "corpus": corpus,
            "queries": queries,
            "qrels": qrels,
            "type": "synthetic_retrieval",
            "size": size
        }
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """データセット情報取得"""
        info = {
            "20newsgroups": {
                "description": "Text classification benchmark with 20 categories",
                "task_type": "classification",
                "size": "~20,000 documents",
                "metrics": ["accuracy", "f1_score", "precision", "recall"]
            },
            "ms_marco": {
                "description": "Microsoft Machine Reading Comprehension dataset",
                "task_type": "qa",
                "size": "~1M questions",
                "metrics": ["exact_match", "f1_score", "bleu"]
            },
            "natural_questions": {
                "description": "Real user questions from Google search",
                "task_type": "qa",
                "size": "~300k questions",
                "metrics": ["exact_match", "f1_score"]
            },
            "hotpot_qa": {
                "description": "Multi-hop reasoning QA dataset",
                "task_type": "multi_hop_qa",
                "size": "~113k questions",
                "metrics": ["exact_match", "f1_score", "reasoning_accuracy"]
            },
            "beir": {
                "description": "Diverse information retrieval benchmark",
                "task_type": "retrieval",
                "size": "Various datasets",
                "metrics": ["ndcg@10", "map", "recall@100"]
            },
            "maze_benchmark": {
                "description": "Pathfinding optimization benchmark",
                "task_type": "pathfinding",
                "size": "Generated mazes",
                "metrics": ["path_length", "computation_time", "memory_usage", "optimality"]
            }
        }
        
        return info.get(dataset_name, {"description": "Unknown dataset", "task_type": "unknown"})
