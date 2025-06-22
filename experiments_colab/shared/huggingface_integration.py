"""
HuggingFace統合モジュール
事前訓練済みモデルとデータセットの効率的利用
"""

import torch
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    TrainingArguments, Trainer,
    pipeline
)
from sentence_transformers import SentenceTransformer
from datasets import Dataset, load_dataset
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class HuggingFaceModelManager:
    """HuggingFaceモデル管理クラス"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.models = {}
        self.tokenizers = {}
    
    def load_sentence_transformer(self, model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
        """Sentence Transformerモデルのロード"""
        if model_name not in self.models:
            model = SentenceTransformer(model_name, device=self.device)
            self.models[model_name] = model
            print(f"✅ Loaded SentenceTransformer: {model_name}")
        
        return self.models[model_name]
    
    def load_language_model(self, model_name: str = 'microsoft/DialoGPT-medium') -> Tuple[AutoModel, AutoTokenizer]:
        """言語モデルとトークナイザーのロード"""
        if model_name not in self.models:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(self.device)
            
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            print(f"✅ Loaded Language Model: {model_name}")
        
        return self.models[model_name], self.tokenizers[model_name]
    
    def create_pipeline(self, task: str, model_name: str = None) -> pipeline:
        """HuggingFace pipelineの作成"""
        return pipeline(
            task,
            model=model_name,
            device=0 if self.device == 'cuda' else -1,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        )
    
    def get_embeddings_batch(self, texts: List[str], model_name: str = 'all-MiniLM-L6-v2',
                           batch_size: int = 32) -> np.ndarray:
        """バッチ処理での埋め込み生成"""
        model = self.load_sentence_transformer(model_name)
        
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.device
        )
        
        return embeddings


class HuggingFaceDatasetManager:
    """HuggingFaceデータセット管理クラス"""
    
    def __init__(self):
        self.datasets = {}
    
    def load_text_dataset(self, dataset_name: str = 'wikitext', 
                         config: str = 'wikitext-2-raw-v1',
                         split: str = 'train') -> Dataset:
        """テキストデータセットのロード"""
        key = f"{dataset_name}_{config}_{split}"
        
        if key not in self.datasets:
            dataset = load_dataset(dataset_name, config, split=split)
            self.datasets[key] = dataset
            print(f"✅ Loaded dataset: {dataset_name} ({len(dataset)} samples)")
        
        return self.datasets[key]
    
    def create_custom_dataset(self, texts: List[str], labels: Optional[List] = None) -> Dataset:
        """カスタムデータセットの作成"""
        data_dict = {"text": texts}
        if labels is not None:
            data_dict["labels"] = labels
        
        dataset = Dataset.from_dict(data_dict)
        print(f"✅ Created custom dataset: {len(dataset)} samples")
        
        return dataset
    
    def preprocess_for_embeddings(self, dataset: Dataset, 
                                text_column: str = 'text',
                                max_length: int = 512) -> Dataset:
        """埋め込み用前処理"""
        def preprocess_function(examples):
            # テキスト長制限
            texts = examples[text_column]
            processed_texts = [text[:max_length] if len(text) > max_length else text 
                             for text in texts]
            return {"processed_text": processed_texts}
        
        processed_dataset = dataset.map(preprocess_function, batched=True)
        print(f"✅ Preprocessed dataset for embeddings")
        
        return processed_dataset


class ScalableRAGSystem:
    """スケーラブルRAGシステム"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model_manager = HuggingFaceModelManager(device)
        self.dataset_manager = HuggingFaceDatasetManager()
        self.embeddings_cache = {}
    
    def setup_retrieval_system(self, documents: List[str],
                             embedding_model: str = 'all-MiniLM-L6-v2') -> Dict[str, Any]:
        """検索システムのセットアップ"""
        # 文書埋め込み生成
        embeddings = self.model_manager.get_embeddings_batch(
            documents, 
            model_name=embedding_model,
            batch_size=64
        )
        
        # FAISS インデックス作成
        try:
            import faiss
            
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # 内積（コサイン類似度）
            
            # GPU利用可能な場合はGPUインデックス使用
            if self.device == 'cuda' and faiss.get_num_gpus() > 0:
                index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
                print("🚀 Using FAISS GPU index")
            
            # 正規化して追加
            faiss.normalize_L2(embeddings)
            index.add(embeddings.astype(np.float32))
            
            return {
                'index': index,
                'embeddings': embeddings,
                'documents': documents,
                'embedding_model': embedding_model
            }
            
        except ImportError:
            print("⚠️ FAISS not available, using simple similarity search")
            return {
                'embeddings': embeddings,
                'documents': documents,
                'embedding_model': embedding_model
            }
    
    def retrieve_documents(self, query: str, retrieval_system: Dict[str, Any],
                         top_k: int = 5) -> List[Tuple[str, float]]:
        """文書検索"""
        # クエリ埋め込み生成
        query_embedding = self.model_manager.get_embeddings_batch(
            [query], 
            model_name=retrieval_system['embedding_model']
        )
        
        if 'index' in retrieval_system:
            # FAISS検索
            faiss.normalize_L2(query_embedding)
            scores, indices = retrieval_system['index'].search(
                query_embedding.astype(np.float32), top_k
            )
            
            results = [
                (retrieval_system['documents'][idx], score)
                for idx, score in zip(indices[0], scores[0])
            ]
        else:
            # 単純類似度検索
            embeddings = retrieval_system['embeddings']
            similarities = np.dot(embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = [
                (retrieval_system['documents'][idx], similarities[idx])
                for idx in top_indices
            ]
        
        return results


def benchmark_huggingface_models(device: str = 'cuda') -> Dict[str, Dict[str, float]]:
    """HuggingFaceモデルのベンチマーク"""
    import time
    
    model_manager = HuggingFaceModelManager(device)
    
    # テストデータ
    test_texts = [
        "This is a test sentence for benchmarking.",
        "HuggingFace models are powerful for NLP tasks.",
        "GPU acceleration significantly improves performance."
    ] * 100  # 300テキストでテスト
    
    results = {}
    
    # Sentence Transformer ベンチマーク
    models_to_test = [
        'all-MiniLM-L6-v2',
        'all-mpnet-base-v2',
        'paraphrase-multilingual-MiniLM-L12-v2'
    ]
    
    for model_name in models_to_test:
        try:
            start_time = time.time()
            embeddings = model_manager.get_embeddings_batch(
                test_texts[:50],  # 50テキストでテスト
                model_name=model_name,
                batch_size=16
            )
            end_time = time.time()
            
            results[model_name] = {
                'processing_time_s': end_time - start_time,
                'texts_per_second': 50 / (end_time - start_time),
                'embedding_dimension': embeddings.shape[1]
            }
            
            print(f"✅ {model_name}: {results[model_name]['texts_per_second']:.1f} texts/sec")
            
        except Exception as e:
            print(f"❌ {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results
