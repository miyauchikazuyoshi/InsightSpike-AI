#!/usr/bin/env python3
"""
Load HuggingFace datasets and prepare for InsightSpike ingestion
"""

import json
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Dict


def load_arrow_file(arrow_path: Path) -> List[Dict]:
    """Load data from Arrow file"""
    try:
        table = pq.read_table(str(arrow_path))
        df = table.to_pandas()
        return df.to_dict('records')
    except Exception as e:
        print(f"Error loading {arrow_path}: {e}")
        return []


def prepare_qa_pairs() -> List[Dict]:
    """Prepare Q&A pairs from HuggingFace datasets"""
    qa_pairs = []
    
    # Dataset paths
    dataset_base = Path("experiments/gedig_embedding_evaluation/data")
    
    # 1. Load SQuAD data
    squad_paths = [
        dataset_base / "huggingface_datasets/squad_30",
        dataset_base / "large_huggingface_datasets/squad_100",
        dataset_base / "mega_huggingface_datasets/squad_200",
    ]
    
    for squad_path in squad_paths:
        if squad_path.exists():
            arrow_file = squad_path / "data-00000-of-00001.arrow"
            if arrow_file.exists():
                data = load_arrow_file(arrow_file)
                for item in data[:10]:  # Limit to 10 per dataset for testing
                    if 'question' in item and 'context' in item:
                        qa_pairs.append({
                            'question': item['question'],
                            'context': item['context'],
                            'answer': item.get('answers', {}).get('text', [''])[0] if isinstance(item.get('answers'), dict) else '',
                            'source': 'squad'
                        })
    
    # 2. Load MS MARCO data
    marco_paths = [
        dataset_base / "huggingface_datasets/ms_marco_20",
        dataset_base / "large_huggingface_datasets/ms_marco_50",
    ]
    
    for marco_path in marco_paths:
        if marco_path.exists():
            arrow_file = marco_path / "data-00000-of-00001.arrow"
            if arrow_file.exists():
                data = load_arrow_file(arrow_file)
                for item in data[:10]:
                    if 'query' in item and 'passage' in item:
                        qa_pairs.append({
                            'question': item['query'],
                            'context': item['passage'],
                            'answer': '',
                            'source': 'ms_marco'
                        })
    
    # 3. Load CoQA data
    coqa_paths = [
        dataset_base / "large_huggingface_datasets/coqa_30",
        dataset_base / "mega_huggingface_datasets/coqa_80",
    ]
    
    for coqa_path in coqa_paths:
        if coqa_path.exists():
            arrow_file = coqa_path / "data-00000-of-00001.arrow"
            if arrow_file.exists():
                data = load_arrow_file(arrow_file)
                for item in data[:5]:
                    if 'question' in item and 'story' in item:
                        qa_pairs.append({
                            'question': item['question'],
                            'context': item['story'],
                            'answer': item.get('answer', ''),
                            'source': 'coqa'
                        })
    
    return qa_pairs


def save_prepared_data(qa_pairs: List[Dict], output_path: str):
    """Save prepared data for InsightSpike ingestion"""
    output = Path(output_path)
    output.parent.mkdir(exist_ok=True)
    
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(qa_pairs)} Q&A pairs to {output}")
    
    # Also save as simple text format for easy CLI ingestion
    text_output = output.with_suffix('.txt')
    with open(text_output, 'w', encoding='utf-8') as f:
        for qa in qa_pairs:
            # Format as knowledge statements
            f.write(f"Question: {qa['question']}\\n")
            f.write(f"Context: {qa['context'][:200]}...\\n")
            if qa['answer']:
                f.write(f"Answer: {qa['answer']}\\n")
            f.write("\\n")
    
    print(f"Saved text format to {text_output}")


def main():
    """Prepare HuggingFace data for InsightSpike"""
    print("Loading HuggingFace datasets...")
    
    qa_pairs = prepare_qa_pairs()
    print(f"Loaded {len(qa_pairs)} Q&A pairs")
    
    # Show sample
    if qa_pairs:
        print("\\nSample data:")
        for i, qa in enumerate(qa_pairs[:3]):
            print(f"\\n{i+1}. {qa['source']}:")
            print(f"   Q: {qa['question'][:80]}...")
            print(f"   C: {qa['context'][:80]}...")
    
    # Save prepared data
    save_prepared_data(qa_pairs, "experiment_2/dynamic_growth/prepared_data.json")


if __name__ == "__main__":
    main()