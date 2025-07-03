#!/usr/bin/env python3
"""
Create test data for dynamic growth experiment
"""

import json
from pathlib import Path


def create_test_knowledge_items():
    """Create diverse test knowledge items"""
    knowledge_items = []
    
    # 1. General knowledge
    general_knowledge = [
        "The capital of Japan is Tokyo, which is the world's most populous metropolitan area.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Python was created by Guido van Rossum and first released in 1991.",
        "The speed of light in vacuum is approximately 299,792,458 meters per second.",
        "DNA stands for deoxyribonucleic acid and contains genetic instructions.",
        "The Great Wall of China is over 13,000 miles long and took centuries to build.",
        "Photosynthesis is the process by which plants convert light energy into chemical energy.",
        "The human brain contains approximately 86 billion neurons.",
        "Climate change is primarily caused by greenhouse gas emissions from human activities.",
        "Quantum computing uses quantum bits or qubits that can exist in multiple states.",
    ]
    
    # 2. Technical knowledge
    technical_knowledge = [
        "REST API stands for Representational State Transfer Application Programming Interface.",
        "Git is a distributed version control system created by Linus Torvalds.",
        "Docker containers package applications with their dependencies for consistent deployment.",
        "Neural networks are composed of layers of interconnected nodes inspired by biological neurons.",
        "SQL is a domain-specific language used for managing relational databases.",
        "Kubernetes is an open-source container orchestration platform originally developed by Google.",
        "TCP/IP is the fundamental communication protocol of the Internet.",
        "Blockchain is a distributed ledger technology that ensures data immutability.",
        "OAuth 2.0 is an authorization framework that enables third-party access to resources.",
        "GraphQL is a query language for APIs developed by Facebook.",
    ]
    
    # 3. Q&A style knowledge
    qa_knowledge = [
        "Question: What is the purpose of RAG systems? Answer: RAG (Retrieval-Augmented Generation) systems combine retrieval mechanisms with language generation to provide accurate, contextual responses.",
        "Question: How does compression work in InsightSpike? Answer: InsightSpike uses graph-based compression and episodic memory to reduce storage requirements while maintaining information accessibility.",
        "Question: What are embeddings? Answer: Embeddings are dense vector representations of text that capture semantic meaning in a high-dimensional space.",
        "Question: Why is FAISS used for vector search? Answer: FAISS provides efficient similarity search in high-dimensional spaces using optimized indexing structures.",
        "Question: What is the advantage of episodic memory? Answer: Episodic memory allows systems to learn from experiences and adapt their behavior based on past interactions.",
    ]
    
    # 4. Contextual knowledge
    contextual_knowledge = [
        "In the context of natural language processing, transformers have revolutionized the field by enabling parallel processing and capturing long-range dependencies.",
        "When discussing database performance, indexing is crucial as it significantly speeds up data retrieval operations at the cost of additional storage space.",
        "Regarding cybersecurity, zero-trust architecture assumes no implicit trust and continuously verifies every transaction.",
        "In software development, agile methodologies emphasize iterative development, collaboration, and adaptability to changing requirements.",
        "For distributed systems, the CAP theorem states that it's impossible to simultaneously guarantee consistency, availability, and partition tolerance.",
    ]
    
    # Combine all knowledge
    for item in general_knowledge:
        knowledge_items.append({'text': item, 'type': 'general'})
    
    for item in technical_knowledge:
        knowledge_items.append({'text': item, 'type': 'technical'})
    
    for item in qa_knowledge:
        knowledge_items.append({'text': item, 'type': 'qa'})
    
    for item in contextual_knowledge:
        knowledge_items.append({'text': item, 'type': 'contextual'})
    
    # Add some longer, more complex items
    complex_items = [
        {
            'text': "Artificial General Intelligence (AGI) represents a theoretical form of AI that would match or exceed human cognitive abilities across all domains. Unlike narrow AI, which excels at specific tasks, AGI would demonstrate human-like reasoning, learning, and problem-solving capabilities. The development of AGI remains one of the most significant challenges in computer science.",
            'type': 'complex'
        },
        {
            'text': "The transformer architecture, introduced in the 'Attention is All You Need' paper, revolutionized NLP by replacing recurrent layers with self-attention mechanisms. This allows for parallel processing of sequences and better capture of long-range dependencies. Models like BERT, GPT, and T5 are all based on transformer architecture.",
            'type': 'complex'
        },
        {
            'text': "Graph neural networks (GNNs) extend deep learning to graph-structured data. They work by aggregating information from neighboring nodes through message passing. Applications include molecular property prediction, social network analysis, and knowledge graph reasoning. InsightSpike uses graph structures for efficient information storage.",
            'type': 'complex'
        }
    ]
    
    knowledge_items.extend(complex_items)
    
    return knowledge_items


def save_test_data(knowledge_items, output_dir):
    """Save test data in various formats"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_path = output_path / "test_knowledge.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_items, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(knowledge_items)} items to {json_path}")
    
    # Save as text file for easy reading
    text_path = output_path / "test_knowledge.txt"
    with open(text_path, 'w', encoding='utf-8') as f:
        for i, item in enumerate(knowledge_items):
            f.write(f"{i+1}. [{item['type']}] {item['text']}\\n\\n")
    print(f"Saved text format to {text_path}")
    
    # Save individual batches for controlled testing
    batch_size = 10
    for i in range(0, len(knowledge_items), batch_size):
        batch = knowledge_items[i:i+batch_size]
        batch_path = output_path / f"batch_{i//batch_size + 1}.json"
        with open(batch_path, 'w', encoding='utf-8') as f:
            json.dump(batch, f, indent=2)
    print(f"Created {len(knowledge_items)//batch_size + 1} batch files")


def main():
    """Create test data for experiment"""
    print("Creating test knowledge items...")
    
    knowledge_items = create_test_knowledge_items()
    print(f"Created {len(knowledge_items)} knowledge items")
    
    # Show statistics
    types = {}
    for item in knowledge_items:
        types[item['type']] = types.get(item['type'], 0) + 1
    
    print("\\nKnowledge distribution:")
    for type_name, count in types.items():
        print(f"  {type_name}: {count} items")
    
    # Calculate average length
    avg_length = sum(len(item['text']) for item in knowledge_items) / len(knowledge_items)
    print(f"\\nAverage item length: {avg_length:.0f} characters")
    
    # Save data
    save_test_data(knowledge_items, "experiment_2/dynamic_growth")
    
    print("\\nTest data created successfully!")


if __name__ == "__main__":
    main()