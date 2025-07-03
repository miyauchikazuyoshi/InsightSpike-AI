#!/usr/bin/env python3
"""
Regenerate Data Structure with Proper Embeddings (Simplified)
===========================================================

Uses the fixed MainAgent with SentenceTransformer embeddings.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'src'))

from insightspike.core.agents.main_agent import MainAgent
import numpy as np
import time
import json
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_simple_documents(n_docs=100):
    """Create simple but diverse documents"""
    topics = [
        "machine learning", "deep learning", "neural networks",
        "natural language processing", "computer vision",
        "reinforcement learning", "data science", "algorithms",
        "artificial intelligence", "robotics"
    ]
    
    variations = [
        "{} is a field of artificial intelligence that enables systems to learn from data.",
        "The principles of {} involve mathematical models and computational algorithms.",
        "Applications of {} include pattern recognition, prediction, and automation.",
        "{} has revolutionized how we approach complex computational problems.",
        "Recent advances in {} have led to breakthrough discoveries in various domains.",
        "{} combines statistical methods with computational power to extract insights.",
        "The key to {} is understanding patterns and relationships in data.",
        "{} enables computers to perform tasks that typically require human intelligence.",
        "Research in {} focuses on improving accuracy and efficiency of models.",
        "The future of {} involves more sophisticated and autonomous systems."
    ]
    
    documents = []
    for i in range(n_docs):
        topic = topics[i % len(topics)]
        template = variations[i % len(variations)]
        doc = template.format(topic)
        documents.append(doc)
    
    return documents

def main():
    """Main regeneration process"""
    print("="*60)
    print("Regenerating Data with Proper Embeddings")
    print("="*60)
    
    # Initialize agent
    agent = MainAgent()
    if not agent.initialize():
        logger.error("Failed to initialize agent")
        return
    
    # Check initial state
    print("\nInitial data state:")
    data_dir = Path("data")
    for filename in ["episodes.json", "index.faiss", "graph_pyg.pt"]:
        filepath = data_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  {filename}: {size:,} bytes")
            if filename == "episodes.json":
                with open(filepath, 'r') as f:
                    episodes = json.load(f)
                print(f"    Episodes: {len(episodes)}")
    
    # Create documents
    print("\nCreating documents...")
    documents = create_simple_documents(100)
    
    # Process documents
    print("\nProcessing documents with proper embeddings...")
    save_frequency = 20
    
    for i, doc in enumerate(documents):
        # Process through agent
        result = agent.process_question(
            f"Learn and remember this information: {doc}",
            max_cycles=2
        )
        
        # Log progress
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i+1}/{len(documents)} documents")
        
        # Save state periodically
        if (i + 1) % save_frequency == 0:
            logger.info(f"Saving state at document {i+1}...")
            agent.save_state()
    
    # Final save
    print("\nFinal save...")
    agent.save_state()
    
    # Check final state
    print("\nFinal data state:")
    for filename in ["episodes.json", "index.faiss", "graph_pyg.pt"]:
        filepath = data_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  {filename}: {size:,} bytes")
            if filename == "episodes.json":
                with open(filepath, 'r') as f:
                    episodes = json.load(f)
                print(f"    Episodes: {len(episodes)}")
    
    # Test retrieval
    print("\nTesting retrieval with proper embeddings...")
    test_queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "Explain neural networks",
        "What are the applications of AI?",
        "Tell me about data science"
    ]
    
    for query in test_queries:
        result = agent.process_question(query, max_cycles=2)
        quality = result.get('reasoning_quality', 0)
        doc_count = len(result.get('documents', []))
        print(f"  Query: '{query[:30]}...' - Quality: {quality:.3f}, Docs: {doc_count}")
    
    print("\nRegeneration complete!")

if __name__ == "__main__":
    main()