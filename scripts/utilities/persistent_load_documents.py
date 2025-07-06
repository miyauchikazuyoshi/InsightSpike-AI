#!/usr/bin/env python3
"""
Modified load_documents command that automatically saves data after loading
"""

import pathlib
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.insightspike.core.agents.main_agent import MainAgent
from src.insightspike.processing.loader import load_corpus


def load_documents_with_save(path: pathlib.Path):
    """Load documents into the agent's memory and save to disk"""
    try:
        print(f"Loading documents from: {path}")
        
        # Create agent
        agent = MainAgent()
        if not agent.initialize():
            print("Failed to initialize agent")
            return False
        
        # Get initial stats
        initial_stats = agent.l2_memory.get_memory_stats()
        print(f"Initial episodes in memory: {initial_stats['total_episodes']}")
        
        # Load documents
        if path.is_file():
            docs = load_corpus(path)
        elif path.is_dir():
            docs = []
            for txt_file in path.glob("*.txt"):
                docs.extend(load_corpus(txt_file))
        else:
            print(f"Path not found: {path}")
            return False
        
        # Add to memory
        added = 0
        for doc in docs:
            if agent.add_document(doc):
                added += 1
        
        print(f"Successfully loaded {added}/{len(docs)} documents")
        
        # Get updated stats
        final_stats = agent.l2_memory.get_memory_stats()
        print(f"Total episodes after loading: {final_stats['total_episodes']}")
        
        # SAVE THE DATA
        print("\nSaving data to disk...")
        save_success = agent.l2_memory.save()
        if save_success:
            print("✓ Data successfully saved to:")
            print("  - data/index.faiss")
            print("  - data/episodes.json")
        else:
            print("✗ Failed to save data")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Load documents with automatic saving")
    parser.add_argument("path", type=pathlib.Path, help="Path to text file or directory")
    args = parser.parse_args()
    
    success = load_documents_with_save(args.path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()