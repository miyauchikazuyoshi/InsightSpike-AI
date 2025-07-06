#!/usr/bin/env python3
"""
Data Integrity Validation Script
================================

Validates the consistency and integrity of data files in the InsightSpike-AI system.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import torch
    import faiss
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not available, skipping graph validation")


class DataValidator:
    """Validates data integrity across all system files."""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path(__file__).parent
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def validate_all(self) -> Tuple[bool, Dict[str, any]]:
        """Run all validation checks."""
        print("🔍 Starting data validation...\n")
        
        results = {
            "episodes": self._validate_episodes(),
            "index": self._validate_index(),
            "graph": self._validate_graph() if PYTORCH_AVAILABLE else None,
            "databases": self._validate_databases(),
            "consistency": self._validate_consistency(),
        }
        
        # Summary
        print("\n📊 Validation Summary:")
        print(f"  ✅ Passed: {sum(1 for v in results.values() if v)}")
        print(f"  ❌ Failed: {sum(1 for v in results.values() if v is False)}")
        print(f"  ⚠️  Warnings: {len(self.warnings)}")
        
        if self.errors:
            print("\n❌ Errors found:")
            for error in self.errors:
                print(f"  - {error}")
                
        if self.warnings:
            print("\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        return len(self.errors) == 0, results
    
    def _validate_episodes(self) -> bool:
        """Validate episodes.json file."""
        print("📄 Validating episodes.json...")
        
        episodes_path = self.data_dir / "episodes.json"
        if not episodes_path.exists():
            self.warnings.append("episodes.json not found")
            return None
            
        try:
            with open(episodes_path, 'r') as f:
                episodes = json.load(f)
            
            if not isinstance(episodes, list):
                self.errors.append("episodes.json should contain a list")
                return False
                
            # Check each episode
            for i, ep in enumerate(episodes):
                if not isinstance(ep, dict):
                    self.errors.append(f"Episode {i} is not a dictionary")
                    continue
                    
                # Required fields
                required = ['id', 'text', 'c', 'vec']
                for field in required:
                    if field not in ep:
                        self.errors.append(f"Episode {i} missing field: {field}")
                
                # Validate vector
                if 'vec' in ep:
                    vec = ep['vec']
                    if not isinstance(vec, list) or len(vec) != 384:
                        self.errors.append(f"Episode {i} has invalid vector (should be 384-dim)")
                        
                # Validate c-value
                if 'c' in ep:
                    if not (0 <= ep['c'] <= 1):
                        self.warnings.append(f"Episode {i} has c-value outside [0,1]: {ep['c']}")
            
            print(f"  ✓ Found {len(episodes)} episodes")
            return True
            
        except Exception as e:
            self.errors.append(f"Failed to load episodes.json: {e}")
            return False
    
    def _validate_index(self) -> bool:
        """Validate FAISS index."""
        print("🔍 Validating FAISS index...")
        
        index_path = self.data_dir / "index.faiss"
        if not index_path.exists():
            self.warnings.append("index.faiss not found")
            return None
            
        try:
            index = faiss.read_index(str(index_path))
            
            print(f"  ✓ Index type: {type(index).__name__}")
            print(f"  ✓ Vectors: {index.ntotal}")
            print(f"  ✓ Dimension: {index.d}")
            
            if index.d != 384:
                self.errors.append(f"Index dimension should be 384, got {index.d}")
                return False
                
            return True
            
        except Exception as e:
            self.errors.append(f"Failed to load index.faiss: {e}")
            return False
    
    def _validate_graph(self) -> bool:
        """Validate PyTorch graph."""
        print("📊 Validating graph_pyg.pt...")
        
        graph_path = self.data_dir / "graph_pyg.pt"
        if not graph_path.exists():
            self.warnings.append("graph_pyg.pt not found")
            return None
            
        try:
            graph = torch.load(graph_path, map_location='cpu')
            
            if hasattr(graph, 'x'):
                print(f"  ✓ Nodes: {graph.x.shape[0]}")
                print(f"  ✓ Features: {graph.x.shape[1]}")
            
            if hasattr(graph, 'edge_index'):
                print(f"  ✓ Edges: {graph.edge_index.shape[1]}")
                
            return True
            
        except Exception as e:
            self.errors.append(f"Failed to load graph_pyg.pt: {e}")
            return False
    
    def _validate_databases(self) -> bool:
        """Validate SQLite databases."""
        print("🗄️  Validating databases...")
        
        all_valid = True
        
        # Check insight_facts.db
        insights_db = self.data_dir / "insight_facts.db"
        if insights_db.exists():
            print(f"  ✓ insight_facts.db: {insights_db.stat().st_size / 1024:.1f} KB")
        else:
            self.warnings.append("insight_facts.db not found")
            
        # Check unknown_learning.db
        learning_db = self.data_dir / "unknown_learning.db"
        if learning_db.exists():
            print(f"  ✓ unknown_learning.db: {learning_db.stat().st_size / 1024:.1f} KB")
        else:
            self.warnings.append("unknown_learning.db not found")
            
        return all_valid
    
    def _validate_consistency(self) -> bool:
        """Validate consistency between files."""
        print("🔗 Validating cross-file consistency...")
        
        # Load episodes
        episodes_path = self.data_dir / "episodes.json"
        if episodes_path.exists():
            with open(episodes_path, 'r') as f:
                episodes = json.load(f)
            episode_count = len(episodes)
            print(f"  Episodes: {episode_count}")
        else:
            return None
            
        # Check index
        index_path = self.data_dir / "index.faiss"
        if index_path.exists():
            index = faiss.read_index(str(index_path))
            index_count = index.ntotal
            print(f"  Index vectors: {index_count}")
            
            if episode_count != index_count:
                self.errors.append(
                    f"Mismatch: {episode_count} episodes but {index_count} index vectors"
                )
                return False
        
        # Check graph
        if PYTORCH_AVAILABLE:
            graph_path = self.data_dir / "graph_pyg.pt"
            if graph_path.exists():
                graph = torch.load(graph_path, map_location='cpu')
                if hasattr(graph, 'x'):
                    graph_nodes = graph.x.shape[0]
                    print(f"  Graph nodes: {graph_nodes}")
                    
                    # Graph nodes can be less than episodes (due to integration)
                    if graph_nodes > episode_count:
                        self.errors.append(
                            f"Graph has more nodes ({graph_nodes}) than episodes ({episode_count})"
                        )
                        return False
        
        print("  ✓ Files are consistent")
        return True


def main():
    """Run validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate InsightSpike-AI data integrity")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Data directory to validate"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix issues (not implemented yet)"
    )
    
    args = parser.parse_args()
    
    validator = DataValidator(args.data_dir)
    success, results = validator.validate_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()