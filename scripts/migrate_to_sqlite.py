#!/usr/bin/env python3
"""
Migrate Legacy Data to SQLite DataStore
=======================================

Migrates existing JSON/PyTorch data files to the new SQLite-based DataStore.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.insightspike.implementations.datastore.sqlite_store import SQLiteDataStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataMigrator:
    """Migrate legacy data to SQLite DataStore"""
    
    def __init__(self, legacy_data_dir: str, target_db_path: str):
        self.legacy_data_dir = Path(legacy_data_dir)
        self.datastore = SQLiteDataStore(target_db_path, vector_dim=384)
        
    def load_legacy_episodes(self, file_path: str) -> List[Dict[str, Any]]:
        """Load episodes from legacy JSON file"""
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"Legacy file not found: {file_path}")
            return []
            
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            # Handle different formats
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                if 'episodes' in data:
                    return data['episodes']
                elif 'data' in data:
                    return data['data']
                else:
                    logger.warning(f"Unknown dict format in {file_path}")
                    return []
            else:
                logger.warning(f"Unknown data format in {file_path}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return []
    
    def load_legacy_graph(self, file_path: str) -> Dict[str, Any]:
        """Load graph from legacy PyTorch file"""
        try:
            import torch
            
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"Legacy graph file not found: {file_path}")
                return {}
                
            # Load PyTorch graph
            graph_data = torch.load(path, map_location='cpu')
            
            # Convert to our format
            graph_dict = {
                'nodes': {},
                'edges': []
            }
            
            # Extract nodes
            if hasattr(graph_data, 'x'):
                # Node features
                num_nodes = graph_data.x.shape[0]
                for i in range(num_nodes):
                    graph_dict['nodes'][f'node_{i}'] = {
                        'type': 'episode',
                        'attributes': {
                            'features': graph_data.x[i].tolist() if hasattr(graph_data.x[i], 'tolist') else []
                        }
                    }
            
            # Extract edges
            if hasattr(graph_data, 'edge_index'):
                edge_index = graph_data.edge_index
                for i in range(edge_index.shape[1]):
                    source = f'node_{edge_index[0, i].item()}'
                    target = f'node_{edge_index[1, i].item()}'
                    graph_dict['edges'].append({
                        'source': source,
                        'target': target,
                        'type': 'similarity',
                        'attributes': {}
                    })
                    
            return graph_dict
            
        except ImportError:
            logger.error("PyTorch not installed. Cannot load graph data.")
            return {}
        except Exception as e:
            logger.error(f"Failed to load graph {file_path}: {e}")
            return {}
    
    async def migrate_episodes(self, episodes: List[Dict[str, Any]], namespace: str = "migrated"):
        """Migrate episodes to DataStore"""
        logger.info(f"Migrating {len(episodes)} episodes to namespace '{namespace}'")
        
        migrated = 0
        failed = 0
        
        for i, episode in enumerate(episodes):
            try:
                # Ensure required fields
                if 'text' not in episode:
                    logger.warning(f"Episode {i} missing 'text' field")
                    failed += 1
                    continue
                
                # Handle vector field
                vec = None
                if 'vec' in episode:
                    vec = np.array(episode['vec'], dtype=np.float32)
                elif 'vector' in episode:
                    vec = np.array(episode['vector'], dtype=np.float32)
                elif 'embedding' in episode:
                    vec = np.array(episode['embedding'], dtype=np.float32)
                else:
                    # Generate random vector as fallback
                    vec = np.random.rand(384).astype(np.float32)
                    logger.warning(f"Episode {i} missing vector, using random")
                
                # Prepare episode data
                episode_data = {
                    'text': episode['text'],
                    'vec': vec,
                    'c': episode.get('c', episode.get('c_value', 0.5)),
                    'metadata': episode.get('metadata', {})
                }
                
                # Add ID if available
                if 'id' in episode:
                    episode_data['id'] = str(episode['id'])
                
                # Save to DataStore (use sync method)
                try:
                    success = self.datastore.save_episodes([episode_data], namespace=namespace)
                except Exception as e:
                    logger.error(f"Failed to save episode {i}: {e}")
                    success = False
                
                if success:
                    migrated += 1
                else:
                    failed += 1
                    
                # Progress update
                if (i + 1) % 100 == 0:
                    logger.info(f"Progress: {i + 1}/{len(episodes)} episodes")
                    
            except Exception as e:
                logger.error(f"Failed to migrate episode {i}: {e}")
                failed += 1
        
        logger.info(f"Migration complete: {migrated} succeeded, {failed} failed")
    
    async def migrate_graph(self, graph_data: Dict[str, Any], graph_id: str, namespace: str = "graphs"):
        """Migrate graph to DataStore"""
        if not graph_data:
            logger.warning("No graph data to migrate")
            return
            
        logger.info(f"Migrating graph '{graph_id}' with {len(graph_data.get('nodes', {}))} nodes")
        
        try:
            success = self.datastore.save_graph(graph_data, graph_id, namespace)
            if success:
                logger.info(f"Successfully migrated graph '{graph_id}'")
            else:
                logger.error(f"Failed to migrate graph '{graph_id}'")
        except Exception as e:
            logger.error(f"Error migrating graph: {e}")
    
    async def run_migration(self):
        """Run the complete migration process"""
        logger.info("Starting data migration to SQLite DataStore")
        
        # 1. Migrate episodes
        episodes_files = [
            self.legacy_data_dir / "core" / "episodes.json",
            self.legacy_data_dir / "episodes" / "episodes.json",
            self.legacy_data_dir / "knowledge_base.json",
            self.legacy_data_dir / "episodes_clean.json"  # Add clean backup support
        ]
        
        all_episodes = []
        for file_path in episodes_files:
            if file_path.exists():
                logger.info(f"Loading episodes from {file_path}")
                episodes = self.load_legacy_episodes(str(file_path))
                all_episodes.extend(episodes)
        
        if all_episodes:
            await self.migrate_episodes(all_episodes, namespace="episodes")
        else:
            logger.warning("No episodes found to migrate")
        
        # 2. Migrate graphs
        graph_files = [
            self.legacy_data_dir / "core" / "graph_pyg.pt",
            self.legacy_data_dir / "graphs" / "knowledge_graph.pt"
        ]
        
        for i, file_path in enumerate(graph_files):
            if file_path.exists():
                logger.info(f"Loading graph from {file_path}")
                graph_data = self.load_legacy_graph(str(file_path))
                if graph_data:
                    await self.migrate_graph(graph_data, f"graph_{i}", namespace="graphs")
        
        # 3. Show statistics
        try:
            stats = self.datastore.get_stats()
            logger.info("Migration statistics:")
            logger.info(f"  Episodes: {stats.get('episodes', {})}")
            logger.info(f"  Graphs: {stats.get('graphs', {})}")
            logger.info(f"  DB Size: {stats.get('db_size_bytes', 0) / 1024 / 1024:.2f} MB")
        except Exception as e:
            logger.warning(f"Could not get statistics: {e}")
        
        # 4. Save indices
        self.datastore.save_indices()
        logger.info("Vector indices saved")
        
        logger.info("Migration complete!")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate legacy data to SQLite DataStore")
    parser.add_argument(
        "--legacy-dir",
        type=str,
        default="./data",
        help="Path to legacy data directory"
    )
    parser.add_argument(
        "--target-db",
        type=str,
        default="./data/sqlite/insightspike.db",
        help="Path to target SQLite database"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean existing database before migration"
    )
    
    args = parser.parse_args()
    
    # Clean database if requested
    if args.clean and os.path.exists(args.target_db):
        os.remove(args.target_db)
        logger.info(f"Removed existing database: {args.target_db}")
    
    # Create target directory
    os.makedirs(os.path.dirname(args.target_db), exist_ok=True)
    
    # Run migration
    migrator = DataMigrator(args.legacy_dir, args.target_db)
    await migrator.run_migration()


if __name__ == "__main__":
    asyncio.run(main())