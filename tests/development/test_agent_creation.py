#!/usr/bin/env python3
import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress all output during imports
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.CRITICAL)

print("Testing MainAgent creation...", flush=True)

try:
    # Step 1: Config
    print("1. Loading config...", flush=True)
    from src.insightspike.config import load_config
    config = load_config(config_path="experiments/english_insight_reproduction/config_experiment.yaml")
    print("   ✓ Config loaded", flush=True)
    
    # Step 2: Try to create MainAgent
    print("2. Creating MainAgent...", flush=True)
    from src.insightspike.implementations.agents.main_agent import MainAgent
    
    # Try without datastore first
    try:
        agent = MainAgent(config)
        print("   ✓ MainAgent created without datastore", flush=True)
    except Exception as e:
        print(f"   ✗ Failed without datastore: {e}", flush=True)
        
        # Try with in-memory datastore
        print("3. Creating with in-memory datastore...", flush=True)
        from src.insightspike.core.base.datastore import DataStore
        
        class InMemoryDataStore(DataStore):
            def __init__(self):
                self.episodes = []
                
            def add_episode(self, episode):
                self.episodes.append(episode)
                return len(self.episodes) - 1
                
            def get_episode(self, id):
                if 0 <= id < len(self.episodes):
                    return self.episodes[id]
                return None
                
            def list_episodes(self, limit=100, offset=0):
                return self.episodes[offset:offset+limit]
                
            def update_episode(self, id, episode):
                if 0 <= id < len(self.episodes):
                    self.episodes[id] = episode
                    return True
                return False
                
            def delete_episode(self, id):
                if 0 <= id < len(self.episodes):
                    del self.episodes[id]
                    return True
                return False
                
            def search_episodes(self, query, limit=10):
                # Simple text search
                results = []
                for i, ep in enumerate(self.episodes):
                    if query.lower() in str(ep).lower():
                        results.append(ep)
                        if len(results) >= limit:
                            break
                return results
                
            def get_stats(self):
                return {"episode_count": len(self.episodes)}
        
        datastore = InMemoryDataStore()
        agent = MainAgent(config, datastore)
        print("   ✓ MainAgent created with datastore", flush=True)
    
    # Step 3: Test basic functionality
    print("4. Testing add_knowledge...", flush=True)
    result = agent.add_knowledge("Test knowledge")
    print(f"   ✓ Knowledge added: {result}", flush=True)
    
    print("\n✅ All tests passed!", flush=True)
    
except Exception as e:
    print(f"\n✗ Error: {e}", flush=True)
    import traceback
    traceback.print_exc()