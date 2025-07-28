#!/usr/bin/env python3
"""
Test full pipeline to find where error occurs
"""

import numpy as np
import logging
import sys

# Redirect stderr to capture error messages
class ErrorCapture:
    def __init__(self):
        self.messages = []
        self.original_stderr = sys.stderr
        
    def write(self, msg):
        self.messages.append(msg)
        self.original_stderr.write(msg)
        
    def flush(self):
        self.original_stderr.flush()

error_capture = ErrorCapture()
sys.stderr = error_capture


def test_with_main_agent():
    """Test through MainAgent to reproduce error."""
    from insightspike.implementations.datastore.memory_store import InMemoryDataStore
    from insightspike.implementations.agents.main_agent import MainAgent
    
    config = {
        "processing": {
            "enable_learning": True,
        },
        "l4_config": {
            "provider": "mock",
        },
        "embedder": {
            "model_name": "mock",
            "vector_dim": 384
        },
        "graph": {
            "use_new_ged_implementation": True
        }
    }
    
    print("1. Creating MainAgent...")
    datastore = InMemoryDataStore()
    agent = MainAgent(config=config, datastore=datastore)
    
    print("2. Adding knowledge...")
    agent.add_knowledge("Test knowledge item 1")
    agent.add_knowledge("Test knowledge item 2")
    
    print("3. Processing question...")
    result = agent.process_question("What is test?")
    
    print(f"\n4. Result: {result.success}")
    print(f"   Spike detected: {result.spike_detected}")
    
    # Check if L3 was initialized
    agent._ensure_l3_initialized()
    if hasattr(agent, 'l3_graph') and agent.l3_graph:
        print("\n5. L3 Graph reasoner exists")
        if hasattr(agent.l3_graph, 'previous_graph'):
            print(f"   Previous graph: {agent.l3_graph.previous_graph}")
    
    # Check captured errors
    print("\n6. Captured error messages:")
    error_msgs = [msg for msg in error_capture.messages if "too many values" in msg]
    for msg in error_msgs:
        print(f"   {msg.strip()}")
    
    # Try to find where the error comes from
    print("\n7. Checking L2 memory retrieval...")
    
    # Get documents from memory
    docs = agent.l2_memory.retrieve(
        query_text="test", 
        top_k=5
    )
    print(f"   Retrieved {len(docs)} documents")
    for i, doc in enumerate(docs):
        print(f"   Doc {i}: keys = {list(doc.keys())}")
        if 'embedding' in doc:
            emb = doc['embedding']
            if isinstance(emb, np.ndarray):
                print(f"          embedding shape = {emb.shape}")
            else:
                print(f"          embedding type = {type(emb)}")


def test_memory_retrieval_directly():
    """Test memory retrieval to see document format."""
    print("\n\n=== Testing Memory Retrieval Directly ===")
    
    from insightspike.implementations.datastore.memory_store import InMemoryDataStore
    from insightspike.implementations.memory import Memory
    
    datastore = InMemoryDataStore()
    memory = Memory(datastore=datastore)
    
    # Add some data
    print("1. Adding episodes...")
    episodes = [
        {
            "text": "Test episode 1",
            "embedding": np.random.randn(384),
            "c_value": 0.5
        },
        {
            "text": "Test episode 2", 
            "embedding": np.random.randn(384),
            "c_value": 0.6
        }
    ]
    
    for ep in episodes:
        memory.add_episode(ep)
    
    print("2. Retrieving...")
    results = memory.retrieve("test", top_k=5)
    
    print(f"   Got {len(results)} results")
    for i, res in enumerate(results):
        print(f"   Result {i}:")
        print(f"     Keys: {list(res.keys())}")
        print(f"     Text: {res.get('text', 'N/A')[:50]}...")
        if 'embedding' in res:
            emb = res['embedding']
            print(f"     Embedding type: {type(emb)}")
            if hasattr(emb, 'shape'):
                print(f"     Embedding shape: {emb.shape}")


if __name__ == "__main__":
    test_with_main_agent()
    test_memory_retrieval_directly()
    
    # Restore stderr
    sys.stderr = error_capture.original_stderr