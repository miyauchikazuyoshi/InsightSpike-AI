#!/usr/bin/env python3
"""
Minimal test to verify InsightSpike is working
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("🧪 Minimal InsightSpike Test")

# Simple test with transformers
from transformers import pipeline

print("\n1️⃣ Testing transformers pipeline...")
generator = pipeline("text-generation", model="distilgpt2", device=-1)
result = generator("Energy is", max_new_tokens=10)
print(f"✅ Pipeline works: {result[0]['generated_text']}")

# Now test InsightSpike imports
print("\n2️⃣ Testing InsightSpike imports...")
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from insightspike.config import InsightSpikeConfig
print("✅ Config imported")

from insightspike.core.agents.main_agent import MainAgent
print("✅ MainAgent imported")

# Create minimal config
print("\n3️⃣ Creating minimal agent...")
config = InsightSpikeConfig()
config.core.model_name = "distilgpt2"
config.core.max_tokens = 20  # Very small for speed

agent = MainAgent(config=config)
print("✅ Agent created")

# Initialize
print("\n4️⃣ Initializing agent...")
if agent.initialize():
    print("✅ Agent initialized")
    
    # Add one episode
    print("\n5️⃣ Adding episode...")
    success = agent.l2_memory.store_episode(
        text="Energy is the capacity to do work.",
        c_value=0.5
    )
    print(f"✅ Episode stored: {success}")
    
    # Try to process without hanging
    print("\n6️⃣ Testing question processing (with timeout)...")
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Processing took too long")
    
    # Set 10 second timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)
    
    try:
        result = agent.process_question(
            "What is energy?",
            max_cycles=1,
            verbose=False
        )
        signal.alarm(0)  # Cancel timeout
        
        if isinstance(result, dict):
            print("✅ Got result!")
            print(f"   Response: {result.get('response', 'No response')[:50]}...")
            print(f"   Spike: {result.get('spike_detected', False)}")
        else:
            print(f"❌ Unexpected result type: {type(result)}")
    
    except TimeoutError:
        signal.alarm(0)
        print("❌ Processing timed out after 10 seconds")
        print("   This suggests an issue with the processing loop")
    
else:
    print("❌ Failed to initialize agent")

print("\n✅ Test complete!")