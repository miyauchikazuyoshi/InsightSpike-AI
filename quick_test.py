#!/usr/bin/env python3
"""Quick test for InsightSpike-AI basic functionality"""

import sys
import os
from pathlib import Path

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def main():
    print("🧠 InsightSpike-AI Quick Test")
    print("=" * 40)
    
    try:
        print("1. Testing imports...")
        from insightspike.core.agents.main_agent import MainAgent
        print("✅ MainAgent import successful")
        
        print("2. Creating agent...")
        agent = MainAgent()
        print("✅ Agent creation successful")
        
        print("3. Testing basic initialization...")
        try:
            # Test without full initialization that might download models
            print("   Agent created, checking basic properties...")
            config = agent.config
            print(f"   ✅ Config loaded: embedding_dim={config.embedding_dim}")
            
            # Skip full initialization that downloads models
            print("✅ Basic functionality verified")
            return True
            
        except Exception as e:
            print(f"⚠️ Initialization issue: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n🎯 Test result: {'✅ SUCCESS' if success else '❌ FAILED'}")
