#!/usr/bin/env python3
"""
Test script to verify LLMConfig fix
===================================

This script tests if the MainAgent can be created and initialized without the 
'LLMConfig' object has no attribute 'provider' error.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_main_agent_creation():
    """Test MainAgent creation and initialization"""
    print("Testing MainAgent creation...")
    
    try:
        from insightspike.core.agents.main_agent import MainAgent
        print("✓ Successfully imported MainAgent")
        
        # Create agent
        agent = MainAgent()
        print("✓ Successfully created MainAgent instance")
        
        # Check config
        print(f"✓ Config type: {type(agent.config)}")
        print(f"✓ LLM provider: {agent.config.llm.provider}")
        print(f"✓ LLM model: {agent.config.llm.model_name}")
        
        # Test initialization
        print("Testing agent initialization...")
        success = agent.initialize()
        print(f"✓ Agent initialization: {'SUCCESS' if success else 'FAILED'}")
        
        return success
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_completeness():
    """Test if config has all required attributes"""
    print("\nTesting config completeness...")
    
    try:
        from insightspike.core.config import get_config
        config = get_config()
        
        # Check required attributes
        required_attrs = [
            'environment',
            'llm.provider',
            'llm.model_name',
            'memory.max_retrieved_docs',
            'graph.spike_ged_threshold',
            'graph.spike_ig_threshold',
            'reasoning.spike_ged_threshold'
        ]
        
        for attr_path in required_attrs:
            try:
                obj = config
                for attr in attr_path.split('.'):
                    obj = getattr(obj, attr)
                print(f"✓ {attr_path}: {obj}")
            except AttributeError as e:
                print(f"✗ Missing: {attr_path} - {e}")
                return False
                
        print("✓ All required config attributes present")
        return True
        
    except Exception as e:
        print(f"✗ Config test error: {e}")
        return False

def test_insight_extraction():
    """Test insight extraction directly"""
    print("\nTesting insight extraction...")
    
    try:
        from insightspike.insight_fact_registry import InsightFactRegistry
        
        registry = InsightFactRegistry()
        print("✓ Created InsightFactRegistry")
        
        # Test insight extraction with simple input
        insights = registry.extract_insights_from_response(
            question="What is quantum entanglement?",
            response="Quantum entanglement is a phenomenon where particles become correlated.",
            l1_analysis=None,
            reasoning_quality=0.8
        )
        
        print(f"✓ Extracted {len(insights)} insights")
        for insight in insights:
            print(f"  - {insight.fact_text[:50]}...")
            
        return True
        
    except Exception as e:
        print(f"✗ Insight extraction error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== LLMConfig Fix Verification ===")
    
    # Run tests
    test1 = test_config_completeness()
    test2 = test_main_agent_creation()
    test3 = test_insight_extraction()
    
    print(f"\n=== Results ===")
    print(f"Config completeness: {'PASS' if test1 else 'FAIL'}")
    print(f"MainAgent creation: {'PASS' if test2 else 'FAIL'}")
    print(f"Insight extraction: {'PASS' if test3 else 'FAIL'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 All tests PASSED! LLMConfig issue is resolved.")
        sys.exit(0)
    else:
        print("\n❌ Some tests FAILED. Issues remain.")
        sys.exit(1)
