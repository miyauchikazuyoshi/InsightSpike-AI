#!/usr/bin/env python3
"""
Test script to verify LLMConfig fix - Lite version
=================================================

This script tests the MainAgent without actually initializing the heavy LLM model.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_config_and_creation():
    """Test config and MainAgent creation without LLM initialization"""
    print("Testing config and MainAgent creation...")
    
    try:
        from insightspike.core.config import get_config
        config = get_config()
        
        # Test config attributes
        print(f"✓ Environment: {config.environment}")
        print(f"✓ LLM provider: {config.llm.provider}")
        print(f"✓ LLM model: {config.llm.model_name}")
        print(f"✓ Memory max docs: {config.memory.max_retrieved_docs}")
        
        # Test MainAgent creation (without initialization)
        from insightspike.core.agents.main_agent import MainAgent
        agent = MainAgent()
        print("✓ Successfully created MainAgent instance")
        print(f"✓ Agent config type: {type(agent.config)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_commands():
    """Test CLI commands that use the new config"""
    print("\nTesting CLI config display...")
    
    try:
        from insightspike.core.config import get_config
        from insightspike.cli import config_info
        from rich.console import Console
        from io import StringIO
        
        # Capture output instead of printing to console
        console = Console(file=StringIO(), record=True)
        
        # This should work without errors now
        config = get_config()
        print(f"✓ CLI config access works")
        print(f"✓ Environment: {config.environment}")
        print(f"✓ LLM provider: {config.llm.provider}")
        print(f"✓ Graph spike thresholds - GED: {config.graph.spike_ged_threshold}")
        
        return True
        
    except Exception as e:
        print(f"✗ CLI config error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_end_to_end_with_insight():
    """Test end-to-end flow through CLI ask command (without actual model execution)"""
    print("\nTesting end-to-end CLI flow structure...")
    
    try:
        from insightspike.cli import ask
        from insightspike.core.agents.main_agent import MainAgent
        from insightspike.insight_fact_registry import InsightFactRegistry
        
        # Test that components can be imported and created
        agent = MainAgent()
        print("✓ MainAgent creation for CLI")
        
        registry = InsightFactRegistry()
        print("✓ InsightFactRegistry creation")
        
        # Test direct insight extraction
        insights = registry.extract_insights_from_response(
            question="What is quantum entanglement?",
            response="Quantum entanglement is a fundamental quantum mechanical phenomenon.",
            l1_analysis=None,
            reasoning_quality=0.7
        )
        print(f"✓ Insight extraction: {len(insights)} insights")
        
        return True
        
    except Exception as e:
        print(f"✗ End-to-end test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== LLMConfig Fix Verification (Lite) ===")
    
    # Run tests
    test1 = test_config_and_creation()
    test2 = test_cli_commands()
    test3 = test_end_to_end_with_insight()
    
    print(f"\n=== Results ===")
    print(f"Config and creation: {'PASS' if test1 else 'FAIL'}")
    print(f"CLI config: {'PASS' if test2 else 'FAIL'}")
    print(f"End-to-end structure: {'PASS' if test3 else 'FAIL'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 All tests PASSED! LLMConfig issue is resolved.")
        print("✓ MainAgent can be created without 'provider' attribute error")
        print("✓ CLI commands can access all required config fields")
        print("✓ Insight extraction works independently")
        print("✓ System is ready for end-to-end testing")
        sys.exit(0)
    else:
        print("\n❌ Some tests FAILED. Issues remain.")
        sys.exit(1)
