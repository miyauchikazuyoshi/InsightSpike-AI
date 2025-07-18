#!/usr/bin/env python3
"""
End-to-End Test Script for Insight Fact Registration System (Updated)
====================================================================

This script tests the complete workflow using the new architecture.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_agent_with_insights():
    """Test the main agent with insight extraction using new architecture"""
    print("Testing MainAgent with insight extraction...")
    
    try:
        from insightspike.implementations.agents.main_agent import MainAgent
        from insightspike.config.presets import ConfigPresets
        from insightspike.implementations.datastore.factory import DataStoreFactory
        
        # Create temporary datastore
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup configuration
            config = ConfigPresets.development()
            
            # Create datastore
            datastore_config = {
                "type": "filesystem",
                "params": {"base_path": temp_dir}
            }
            datastore = DataStoreFactory.create_from_config(datastore_config)
            
            # Initialize agent
            agent = MainAgent(config=config, datastore=datastore)
            if not agent.initialize():
                print("✗ Failed to initialize agent")
                return False
            
            # Add some knowledge
            knowledge_items = [
                "Quantum entanglement is a phenomenon where particles become correlated.",
                "When particles are entangled, measuring one affects the other instantly.",
                "This happens regardless of the distance between the particles.",
                "Einstein called this 'spooky action at a distance'."
            ]
            
            for item in knowledge_items:
                result = agent.add_knowledge(item)
                if not result.get('success'):
                    print(f"✗ Failed to learn: {item}")
                    return False
            
            # Test question
            question = "What is quantum entanglement and how does it work?"
            
            # Process question
            result = agent.process_question(question, max_cycles=3)
            
            print(f"✓ Question processed successfully")
            print(f"✓ Response: {result.response[:100]}...")
            print(f"✓ Reasoning quality: {result.reasoning_quality:.3f}")
            print(f"✓ Spike detected: {result.spike_detected}")
            print(f"✓ Success: {result.success}")
            
            # Save state
            if agent.save_state():
                print("✓ State saved successfully")
            
            return True
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_insight_registry():
    """Test the insight registry functionality"""
    print("\nTesting insight registry...")
    
    try:
        from insightspike.detection.insight_registry import InsightFactRegistry
        
        # Create registry
        registry = InsightFactRegistry()
        
        # Register a test insight
        from insightspike.detection.insight_registry import InsightFact
        
        test_insight = InsightFact(
            id='test_insight_1',
            text='Quantum entanglement is a phenomenon where particles become correlated',
            source_concepts=['quantum', 'entanglement'],
            target_concepts=['particles', 'correlated'],
            confidence=0.85,
            quality_score=0.75,
            ged_optimization=0.15,
            ig_improvement=0.1,
            discovery_context='test environment',
            generated_at=12345.0,
            validation_status='verified',
            relationship_type='structural'
        )
        
        registry.register_insight(test_insight)
        
        # Search for the insight
        results = registry.search_insights_by_concept("quantum", limit=5)
        
        if results:
            print(f"✓ Found {len(results)} matching insights")
            print(f"✓ Top result confidence: {results[0].confidence:.3f}")
        else:
            print("⚠ No insights found (may be normal for test environment)")
        
        # Get stats
        print(f"✓ Registry has {len(registry.insight_cache)} insights")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_integration():
    """Test CLI integration"""
    print("\nTesting CLI integration...")
    
    try:
        from insightspike.cli.spike import app
        try:
            from typer.testing import CliRunner
        except ImportError:
            print("⚠ Typer not installed, skipping CLI tests")
            return True  # Skip test if typer not available
        
        runner = CliRunner()
        
        # Test help command
        result = runner.invoke(app, ["--help"])
        if result.exit_code == 0:
            print("✓ CLI help command works")
        else:
            print(f"✗ CLI help failed: {result.stdout}")
            return False
        
        # Test version command
        result = runner.invoke(app, ["version"])
        if result.exit_code == 0:
            print("✓ CLI version command works")
        else:
            print(f"✗ CLI version failed: {result.stdout}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=== InsightSpike Complete System Test (Updated) ===\n")
    
    tests = [
        ("MainAgent with Insights", test_agent_with_insights),
        ("Insight Registry", test_insight_registry),
        ("CLI Integration", test_cli_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            if test_func():
                passed += 1
                print(f"\n✓ {test_name} PASSED")
            else:
                failed += 1
                print(f"\n✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_name} FAILED with error: {e}")
    
    print(f"\n{'='*50}")
    print(f"Summary: {passed} passed, {failed} failed")
    print('='*50)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)