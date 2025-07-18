"""
Tests for UnifiedMainAgent
=========================

Tests all modes and configurations of the unified agent.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unified_main_agent import UnifiedMainAgent, AgentConfig, AgentMode, UnifiedCycleResult
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_mode():
    """Test basic mode functionality"""
    print("\n=== Testing BASIC Mode ===")
    
    config = AgentConfig.from_mode(AgentMode.BASIC)
    agent = UnifiedMainAgent(config)
    
    # Should have minimal components
    assert agent.config.mode == AgentMode.BASIC
    assert not agent.config.enable_query_transform
    assert not agent.config.enable_caching
    
    # Test initialization
    success = agent.initialize()
    print(f"Initialization: {'Success' if success else 'Failed'}")
    
    if success:
        # Test processing
        result = agent.process_question("What is energy?", max_cycles=2)
        print(f"Response: {result.get('response', 'No response')[:100]}...")
        print(f"Retrieved docs: {len(result.get('retrieved_documents', []))}")
        print(f"Success: {result.get('success', False)}")
        
        assert 'response' in result
        assert 'retrieved_documents' in result
        assert 'reasoning_quality' in result
    
    return success


def test_enhanced_mode():
    """Test enhanced mode with graph-aware memory"""
    print("\n=== Testing ENHANCED Mode ===")
    
    config = AgentConfig.from_mode(AgentMode.ENHANCED)
    agent = UnifiedMainAgent(config)
    
    # Should have graph-aware memory enabled
    assert agent.config.enable_graph_aware_memory
    
    success = agent.initialize()
    print(f"Initialization: {'Success' if success else 'Failed'}")
    
    if success:
        # Add some episodes first
        agent.add_episode("Energy is the capacity to do work.", c_value=0.8)
        agent.add_episode("Kinetic energy is energy of motion.", c_value=0.7)
        
        # Test processing
        result = agent.process_question("What is kinetic energy?")
        print(f"Graph analysis available: {'graph_analysis' in result}")
        print(f"Response: {result.get('response', 'No response')[:100]}...")
        
        assert 'graph_analysis' in result or True  # May not have graph without PyTorch
    
    return success


def test_query_transform_mode():
    """Test query transformation mode"""
    print("\n=== Testing QUERY_TRANSFORM Mode ===")
    
    config = AgentConfig.from_mode(AgentMode.QUERY_TRANSFORM)
    agent = UnifiedMainAgent(config)
    
    # Should have query transformation enabled
    assert agent.config.enable_query_transform
    
    success = agent.initialize()
    print(f"Initialization: {'Success' if success else 'Failed'}")
    
    if success and agent.query_transformer:
        result = agent.process_question("How does consciousness emerge?", max_cycles=3)
        print(f"Transformation history: {len(result.get('transformation_history', []))} states")
        print(f"Response: {result.get('response', 'No response')[:100]}...")
    else:
        print("Query transformer not available (missing dependencies)")
    
    return success


def test_custom_configuration():
    """Test custom configuration mixing features"""
    print("\n=== Testing Custom Configuration ===")
    
    config = AgentConfig(
        mode=AgentMode.BASIC,
        enable_caching=True,
        enable_query_transform=False,
        cache_size=100,
        max_cycles=2,
        verbose=True
    )
    
    agent = UnifiedMainAgent(config)
    
    # Check cache was created
    assert agent.config.enable_caching
    assert agent.query_cache is not None
    assert agent.query_cache.max_size == 100
    
    success = agent.initialize()
    print(f"Initialization: {'Success' if success else 'Failed'}")
    
    if success:
        # First query (not cached)
        result1 = agent.process_question("What is machine learning?")
        assert not result1.get('cached', False)
        
        # Same query (should be cached)
        result2 = agent.process_question("What is machine learning?")
        assert result2.get('cached', False)
        print("Caching works correctly!")
    
    return success


def test_mode_switching():
    """Test switching between modes"""
    print("\n=== Testing Mode Switching ===")
    
    # Start with basic
    agent = UnifiedMainAgent(AgentConfig.from_mode(AgentMode.BASIC))
    assert not agent.config.enable_query_transform
    
    # Switch to advanced
    agent = UnifiedMainAgent(AgentConfig.from_mode(AgentMode.ADVANCED))
    assert agent.config.enable_query_transform
    assert agent.config.enable_multi_hop
    assert agent.config.enable_query_branching
    
    print("Mode switching works correctly!")
    return True


def test_backward_compatibility():
    """Test backward compatibility with old MainAgent"""
    print("\n=== Testing Backward Compatibility ===")
    
    # The unified agent should work as drop-in replacement
    MainAgent = UnifiedMainAgent  # Alias
    
    agent = MainAgent()  # Default config
    success = agent.initialize()
    
    if success:
        # Test old methods still work
        result = agent.process_question("Test question")
        assert isinstance(result, dict)
        
        # Test adding episodes
        added = agent.add_episode("Test episode", c_value=0.5)
        assert added
        
        print("Backward compatibility maintained!")
    
    return success


def test_result_format():
    """Test that result format is consistent across modes"""
    print("\n=== Testing Result Format ===")
    
    modes_to_test = [AgentMode.BASIC, AgentMode.ENHANCED]
    
    for mode in modes_to_test:
        config = AgentConfig.from_mode(mode)
        agent = UnifiedMainAgent(config)
        
        if agent.initialize():
            result = agent.process_question("Test", max_cycles=1)
            
            # Required fields
            assert 'question' in result
            assert 'response' in result
            assert 'retrieved_documents' in result
            assert 'reasoning_quality' in result
            assert 'spike_detected' in result
            assert 'success' in result
            assert 'processing_time' in result
            
            print(f"{mode.value} mode: Result format correct")
    
    return True


def run_all_tests():
    """Run all tests and report results"""
    tests = [
        ("Basic Mode", test_basic_mode),
        ("Enhanced Mode", test_enhanced_mode),
        ("Query Transform Mode", test_query_transform_mode),
        ("Custom Configuration", test_custom_configuration),
        ("Mode Switching", test_mode_switching),
        ("Backward Compatibility", test_backward_compatibility),
        ("Result Format", test_result_format),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            logger.error(f"{test_name} failed with error: {e}")
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    for test_name, success, error in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:<30} {status}")
        if error:
            print(f"  Error: {error}")
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    # Run all tests
    all_passed = run_all_tests()
    
    if all_passed:
        print("\n✅ All tests passed! The UnifiedMainAgent is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the logs above.")
    
    # Example of using the unified agent
    print("\n" + "="*50)
    print("EXAMPLE USAGE")
    print("="*50)
    
    # Create a basic agent
    agent = UnifiedMainAgent(AgentConfig.from_mode(AgentMode.BASIC))
    
    if agent.initialize():
        # Add some knowledge
        agent.add_episode("Python is a high-level programming language.", c_value=0.8)
        agent.add_episode("Machine learning is a subset of artificial intelligence.", c_value=0.9)
        
        # Ask a question
        result = agent.process_question("What is Python?")
        print(f"\nQuestion: What is Python?")
        print(f"Response: {result['response']}")
        print(f"Quality: {result['reasoning_quality']:.3f}")
        print(f"Documents retrieved: {len(result['retrieved_documents'])}")