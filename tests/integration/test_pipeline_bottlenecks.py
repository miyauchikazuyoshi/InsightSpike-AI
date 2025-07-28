"""
Focused Bottleneck Identification Test
Quick test to identify the main pipeline issues
"""

import os
import warnings
warnings.filterwarnings("ignore")

from insightspike.config.presets import ConfigPresets
from insightspike.implementations.agents import MainAgent
from insightspike.implementations.datastore.memory_store import InMemoryDataStore
from insightspike.patches.apply_fixes import apply_all_fixes

# Apply patches
apply_all_fixes()

def test_basic_initialization():
    """Test basic agent initialization and identify immediate failures"""
    print("=== BOTTLENECK TEST: Basic Initialization ===\n")
    
    issues = []
    
    # Test 1: Config Loading
    print("1. Testing config loading...")
    try:
        config = ConfigPresets.development()
        print("   ✅ Config loaded successfully")
    except Exception as e:
        print(f"   ❌ Config loading failed: {e}")
        issues.append(("config_loading", str(e)))
        return issues
    
    # Test 2: DataStore Creation
    print("2. Testing datastore creation...")
    try:
        datastore = InMemoryDataStore()
        print("   ✅ DataStore created successfully")
    except Exception as e:
        print(f"   ❌ DataStore creation failed: {e}")
        issues.append(("datastore_creation", str(e)))
        return issues
    
    # Test 3: Agent Creation
    print("3. Testing agent creation...")
    try:
        agent = MainAgent(config=config, datastore=datastore)
        print("   ✅ Agent created successfully")
    except Exception as e:
        print(f"   ❌ Agent creation failed: {e}")
        issues.append(("agent_creation", str(e)))
        return issues
    
    # Test 4: Agent Initialization
    print("4. Testing agent initialization...")
    try:
        result = agent.initialize()
        if result:
            print("   ✅ Agent initialized successfully")
        else:
            print("   ❌ Agent initialization returned False")
            issues.append(("agent_init", "Initialization returned False"))
    except Exception as e:
        print(f"   ❌ Agent initialization failed: {e}")
        issues.append(("agent_init", str(e)))
    
    # Test 5: Check for missing attributes
    print("\n5. Checking for missing attributes...")
    missing_attrs = []
    expected_attrs = ['l1_embedder', 'l1_error_monitor', 'l2_memory', 'l3_graph', 'l4_llm']
    
    for attr in expected_attrs:
        if not hasattr(agent, attr):
            missing_attrs.append(attr)
            print(f"   ❌ Missing attribute: {attr}")
        else:
            print(f"   ✅ Has attribute: {attr}")
    
    if missing_attrs:
        issues.append(("missing_attributes", missing_attrs))
    
    # Test 6: Basic Operations
    print("\n6. Testing basic operations...")
    
    # Test add_knowledge
    try:
        result = agent.add_knowledge("Test knowledge")
        if result.get("success"):
            print("   ✅ add_knowledge succeeded")
        else:
            print(f"   ❌ add_knowledge failed: {result.get('error', 'Unknown error')}")
            issues.append(("add_knowledge", result.get('error', 'Unknown error')))
    except Exception as e:
        print(f"   ❌ add_knowledge exception: {e}")
        issues.append(("add_knowledge", str(e)))
    
    # Test process_question
    try:
        answer = agent.process_question("What is test?", max_cycles=1, verbose=False)
        if hasattr(answer, 'success') and answer.success:
            print("   ✅ process_question succeeded")
        else:
            print(f"   ❌ process_question failed")
            issues.append(("process_question", "Failed to process"))
    except Exception as e:
        print(f"   ❌ process_question exception: {e}")
        issues.append(("process_question", str(e)))
    
    return issues

def test_layer_specific_issues():
    """Test each layer to identify specific problems"""
    print("\n\n=== BOTTLENECK TEST: Layer-Specific Issues ===\n")
    
    issues = []
    
    config = ConfigPresets.development()
    datastore = InMemoryDataStore()
    agent = MainAgent(config=config, datastore=datastore)
    
    # Don't initialize fully, test components individually
    
    # Test L2 Memory Manager
    print("Testing L2 Memory Manager...")
    try:
        from insightspike.implementations.layers.layer2_compatibility import CompatibleL2MemoryManager
        l2_config = type('Config', (), {
            'embedding_dim': 384,
            'enable_graph_memory': False,
            'graph_search_config': None
        })()
        l2 = CompatibleL2MemoryManager(dim=384, config=l2_config)
        
        # Check for _encode_text method
        if hasattr(l2, '_encode_text'):
            print("   ✅ L2 has _encode_text method")
        else:
            print("   ❌ L2 missing _encode_text method")
            issues.append(("l2_memory", "Missing _encode_text method"))
            
    except Exception as e:
        print(f"   ❌ L2 Memory test failed: {e}")
        issues.append(("l2_memory", str(e)))
    
    # Test L3 Graph Reasoner
    print("\nTesting L3 Graph Reasoner...")
    try:
        from insightspike.implementations.layers.layer3_graph_reasoner import L3GraphReasoner
        # Test with minimal config
        l3 = L3GraphReasoner(config={'graph': {'similarity_threshold': 0.7}})
        print("   ✅ L3 Graph Reasoner created")
    except Exception as e:
        print(f"   ❌ L3 Graph test failed: {e}")
        issues.append(("l3_graph", str(e)))
    
    # Test GraphAnalyzer
    print("\nTesting GraphAnalyzer...")
    try:
        from insightspike.features.graph_reasoning.graph_analyzer import GraphAnalyzer
        import networkx as nx
        
        analyzer = GraphAnalyzer()
        G = nx.Graph()
        G.add_node(0, features=[0.1] * 384)
        
        # Test if it handles NetworkX graphs
        try:
            metrics = analyzer.calculate_metrics(G, None, None, None)
            print("   ✅ GraphAnalyzer handles NetworkX graphs")
        except Exception as e:
            print(f"   ❌ GraphAnalyzer NetworkX error: {e}")
            issues.append(("graph_analyzer", f"NetworkX handling: {e}"))
            
    except Exception as e:
        print(f"   ❌ GraphAnalyzer test failed: {e}")
        issues.append(("graph_analyzer", str(e)))
    
    return issues

def main():
    """Run bottleneck tests"""
    print("INSIGHTSPIKE PIPELINE BOTTLENECK IDENTIFICATION")
    print("=" * 50)
    
    # Run basic initialization test
    basic_issues = test_basic_initialization()
    
    # Run layer-specific tests
    layer_issues = test_layer_specific_issues()
    
    # Summary
    print("\n\n=== BOTTLENECK SUMMARY ===")
    print("\nCritical Issues Found:")
    
    all_issues = basic_issues + layer_issues
    
    if not all_issues:
        print("✅ No critical issues found!")
    else:
        for category, issue in all_issues:
            print(f"\n{category}:")
            print(f"  - {issue}")
    
    print("\n\nKey Bottlenecks:")
    print("1. Missing l1_embedder attribute in MainAgent")
    print("2. Missing _encode_text method in L2MemoryManager")
    print("3. Graph type mismatches (NetworkX vs PyTorch Geometric)")
    print("4. Config access inconsistencies (dict vs object)")
    print("5. Patch system not fully effective")
    
    print("\n\nRecommended Actions:")
    print("1. Add l1_embedder initialization in MainAgent.__init__")
    print("2. Implement _encode_text in CompatibleL2MemoryManager")
    print("3. Standardize on single graph representation")
    print("4. Use consistent config access pattern")
    print("5. Consider removing patch system in favor of proper fixes")

if __name__ == "__main__":
    main()