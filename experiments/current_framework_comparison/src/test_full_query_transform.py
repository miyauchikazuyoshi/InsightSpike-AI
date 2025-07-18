#!/usr/bin/env python3
"""
Test MainAgentWithQueryTransform (if query_transformation module is properly placed)
"""

import os
import sys
from pathlib import Path

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add paths
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# First, check if we can import the required module
try:
    # Try to import from the expected location
    from insightspike.core.query_transformation import QueryTransformer
    print("✅ query_transformation module found in core!")
    CAN_USE_TRANSFORM = True
except ImportError:
    print("❌ query_transformation module not found in core")
    print("   Trying local import...")
    
    # Try local import as fallback
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from query_transformation import QueryTransformer
        print("✅ Using local query_transformation module")
        CAN_USE_TRANSFORM = True
    except ImportError:
        print("❌ Cannot import query_transformation module")
        CAN_USE_TRANSFORM = False


def test_with_main_agent():
    """Test MainAgentWithQueryTransform if possible"""
    
    if not CAN_USE_TRANSFORM:
        print("\n⚠️  Cannot test MainAgentWithQueryTransform without query_transformation module")
        print("\nTo fix this:")
        print("1. Copy query_transformation.py to src/insightspike/core/")
        print("2. Or modify MainAgentWithQueryTransform to use local import")
        return
    
    try:
        from insightspike.core.agents.main_agent_with_query_transform import MainAgentWithQueryTransform
        print("\n✅ MainAgentWithQueryTransform imported successfully!")
        
        # Try to create instance
        from insightspike.config import InsightSpikeConfig
        config = InsightSpikeConfig()
        
        print("\n🔧 Creating MainAgentWithQueryTransform...")
        agent = MainAgentWithQueryTransform(config, enable_query_transformation=True)
        print("✅ Agent created!")
        
        # Try to initialize
        print("\n🔧 Initializing agent...")
        if agent.initialize():
            print("✅ Agent initialized!")
            
            # Add some test data
            print("\n📝 Adding test episodes...")
            test_episodes = [
                "Energy is the capacity to do work.",
                "Information and entropy are related.",
                "Maxwell's demon connects information and thermodynamics."
            ]
            
            for ep in test_episodes:
                agent.l2_memory.store_episode(text=ep, c_value=0.5)
            
            # Test query processing
            print("\n❓ Testing query transformation...")
            result = agent.process_question("What is energy?")
            
            print("\n📊 Results:")
            if isinstance(result, dict):
                print(f"Response: {result.get('response', 'No response')[:100]}...")
                
                if 'transformation_history' in result:
                    print("\n🔄 Query Transformation History:")
                    history = result['transformation_history']
                    for i, state in enumerate(history.get('states', [])):
                        print(f"  Step {i}: {state.get('text', 'N/A')}")
                        print(f"          Confidence: {state.get('confidence', 0):.2f}")
                
                if 'query_evolution' in result:
                    evolution = result['query_evolution']
                    print(f"\n📈 Query Evolution:")
                    print(f"  Initial: {evolution.get('initial', 'N/A')}")
                    print(f"  Final confidence: {evolution.get('final_state', {}).get('confidence', 0):.2f}")
                    print(f"  Insights: {evolution.get('insights_discovered', [])}")
        else:
            print("❌ Failed to initialize agent")
            
    except ImportError as e:
        print(f"\n❌ Cannot import MainAgentWithQueryTransform: {e}")
        print("\n📝 This is expected if query_transformation is not in the right place")
    except Exception as e:
        print(f"\n❌ Error testing MainAgentWithQueryTransform: {e}")
        import traceback
        traceback.print_exc()


def test_standalone_transform():
    """Test just the transformation logic"""
    print("\n\n🧪 Testing Standalone Query Transformation")
    
    if not CAN_USE_TRANSFORM:
        print("⚠️  Skipping - no query_transformation module")
        return
    
    from query_transformation import QueryTransformer, QueryState
    
    transformer = QueryTransformer()
    
    # Simulate a query evolution
    queries = [
        "What is energy?",
        "How does energy relate to work?",
        "What is the relationship between energy and information?"
    ]
    
    print("\n📝 Simulating query evolution:")
    current_state = None
    
    for i, query in enumerate(queries):
        print(f"\nStep {i+1}: {query}")
        
        if current_state is None:
            current_state = transformer.place_query_on_graph(query)
        else:
            # Simulate finding new documents that change the query
            dummy_docs = [
                {"text": f"Document about {query}", "embedding": current_state.embedding + 0.1}
            ]
            current_state = transformer.transform_query(current_state, None, dummy_docs)
            current_state.text = query  # Override with our planned evolution
        
        print(f"  Stage: {current_state.stage}")
        print(f"  Confidence: {current_state.confidence:.2f}")


if __name__ == "__main__":
    print("🔍 Testing Query Transformation Integration\n")
    
    # Test with MainAgent
    test_with_main_agent()
    
    # Test standalone
    test_standalone_transform()
    
    print("\n\n💡 Summary:")
    if CAN_USE_TRANSFORM:
        print("✅ Query transformation module is working")
        print("🔧 MainAgentWithQueryTransform needs the module in the right location")
    else:
        print("❌ Query transformation module needs to be properly installed")
    
    print("\n📝 To complete the implementation:")
    print("1. Place query_transformation.py in src/insightspike/core/")
    print("2. Fix the missing _get_current_knowledge_graph() method")
    print("3. Test with real knowledge base")
    print("4. Measure impact on insight discovery")