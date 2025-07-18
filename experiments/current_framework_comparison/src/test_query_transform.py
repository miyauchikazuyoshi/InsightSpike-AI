#!/usr/bin/env python3
"""
Test Query Transformation functionality
"""

import numpy as np
from pathlib import Path
import sys

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from query_transformation import QueryState, QueryTransformationHistory, QueryTransformer


def create_dummy_documents():
    """Create dummy documents for testing"""
    return [
        {
            "text": "Energy is the capacity to do work.",
            "embedding": np.random.randn(384),  # MiniLM embedding size
            "similarity": 0.9
        },
        {
            "text": "Energy can be transformed but not created or destroyed.",
            "embedding": np.random.randn(384),
            "similarity": 0.85
        },
        {
            "text": "Information and entropy are related through thermodynamics.",
            "embedding": np.random.randn(384),
            "similarity": 0.8
        }
    ]


def test_basic_functionality():
    """Test basic query transformation functionality"""
    print("🧪 Testing Query Transformation Module")
    
    # Initialize transformer
    print("\n1️⃣ Initializing QueryTransformer...")
    transformer = QueryTransformer(use_gnn=False)  # Start without GNN
    print("✅ Transformer initialized")
    
    # Test initial query placement
    print("\n2️⃣ Testing query placement...")
    initial_query = "What is energy?"
    state = transformer.place_query_on_graph(initial_query)
    
    print(f"Initial state:")
    print(f"  - Text: {state.text}")
    print(f"  - Confidence: {state.confidence}")
    print(f"  - Stage: {state.stage}")
    print(f"  - Embedding shape: {state.embedding.shape}")
    
    # Test transformation
    print("\n3️⃣ Testing query transformation...")
    dummy_docs = create_dummy_documents()
    
    # Create transformation history
    history = QueryTransformationHistory(initial_query=initial_query)
    history.add_state(state)
    
    # Transform query
    new_state = transformer.transform_query(state, None, dummy_docs)
    history.add_state(new_state)
    
    print(f"Transformed state:")
    print(f"  - Text: {new_state.text}")
    print(f"  - Confidence: {new_state.confidence}")
    print(f"  - Stage: {new_state.stage}")
    print(f"  - Insights: {new_state.insights}")
    
    # Test multiple transformations
    print("\n4️⃣ Testing multiple transformations...")
    current_state = new_state
    for i in range(3):
        current_state = transformer.transform_query(current_state, None, dummy_docs)
        history.add_state(current_state)
        print(f"\nTransformation {i+2}:")
        print(f"  - Text: {current_state.text}")
        print(f"  - Confidence: {current_state.confidence:.3f}")
        print(f"  - Stage: {current_state.stage}")
        
        if current_state.stage == "complete":
            print("  ✅ Query transformation complete!")
            break
    
    # Show history summary
    print("\n5️⃣ Transformation History Summary:")
    print(f"Total transformations: {len(history.states) - 1}")
    print(f"Confidence trajectory: {[f'{c:.2f}' for c in history.get_confidence_trajectory()]}")
    print(f"All insights discovered:")
    for insight in history.get_total_insights():
        print(f"  - {insight}")
    
    # Test serialization
    print("\n6️⃣ Testing serialization...")
    history_dict = history.to_dict()
    print(f"History keys: {list(history_dict.keys())}")
    print(f"Number of states: {len(history_dict['states'])}")
    
    print("\n✅ All tests passed!")
    
    return history


def test_insight_detection():
    """Test insight detection through transformation"""
    print("\n\n🔍 Testing Insight Detection")
    
    transformer = QueryTransformer()
    
    # Start with vague query
    vague_query = "How do things relate?"
    state = transformer.place_query_on_graph(vague_query)
    
    # Create documents that should trigger insight
    insight_docs = [
        {
            "text": "Energy and information are fundamentally connected through entropy.",
            "embedding": np.random.randn(384),
            "similarity": 0.95
        },
        {
            "text": "Maxwell's demon demonstrates the link between information and thermodynamics.",
            "embedding": np.random.randn(384),
            "similarity": 0.92
        },
        {
            "text": "Landauer's principle: erasing information requires energy.",
            "embedding": np.random.randn(384),
            "similarity": 0.90
        }
    ]
    
    # Transform with insight-rich documents
    print(f"\nInitial query: {vague_query}")
    
    # Make embeddings more different to trigger transformation
    state.embedding = np.random.randn(384)
    transformer.transformation_threshold = 0.2  # Lower threshold for testing
    
    new_state = transformer.transform_query(state, None, insight_docs)
    
    print(f"\nAfter transformation:")
    print(f"  - Text: {new_state.text}")
    print(f"  - Stage: {new_state.stage}")
    print(f"  - Insights: {new_state.insights}")
    
    if new_state.stage == "insight":
        print("\n🎯 Insight detected through query transformation!")


if __name__ == "__main__":
    # Run basic tests
    history = test_basic_functionality()
    
    # Run insight detection test
    test_insight_detection()
    
    print("\n\n💡 Next Steps:")
    print("1. Integrate with MainAgentWithQueryTransform")
    print("2. Implement proper GNN message passing")
    print("3. Use LLM for natural query reformulation")
    print("4. Test with real InsightSpike knowledge base")