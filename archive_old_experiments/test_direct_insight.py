#!/usr/bin/env python3
"""
Direct test of InsightFactRegistry functionality
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from insightspike.detection.insight_registry import InsightFactRegistry, InsightFact
from insightspike.layer1_error_monitor import analyze_input
import time

class MockL1Analysis:
    """Mock Layer1 analysis for testing"""
    def __init__(self):
        self.known_elements = ["quantum", "information"]
        self.unknown_elements = ["entanglement", "theory"]
        self.certainty_scores = {"quantum": 0.8, "information": 0.7, "entanglement": 0.3, "theory": 0.4}
        self.query_complexity = 0.7
        self.requires_synthesis = True
        self.error_threshold = 0.5
        self.analysis_confidence = 0.6

def test_direct_insight_extraction():
    """Test insight extraction directly"""
    print("🧪 Direct InsightFactRegistry Test")
    print("=" * 40)
    
    # Initialize registry
    registry = InsightFactRegistry()
    print(f"Registry initialized with {len(registry.insights)} insights")
    
    # Test data
    question = "How does quantum entanglement relate to information theory?"
    response = """
    Quantum entanglement fundamentally connects to information theory through several key insights:
    
    1. Entanglement creates correlations that transcend classical information limits
    2. The measurement of one entangled particle instantly affects its partner, demonstrating non-local information transfer
    3. This phenomenon essentially bridges quantum mechanics and information science
    4. By combining quantum states with information encoding, we get quantum computing capabilities
    5. Entanglement resembles a distributed information network where particles share quantum states
    """
    
    l1_analysis = MockL1Analysis()
    reasoning_quality = 0.8
    
    print(f"\nQuestion: {question}")
    print(f"Response length: {len(response)} characters")
    print(f"Reasoning quality: {reasoning_quality}")
    
    # Extract insights
    print(f"\n🔍 Extracting insights...")
    try:
        insights = registry.extract_insights_from_response(
            question=question,
            response=response,
            l1_analysis=l1_analysis,
            reasoning_quality=reasoning_quality
        )
        
        print(f"✅ Extraction successful!")
        print(f"📊 Results: {len(insights)} insights discovered")
        
        for i, insight in enumerate(insights, 1):
            print(f"\n🧠 Insight {i}:")
            print(f"   Type: {insight.relationship_type}")
            print(f"   Text: {insight.text}")
            print(f"   Quality: {insight.quality_score:.3f}")
            print(f"   Confidence: {insight.confidence:.3f}")
            print(f"   Source concepts: {insight.source_concepts}")
            print(f"   Target concepts: {insight.target_concepts}")
            print(f"   GED optimization: {insight.ged_optimization:.3f}")
            print(f"   IG improvement: {insight.ig_improvement:.3f}")
            
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test registry status
    print(f"\n📈 Registry Status:")
    print(f"Total insights: {len(registry.insights)}")
    
    stats = registry.get_optimization_stats()
    print(f"Statistics: {stats}")
    
    # Test search functionality
    print(f"\n🔍 Testing search functionality:")
    search_concepts = ["quantum", "information", "entanglement"]
    for concept in search_concepts:
        relevant = registry.find_relevant_insights([concept], limit=3)
        print(f"   '{concept}': {len(relevant)} relevant insights")
    
    # Test manual insight creation
    print(f"\n✏️  Testing manual insight creation:")
    manual_insight = InsightFact(
        id="test_manual_001",
        text="Machine learning algorithms mimic neural plasticity in biological systems",
        source_concepts=["machine learning", "algorithms"],
        target_concepts=["neural plasticity", "biological systems"],
        confidence=0.9,
        quality_score=0.8,
        ged_optimization=0.2,
        ig_improvement=0.15,
        discovery_context="Manual test creation",
        generated_at=time.time(),
        validation_status='pending',
        relationship_type='analogical'
    )
    
    success = registry.register_insight(manual_insight)
    print(f"Manual insight registration: {'✅ Success' if success else '❌ Failed'}")
    
    print(f"\n📊 Final Registry Status:")
    print(f"Total insights: {len(registry.insights)}")
    
    return len(registry.insights)

def test_cli_integration():
    """Test CLI integration"""
    print(f"\n🖥️  Testing CLI Integration")
    print("=" * 30)
    
    import subprocess
    
    # Test insights command
    try:
        result = subprocess.run([
            sys.executable, "-c", 
            "import sys; sys.path.insert(0, 'src'); from insightspike.cli import app; app(['insights'])"
        ], cwd=Path(__file__).parent.parent, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ CLI insights command works")
        else:
            print(f"❌ CLI insights command failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ CLI test failed: {e}")

if __name__ == "__main__":
    try:
        insights_count = test_direct_insight_extraction()
        test_cli_integration()
        
        print(f"\n🎉 Direct test completed!")
        print(f"📊 Final result: {insights_count} insights in registry")
        
        if insights_count > 0:
            print(f"✅ InsightFactRegistry is working correctly!")
            print(f"📋 Try these commands:")
            print(f"   - poetry run insightspike insights")
            print(f"   - poetry run insightspike insights-search quantum")
        else:
            print(f"⚠️  No insights were created - check configuration")
            
    except Exception as e:
        print(f"❌ Direct test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
