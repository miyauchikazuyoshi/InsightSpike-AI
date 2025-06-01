#!/usr/bin/env python3
"""
Test script for InsightFactRegistry integration with agent_loop
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from insightspike.detection.insight_registry import InsightFactRegistry
from insightspike import agent_loop
from insightspike.unknown_learner import UnknownLearner
from insightspike.layer1_error_monitor import analyze_input

class MockMemory:
    """Mock memory for testing"""
    def __init__(self):
        self.episodes = []

def test_insight_integration():
    """Test insight extraction and registration in agent loop"""
    print("🧪 Testing InsightFactRegistry Integration")
    print("=" * 50)
    
    # Initialize components
    registry = InsightFactRegistry()
    memory = MockMemory()
    
    # Test questions that should generate insights
    test_questions = [
        "How does quantum entanglement relate to information theory?",
        "What is the connection between neural networks and human cognition?",
        "Why do complex systems exhibit emergent behavior?",
        "How can we synthesize biological and artificial intelligence?",
    ]
    
    print(f"Initial insights in registry: {len(registry.insights)}")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Test {i}: {question}")
        
        try:
            # Run cycle with insight extraction
            result = agent_loop.cycle(
                memory=memory,
                question=question,
                top_k=10
            )
            
            # Check results
            if 'discovered_insights' in result:
                insights_count = result.get('insight_count', 0)
                avg_quality = result.get('insight_quality_avg', 0.0)
                
                print(f"   ✅ Processing successful")
                print(f"   🧠 Insights discovered: {insights_count}")
                print(f"   📊 Average quality: {avg_quality:.3f}")
                print(f"   💡 Response quality: {result.get('reasoning_quality', 0.0):.3f}")
                
                if insights_count > 0:
                    print(f"   🎯 New insights registered!")
                    for insight in result['discovered_insights']:
                        print(f"      - {insight.relationship_type}: {insight.text[:80]}...")
                
            else:
                print(f"   ❌ No insight extraction occurred")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # Final statistics
    print(f"\n📊 Final Statistics:")
    print(f"Total insights in registry: {len(registry.insights)}")
    
    stats = registry.get_optimization_stats()
    print(f"Average quality: {stats.get('avg_quality_score', 0.0):.3f}")
    print(f"Average GED improvement: {stats.get('avg_ged_improvement', 0.0):.3f}")
    print(f"Average IG improvement: {stats.get('avg_ig_improvement', 0.0):.3f}")
    
    # Show recent insights
    if registry.insights:
        print(f"\n🧠 Recent Insights:")
        recent = sorted(registry.insights.values(), key=lambda x: x.generated_at, reverse=True)[:3]
        for insight in recent:
            print(f"   - {insight.relationship_type}: {insight.text[:60]}...")
            print(f"     Quality: {insight.quality_score:.3f}, GED: {insight.ged_optimization:.3f}")
    
    print(f"\n✅ Integration test completed!")

def test_cli_commands():
    """Test CLI insight commands"""
    print(f"\n🖥️  Testing CLI Commands")
    print("=" * 30)
    
    # Test insight listing
    print("Testing 'insights' command...")
    try:
        registry = InsightFactRegistry()
        print(f"✅ Registry access successful ({len(registry.insights)} insights)")
    except Exception as e:
        print(f"❌ Registry access failed: {e}")
    
    # Test insight search
    print("Testing insight search...")
    try:
        relevant = registry.find_relevant_insights(["quantum", "neural"], limit=5)
        print(f"✅ Search successful ({len(relevant)} relevant insights)")
    except Exception as e:
        print(f"❌ Search failed: {e}")

if __name__ == "__main__":
    try:
        test_insight_integration()
        test_cli_commands()
        
        print(f"\n🎉 All tests completed!")
        print(f"📋 Next steps:")
        print(f"   - Run: poetry run insightspike insights")
        print(f"   - Run: poetry run insightspike ask \"How do quantum effects relate to consciousness?\"")
        print(f"   - Run: poetry run insightspike insights-search consciousness")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        sys.exit(1)
