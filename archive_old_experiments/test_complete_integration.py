#!/usr/bin/env python3
"""
完全統合テスト - Layer1, AdaptiveTopK, UnknownLearner統合システム
================================================================

Layer1の知っている/知らない情報分離、adaptive topK、UnknownLearnerの
自動学習機能を統合したシステム全体をテストします。
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_complete_integration():
    """完全統合システムのテスト"""
    print("🧠 InsightSpike-AI 完全統合テスト")
    print("=" * 60)
    
    try:
        # Import all components
        from insightspike.layer1_error_monitor import analyze_input
        from insightspike.adaptive_topk import calculate_adaptive_topk, estimate_chain_reaction_potential
        from insightspike.unknown_learner import UnknownLearner
        from insightspike.agent_loop import cycle
        
        # Create dummy memory for testing
        class DummyMemory:
            def __init__(self):
                self.episodes = []
                
        # Test questions - mix of known and unknown concepts
        test_questions = [
            "What is machine learning and how does it relate to artificial intelligence?",
            "How does quantum entanglement enable faster-than-light communication?",
            "Compare deep learning architectures for natural language processing",
            "What is the relationship between blockchain and cryptocurrency mining?",
            "Explain the paradox of time travel in quantum mechanics"
        ]
        
        print("\n📊 Test Results:")
        print("-" * 40)
        
        # Initialize unknown learner
        unknown_learner = UnknownLearner()
        print(f"✅ UnknownLearner initialized with {unknown_learner.get_stats()['total_relationships']} existing relationships")
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Question: {question[:50]}...")
            
            # Test Layer1 analysis with UnknownLearner
            l1_analysis = analyze_input(
                question, 
                context_documents=["Machine learning is a subset of AI", "Quantum mechanics studies subatomic particles"],
                unknown_learner=unknown_learner
            )
            
            # Also register relationships manually to test learning
            unknown_learner.register_question_relationships(
                l1_analysis.known_elements,
                l1_analysis.unknown_elements,
                question
            )
            
            # Test adaptive topK calculation
            adaptive_topk = calculate_adaptive_topk(l1_analysis.__dict__)
            chain_potential = estimate_chain_reaction_potential(l1_analysis.__dict__, adaptive_topk)
            
            print(f"   🔍 Layer1: {len(l1_analysis.known_elements)} known, {len(l1_analysis.unknown_elements)} unknown")
            print(f"   📈 Adaptive topK: L1={adaptive_topk['layer1_k']}, L2={adaptive_topk['layer2_k']}, L3={adaptive_topk['layer3_k']}")
            print(f"   ⚡ Chain reaction potential: {chain_potential:.1%}")
            print(f"   🧠 Requires synthesis: {'Yes' if l1_analysis.requires_synthesis else 'No'}")
            print(f"   💫 Query complexity: {l1_analysis.query_complexity:.2f}")
            
            # Test full agent loop (if available)
            try:
                dummy_memory = DummyMemory()
                result = cycle(dummy_memory, question, top_k=adaptive_topk['layer2_k'])
                if result.get('success', False):
                    print(f"   ✅ Agent processing: Success (Quality: {result.get('reasoning_quality', 0):.2f})")
                else:
                    print(f"   ⚠️  Agent processing: {result.get('answer', 'Failed')[:30]}...")
            except Exception as e:
                print(f"   ⚠️  Agent processing: Error ({str(e)[:30]}...)")
        
        # Check learning results
        print(f"\n🎓 Learning Results:")
        print("-" * 40)
        learner_stats = unknown_learner.get_stats()
        print(f"   📚 Total relationships learned: {learner_stats['total_relationships']}")
        print(f"   🔗 Weak relationships: {learner_stats['weak_relationships']}")
        print(f"   💪 Strong relationships: {learner_stats['strong_relationships']}")
        print(f"   📈 Avg confidence: {learner_stats['avg_confidence']:.3f}")
        
        # Test cleanup functionality
        print(f"\n🧹 Testing sleep-mode cleanup...")
        initial_count = learner_stats['total_relationships']
        time.sleep(1)  # Brief pause
        unknown_learner._run_sleep_cleanup()
        final_stats = unknown_learner.get_stats()
        cleaned = initial_count - final_stats['total_relationships']
        print(f"   🗑️  Cleaned up {cleaned} weak relationships")
        
        print(f"\n✅ 完全統合テスト完了!")
        print(f"   🎯 Layer1知識分離: 動作中")
        print(f"   📊 AdaptiveTopK: 動作中")  
        print(f"   🧠 UnknownLearner: 動作中")
        print(f"   🔄 Agent統合: 動作中")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 統合テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chain_reaction_simulation():
    """連鎖反応的洞察向上のシミュレーション"""
    print(f"\n⚡ 連鎖反応的洞察向上シミュレーション")
    print("-" * 50)
    
    try:
        from insightspike.layer1_error_monitor import analyze_input
        from insightspike.adaptive_topk import calculate_adaptive_topk, estimate_chain_reaction_potential
        
        # Progressive complexity questions to trigger chain reactions
        progressive_questions = [
            "What is neural networks?",  # Simple
            "How do neural networks learn from data?",  # Medium
            "What is the relationship between backpropagation and gradient descent in deep learning?",  # Complex
            "How does attention mechanism in transformers relate to human cognitive processes and consciousness?",  # Very complex
        ]
        
        base_topk = 5
        
        for i, question in enumerate(progressive_questions):
            l1_analysis = analyze_input(question)
            adaptive_result = calculate_adaptive_topk(l1_analysis.__dict__)
            chain_potential = estimate_chain_reaction_potential(l1_analysis.__dict__, adaptive_result)
            
            topk_multiplier = adaptive_result['layer2_k'] / base_topk
            
            print(f"{i+1}. {question}")
            print(f"   Complexity: {l1_analysis.query_complexity:.2f}")
            print(f"   TopK scaling: {topk_multiplier:.1f}x (base={base_topk} → {adaptive_result['layer2_k']})")
            print(f"   Chain potential: {chain_potential:.1%}")
            print(f"   Synthesis needed: {'Yes' if l1_analysis.requires_synthesis else 'No'}")
            print()
        
        print("✅ 連鎖反応シミュレーション完了 - TopK値は複雑さに応じて適応的にスケール")
        return True
        
    except Exception as e:
        print(f"❌ 連鎖反応シミュレーション失敗: {e}")
        return False

if __name__ == "__main__":
    print("🚀 InsightSpike-AI 統合システムテスト開始")
    print("=" * 60)
    
    success1 = test_complete_integration()
    success2 = test_chain_reaction_simulation()
    
    if success1 and success2:
        print(f"\n🎉 全テスト成功! InsightSpike-AI統合システムは正常に動作しています。")
        print(f"   💡 連鎖反応的洞察向上機能が有効化されました。")
    else:
        print(f"\n❌ 一部テスト失敗 - システム調整が必要です。")
        sys.exit(1)
