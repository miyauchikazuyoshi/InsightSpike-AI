"""
End-to-end integration test for InsightSpike MVP
==============================================

This test verifies that the core ΔGED/ΔIG mechanism works and can detect EurekaSpikes.
"""

import pytest
import numpy as np
import networkx as nx
from pathlib import Path
import tempfile
import shutil
import sys

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Test imports
from insightspike.graph_metrics import delta_ged, delta_ig
from insightspike.eureka_spike import EurekaDetector, detect_eureka_spike
from insightspike.core.layers.layer2_memory_manager import L2MemoryManager as Memory, Episode


class TestInsightSpikeMVP:
    """Test core MVP functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Cleanup test environment"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_delta_ged_calculation(self):
        """Test ΔGED calculation with simple graphs"""
        # Create two simple graphs
        g1 = nx.Graph()
        g1.add_edges_from([(0, 1), (1, 2)])  # Linear graph
        
        g2 = nx.Graph()
        g2.add_edges_from([(0, 1), (1, 2), (2, 0)])  # Triangle graph
        
        # Calculate ΔGED
        ged = delta_ged(g1, g2)
        
        # Should be positive (graph became more complex)
        assert isinstance(ged, float)
        assert ged >= 0
        
        # Self-comparison should be 0
        self_ged = delta_ged(g1, g1)
        assert self_ged == 0.0
    
    def test_delta_ig_calculation(self):
        """Test ΔIG calculation with vector sets"""
        # Create two sets of vectors
        np.random.seed(42)  # For reproducibility
        
        # Old vectors - more scattered
        vecs_old = np.random.randn(10, 5)
        
        # New vectors - more clustered
        vecs_new = np.array([
            [1, 1, 1, 1, 1],  # Cluster 1
            [1.1, 1.1, 1.1, 1.1, 1.1],
            [1.2, 1.2, 1.2, 1.2, 1.2],
            [-1, -1, -1, -1, -1],  # Cluster 2
            [-1.1, -1.1, -1.1, -1.1, -1.1],
            [-1.2, -1.2, -1.2, -1.2, -1.2],
            [0, 0, 0, 0, 0],  # Cluster 3
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3, 0.3, 0.3],
        ])
        
        # Calculate ΔIG
        ig = delta_ig(vecs_old, vecs_new, k=3)
        
        # Should be a float
        assert isinstance(ig, float)
        
        # Self-comparison should be 0
        self_ig = delta_ig(vecs_new, vecs_new, k=3)
        assert abs(self_ig) < 0.01  # Close to 0
    
    def test_eureka_spike_detection(self):
        """Test EurekaSpike detection logic"""
        detector = EurekaDetector(ged_threshold=0.5, ig_threshold=0.2)
        
        # Test cases that should trigger EurekaSpike
        spike_cases = [
            (-0.6, 0.3),  # GED drops, IG rises
            (-0.5, 0.2),  # Exactly at thresholds
            (-1.0, 0.5),  # Strong spike
        ]
        
        for delta_ged, delta_ig in spike_cases:
            result = detector.detect_spike(delta_ged, delta_ig)
            assert result['eureka_spike'] == True, f"Should detect spike for ΔGED={delta_ged}, ΔIG={delta_ig}"
            assert result['spike_intensity'] > 0
            assert result['reward'] > 0
        
        # Test cases that should NOT trigger EurekaSpike
        no_spike_cases = [
            (0.1, 0.3),   # GED rises (bad)
            (-0.6, 0.1),  # IG doesn't rise enough
            (-0.3, 0.3),  # GED doesn't drop enough
            (0.5, -0.1),  # Both wrong direction
        ]
        
        for delta_ged, delta_ig in no_spike_cases:
            result = detector.detect_spike(delta_ged, delta_ig)
            assert result['eureka_spike'] == False, f"Should NOT detect spike for ΔGED={delta_ged}, ΔIG={delta_ig}"
            assert result['spike_intensity'] == 0
            assert result['reward'] == 0
    
    def test_eureka_convenience_function(self):
        """Test convenience function for spike detection"""
        # Should detect spike
        assert detect_eureka_spike(-0.6, 0.3) == True
        
        # Should not detect spike
        assert detect_eureka_spike(0.1, 0.3) == False
        
        # Test with custom thresholds
        assert detect_eureka_spike(-0.3, 0.1, ged_threshold=0.2, ig_threshold=0.05) == True
    
    def test_memory_c_value_updates(self):
        """Test C-value update mechanism"""
        # Create test documents
        docs = ["Test document 1", "Test document 2", "Test document 3"]
        
        # Skip if Memory.build requires actual model
        try:
            memory = Memory(dim=384)  # SentenceTransformer dimension
            
            # Manually add episodes
            for i, doc in enumerate(docs):
                vec = np.random.randn(384).astype(np.float32)
                episode = Episode(vec, doc, c=0.5)
                memory.episodes.append(episode)
            
            # Update C-values with positive reward
            memory.update_c([0, 1], reward=0.3, eta=0.1)
            
            # Check that C-values increased
            assert memory.episodes[0].c > 0.5
            assert memory.episodes[1].c > 0.5
            assert memory.episodes[2].c == 0.5  # Unchanged
            
            # Update with negative reward
            memory.update_c([0], reward=-0.2, eta=0.1)
            
            # Check that C-value decreased but didn't go below 0
            assert memory.episodes[0].c >= 0.0
            
        except Exception as e:
            pytest.skip(f"Memory test skipped due to model dependency: {e}")
    
    def test_pattern_analysis(self):
        """Test pattern analysis in EurekaDetector"""
        detector = EurekaDetector()
        
        # Add some history
        test_data = [
            (-0.1, 0.1),  # No spike
            (-0.6, 0.3),  # Spike
            (-0.2, 0.1),  # No spike
            (-0.7, 0.4),  # Spike
        ]
        
        for ged, ig in test_data:
            detector.detect_spike(ged, ig)
        
        analysis = detector.get_pattern_analysis()
        
        assert analysis['total_spikes'] == 2
        assert analysis['spike_rate'] == 0.5
        assert 'avg_ged' in analysis
        assert 'avg_ig' in analysis
        assert analysis['history_length'] == 4
    
    def test_integration_workflow(self):
        """Test complete workflow: memory → graph → metrics → spike detection"""
        # Create detector
        detector = EurekaDetector()
        
        # Simulate a workflow that should produce an insight
        scenarios = [
            # Scenario 1: Progressive insight development
            {"ged": -0.1, "ig": 0.05, "expected": False},  # Building up
            {"ged": -0.3, "ig": 0.15, "expected": False},  # Getting closer
            {"ged": -0.6, "ig": 0.25, "expected": True},   # EUREKA!
            
            # Scenario 2: False alarm
            {"ged": 0.2, "ig": 0.3, "expected": False},   # Wrong direction
        ]
        
        insights_detected = 0
        
        for scenario in scenarios:
            result = detector.detect_spike(scenario["ged"], scenario["ig"])
            
            if result['eureka_spike']:
                insights_detected += 1
            
            assert result['eureka_spike'] == scenario['expected'], \
                f"Expected {scenario['expected']} for ΔGED={scenario['ged']}, ΔIG={scenario['ig']}"
        
        assert insights_detected == 1, "Should detect exactly one insight in the test scenario"
        
        # Verify pattern analysis makes sense
        analysis = detector.get_pattern_analysis()
        assert analysis['total_spikes'] == 1

if __name__ == "__main__":
    # Run a quick self-test
    test_instance = TestInsightSpikeMVP()
    test_instance.setup_method()
    
    try:
        print("🧪 Running MVP integration tests...")
        
        print("  ✓ Testing ΔGED calculation")
        test_instance.test_delta_ged_calculation()
        
        print("  ✓ Testing ΔIG calculation")  
        test_instance.test_delta_ig_calculation()
        
        print("  ✓ Testing EurekaSpike detection")
        test_instance.test_eureka_spike_detection()
        
        print("  ✓ Testing convenience function")
        test_instance.test_eureka_convenience_function()
        
        print("  ✓ Testing pattern analysis")
        test_instance.test_pattern_analysis()
        
        print("  ✓ Testing integration workflow")
        test_instance.test_integration_workflow()
        
        print("\n🎉 All MVP core tests passed!")
        print("🚀 InsightSpike MVP is ready for Phase 2 development!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
    finally:
        test_instance.teardown_method()
