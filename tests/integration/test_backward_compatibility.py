"""
Backward compatibility tests for new metrics implementations
"""

import pytest
import numpy as np
import torch
import pytest
pytest.importorskip("torch_geometric")
pytest.importorskip(
    "insightspike.algorithms.normalized_ged",
    reason="Legacy module integrated into GeDIGCore; skip in cloud-safe runs.",
)
pytest.importorskip(
    "insightspike.algorithms.local_information_gain_v2",
    reason="Legacy module integrated; skip in cloud-safe runs.",
)
from torch_geometric.data import Data

from insightspike.algorithms.metrics_selector import MetricsSelector
from insightspike.algorithms.graph_structure_analyzer import GraphStructureAnalyzer
from insightspike.algorithms.normalized_ged import NormalizedGED
from insightspike.algorithms.local_information_gain_v2 import LocalInformationGainV2
from insightspike.algorithms.graph_edit_distance_fixed import compute_delta_ged_fixed


class TestBackwardCompatibility:
    """Test that new implementations maintain backward compatibility."""
    
    @pytest.fixture
    def test_graphs(self):
        """Create test graph data."""
        # Simple graph transformation
        data1 = Data(
            x=torch.randn(4, 10),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        )
        
        data2 = Data(
            x=torch.randn(5, 10),
            edge_index=torch.tensor([
                [0, 1, 2, 3, 4, 4, 4, 4],
                [1, 2, 3, 0, 0, 1, 2, 3]
            ], dtype=torch.long)
        )
        
        return data1, data2
    
    def test_ged_sign_convention(self, test_graphs):
        """Test that GED maintains sign convention for structural improvement."""
        data1, data2 = test_graphs
        
        # Old implementation
        old_result = compute_delta_ged_fixed(data1, data2)
        
        # New implementation
        new_calc = GraphStructureAnalyzer()
        new_result = new_calc.analyze_structure(data1, data2)
        
        # Check sign convention
        if old_result < 0:  # Old implementation returns negative for improvement
            assert new_result["structural_improvement"] > 0  # New returns positive
        
        # Normalized GED
        norm_calc = NormalizedGED()
        norm_result = norm_calc.calculate(data1, data2)
        
        # Structural improvement should have opposite sign to old GED
        if old_result < 0:
            assert norm_result.structural_improvement < 0
    
    def test_metrics_selector_compatibility(self, test_graphs):
        """Test MetricsSelector maintains compatibility."""
        data1, data2 = test_graphs
        
        # Default config (old implementation)
        config_old = {"graph": {}}
        selector_old = MetricsSelector(config_old)
        
        ged_old = selector_old.delta_ged(data1, data2)
        ig_old = selector_old.delta_ig(data1, data2)
        
        # New implementations
        config_new = {
            "graph": {
                "use_normalized_ged": True,
                "use_local_ig": True
            }
        }
        selector_new = MetricsSelector(config_new)
        
        ged_new = selector_new.delta_ged(data1, data2)
        ig_new = selector_new.delta_ig(data1, data2)
        
        # Both should return floats
        assert isinstance(ged_old, (int, float))
        assert isinstance(ged_new, (int, float))
        assert isinstance(ig_old, (int, float))
        assert isinstance(ig_new, (int, float))
        
        # Signs should be consistent for spike detection
        # If old detects spike, new should too (with adjusted thresholds)
        old_spike = (ged_old < -0.5) and (ig_old > 0.2)
        new_spike = (ged_new < -0.05) and (ig_new > 0.02)
        
        # Log for debugging
        print(f"\nOld: GED={ged_old:.3f}, IG={ig_old:.3f}, Spike={old_spike}")
        print(f"New: GED={ged_new:.3f}, IG={ig_new:.3f}, Spike={new_spike}")
    
    def test_value_ranges(self, test_graphs):
        """Test that new implementations produce values in expected ranges."""
        data1, data2 = test_graphs
        
        # Normalized GED should be in [-1, 1]
        norm_calc = NormalizedGED()
        norm_result = norm_calc.calculate(data1, data2)
        
        assert -1.0 <= norm_result.normalized_ged <= 1.0
        assert -1.0 <= norm_result.structural_improvement <= 1.0
        
        # Local IG should be in reasonable range
        local_calc = LocalInformationGainV2(normalize=True)
        local_result = local_calc.calculate(data1, data2)
        
        assert -1.0 <= local_result.total_ig <= 1.0
    
    def test_empty_graph_handling(self):
        """Test handling of edge cases."""
        # Empty graphs
        empty1 = Data(x=torch.zeros(0, 10), edge_index=torch.zeros(2, 0, dtype=torch.long))
        empty2 = Data(x=torch.zeros(1, 10), edge_index=torch.zeros(2, 0, dtype=torch.long))
        
        # All implementations should handle this gracefully
        configs = [
            {"graph": {}},  # Old
            {"graph": {"use_new_ged_implementation": True}},  # New GED
            {"graph": {"use_normalized_ged": True}},  # Normalized
            {"graph": {"use_local_ig": True}},  # Local IG
        ]
        
        for config in configs:
            selector = MetricsSelector(config)
            
            # Should not crash
            ged = selector.delta_ged(empty1, empty2)
            ig = selector.delta_ig(empty1, empty2)
            
            assert isinstance(ged, (int, float))
            assert isinstance(ig, (int, float))
            assert not np.isnan(ged)
            assert not np.isnan(ig)
    
    def test_flag_combinations(self, test_graphs):
        """Test different flag combinations work correctly."""
        data1, data2 = test_graphs
        
        flag_combos = [
            {},  # All default
            {"use_new_ged_implementation": True},
            {"use_normalized_ged": True},
            {"use_local_ig": True},
            {"use_normalized_ged": True, "use_local_ig": True},
            {"use_new_ged_implementation": True, "use_local_ig": True},
        ]
        
        for flags in flag_combos:
            config = {"graph": flags}
            selector = MetricsSelector(config)
            
            ged = selector.delta_ged(data1, data2)
            ig = selector.delta_ig(data1, data2)
            
            # Basic sanity checks
            assert isinstance(ged, (int, float))
            assert isinstance(ig, (int, float))
            assert -100 < ged < 100  # Reasonable range
            assert -10 < ig < 10  # Reasonable range
            
            print(f"\nFlags: {flags}")
            print(f"GED: {ged:.4f}, IG: {ig:.4f}")


def test_mainagent_compatibility():
    """Test compatibility with MainAgent usage patterns."""
    from insightspike.config.models import ExperimentConfig
    
    # Create config with new flags
    config_dict = {
        "llm_provider": "mock",
        "graph": {
            "use_normalized_ged": True,
            "use_local_ig": True
        },
        "spike_detection": {
            "ged_threshold": -0.05,  # Adjusted for normalized
            "ig_threshold": 0.02,    # Adjusted for local
        }
    }
    
    config = ExperimentConfig(**config_dict)
    
    # Create selector
    selector = MetricsSelector(config)
    
    # Test with dummy data
    data1 = Data(
        x=torch.randn(3, 384),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    )
    
    data2 = Data(
        x=torch.randn(4, 384),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    )
    
    # Should work without errors
    ged = selector.delta_ged(data1, data2)
    ig = selector.delta_ig(data1, data2)
    
    assert isinstance(ged, float)
    assert isinstance(ig, float)
    
    # Check spike detection with new thresholds
    is_spike = (ged < config_dict["spike_detection"]["ged_threshold"] and
                ig > config_dict["spike_detection"]["ig_threshold"])
    
    print(f"\nMainAgent compatibility test:")
    print(f"GED: {ged:.4f} (threshold: {config_dict['spike_detection']['ged_threshold']})")
    print(f"IG: {ig:.4f} (threshold: {config_dict['spike_detection']['ig_threshold']})")
    print(f"Spike: {is_spike}")


if __name__ == "__main__":
    # Run tests
    test = TestBackwardCompatibility()
    
    # Create test data
    data1 = Data(
        x=torch.randn(4, 10),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    )
    
    data2 = Data(
        x=torch.randn(5, 10),
        edge_index=torch.tensor([
            [0, 1, 2, 3, 4, 4, 4, 4],
            [1, 2, 3, 0, 0, 1, 2, 3]
        ], dtype=torch.long)
    )
    
    test_data = (data1, data2)
    
    print("Running backward compatibility tests...\n")
    
    test.test_metrics_selector_compatibility(test_data)
    test.test_value_ranges(test_data)
    test.test_empty_graph_handling()
    test.test_flag_combinations(test_data)
    test_mainagent_compatibility()
    
    print("\nAll backward compatibility tests passed!")
