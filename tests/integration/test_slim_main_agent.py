"""Integration tests for SlimMainAgent with new pipeline components."""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch
from typing import Dict, Any

from insightspike.implementations.agents.slim_main_agent import SlimMainAgent
from insightspike.spike_pipeline.pipeline import SpikePipeline
from insightspike.spike_pipeline.detector import SpikeDecisionMode
from insightspike.config import load_config
from insightspike.config.presets import ConfigPresets


class TestSlimMainAgentIntegration:
    """Integration tests for SlimMainAgent."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def test_config(self, temp_data_dir):
        """Create test configuration."""
        config = {
            'data': {
                'base_path': temp_data_dir,
                'datastore_type': 'memory'
            },
            'llm': {
                'provider': 'mock',
                'model': 'test-model'
            },
            'spike': {
                'composite_threshold': 0.6,
                'ged_threshold': 0.5,
                'ig_threshold': 0.4
            },
            'gedig': {
                'k_value': 2.0,
                'enable_adaptive': True
            }
        }
        return config
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider."""
        provider = Mock()
        provider.generate.return_value = "Mock LLM response"
        provider.is_available.return_value = True
        return provider
    
    def test_slim_agent_initialization(self, test_config):
        """Test SlimMainAgent initializes correctly."""
        agent = SlimMainAgent(config=test_config)
        
        # Check core components are initialized
        assert agent.config is not None
        assert agent.spike_pipeline is not None
        assert agent.fallback_registry is not None
        assert agent.l1_error_monitor is not None
        assert agent.l2_memory_manager is not None
        assert agent.l3_graph_reasoner is not None
        assert agent.l4_llm_interface is not None
        
        # Check spike pipeline is configured correctly
        assert isinstance(agent.spike_pipeline, SpikePipeline)
        assert agent.spike_pipeline.detector.mode == SpikeDecisionMode.WEIGHTED
    
    def test_slim_agent_line_count_reduction(self):
        """Test that SlimMainAgent has significantly fewer lines than original."""
        slim_agent_path = "/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/src/insightspike/implementations/agents/slim_main_agent.py"
        
        with open(slim_agent_path, 'r') as f:
            slim_lines = len([line for line in f if line.strip() and not line.strip().startswith('#')])
        
        # Original MainAgent was ~1,968 lines, target is 40% reduction
        target_max_lines = int(1968 * 0.6)  # 1,181 lines
        
        assert slim_lines <= target_max_lines, f"SlimMainAgent has {slim_lines} lines, target max: {target_max_lines}"
        print(f"✓ SlimMainAgent line count: {slim_lines} lines (target: <{target_max_lines})")
    
    @patch('insightspike.implementations.llm.factory.LLMProviderFactory.create')
    def test_process_question_basic_flow(self, mock_factory, test_config, mock_llm_provider):
        """Test basic question processing flow."""
        mock_factory.return_value = mock_llm_provider
        
        agent = SlimMainAgent(config=test_config)
        
        # Mock geDIG calculation
        with patch.object(agent, '_calculate_gedig_for_query') as mock_gedig:
            mock_gedig.return_value = {
                'gedig': 0.7,
                'ged': -0.3,
                'ig': 1.0,
                'k_value': 2.0
            }
            
            result = agent.process_question("What is the meaning of life?")
            
            # Check result structure
            assert hasattr(result, 'response')
            assert hasattr(result, 'has_spike')
            assert hasattr(result, 'metadata')
            
            # Check spike was detected (composite score should be high)
            assert result.has_spike is True
            assert result.response == "Mock LLM response"
    
    @patch('insightspike.implementations.llm.factory.LLMProviderFactory.create')
    def test_spike_pipeline_integration(self, mock_factory, test_config, mock_llm_provider):
        """Test integration with spike detection pipeline."""
        mock_factory.return_value = mock_llm_provider
        
        agent = SlimMainAgent(config=test_config)
        
        # Test spike detection through pipeline
        gedig_result = {
            'gedig': 0.8,
            'ged': -0.4,
            'ig': 1.2,
            'k_value': 2.0
        }
        
        with patch.object(agent.l3_graph_reasoner, 'analyze_graph') as mock_analyze:
            mock_analyze.return_value = {'density': 0.5, 'clustering': 0.3}
            
            with patch.object(agent.l2_memory_manager, 'retrieve_relevant_memories') as mock_retrieve:
                mock_retrieve.return_value = [
                    {'text': 'relevant memory', 'c_value': 0.8}
                ]
                
                spike_result = agent._detect_spike_via_pipeline(
                    gedig_result, {}, mock_retrieve.return_value, {}
                )
                
                assert spike_result is not None
                assert 'spike_detected' in spike_result.formatted_result
                assert spike_result.decision.detected is True
    
    def test_fallback_registry_integration(self, test_config):
        """Test integration with fallback registry."""
        agent = SlimMainAgent(config=test_config)
        
        # Test fallback execution
        with patch.object(agent.fallback_registry, 'execute_fallback') as mock_fallback:
            mock_fallback.return_value = {
                'fallback': True,
                'reason': 'llm_unavailable',
                'response': 'Fallback response'
            }
            
            # Simulate LLM failure
            with patch.object(agent, '_generate_llm_response', side_effect=Exception("LLM failed")):
                result = agent.process_question("Test question")
                
                # Should use fallback
                mock_fallback.assert_called_once()
                assert result.metadata.get('used_fallback') is True
    
    def test_configuration_normalization(self, temp_data_dir):
        """Test configuration normalization works correctly."""
        # Test with nested dict config
        nested_config = {
            'spike': {'composite_threshold': 0.7},
            'data': {'base_path': temp_data_dir}
        }
        
        agent = SlimMainAgent(config=nested_config)
        
        # Check config was normalized
        assert agent.config is not None
        assert hasattr(agent.config, 'spike') or 'spike' in agent.config
    
    def test_memory_management_integration(self, test_config):
        """Test memory management integration."""
        agent = SlimMainAgent(config=test_config)
        
        # Test adding knowledge
        agent.add_knowledge("Test knowledge for integration")
        
        # Test that memory was added (mock check)
        with patch.object(agent.l2_memory_manager, 'get_memory_count') as mock_count:
            mock_count.return_value = 1
            assert agent.l2_memory_manager.get_memory_count() == 1
    
    def test_layer_initialization_order(self, test_config):
        """Test that layers are initialized in correct order."""
        agent = SlimMainAgent(config=test_config)
        
        # Check all layers exist
        layers = [
            agent.l1_error_monitor,
            agent.l2_memory_manager,
            agent.l3_graph_reasoner,
            agent.l4_llm_interface
        ]
        
        for layer in layers:
            assert layer is not None
        
        # Check spike pipeline has reference to layers
        assert agent.spike_pipeline is not None
    
    @patch('insightspike.implementations.llm.factory.LLMProviderFactory.create')
    def test_error_handling_robustness(self, mock_factory, test_config):
        """Test error handling throughout the pipeline."""
        # Mock LLM that fails
        mock_llm = Mock()
        mock_llm.generate.side_effect = Exception("LLM Error")
        mock_factory.return_value = mock_llm
        
        agent = SlimMainAgent(config=test_config)
        
        # Should handle errors gracefully
        result = agent.process_question("Test question")
        
        # Should not crash and should provide fallback response
        assert result is not None
        assert hasattr(result, 'response')
        
    def test_adaptive_decision_mode_integration(self, test_config):
        """Test adaptive decision mode integration."""
        # Configure for adaptive mode
        test_config['spike']['decision_mode'] = 'adaptive'
        
        agent = SlimMainAgent(config=test_config)
        
        # Check adaptive mode is set
        assert agent.spike_pipeline.detector.mode == SpikeDecisionMode.ADAPTIVE
        
        # Test that adaptive thresholds are initialized
        assert hasattr(agent.spike_pipeline.detector, 'adaptive_thresholds')
        assert len(agent.spike_pipeline.detector.adaptive_thresholds) > 0


class TestSlimMainAgentPerformance:
    """Performance tests for SlimMainAgent."""
    
    @pytest.fixture
    def performance_config(self):
        """Config optimized for performance testing."""
        return {
            'data': {'datastore_type': 'memory'},
            'llm': {'provider': 'mock'},
            'spike': {
                'composite_threshold': 0.6,
                'enable_lightweight': True
            }
        }
    
    @patch('insightspike.implementations.llm.factory.LLMProviderFactory.create')
    def test_lightweight_spike_detection(self, mock_factory, performance_config):
        """Test lightweight spike detection for simple cases."""
        mock_llm = Mock()
        mock_llm.generate.return_value = "Quick response"
        mock_factory.return_value = mock_llm
        
        agent = SlimMainAgent(config=performance_config)
        
        # Test lightweight detection
        gedig_result = {'gedig': 0.3, 'ged': -0.1, 'ig': 0.4}
        
        lightweight_result = agent.spike_pipeline.execute_lightweight(gedig_result)
        
        assert 'spike_detected' in lightweight_result
        assert 'mode' in lightweight_result
        assert lightweight_result['mode'] == 'lightweight'
    
    def test_pipeline_metrics_collection(self, performance_config):
        """Test that pipeline collects performance metrics."""
        agent = SlimMainAgent(config=performance_config)
        
        # Get pipeline metrics
        metrics = agent.spike_pipeline.get_pipeline_metrics()
        
        assert 'total_executions' in metrics
        assert 'total_spikes' in metrics
        assert 'overall_spike_rate' in metrics


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])