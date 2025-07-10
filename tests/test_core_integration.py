#!/usr/bin/env python3
"""
Integration tests for the generic experiment framework and visualization utilities.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

# Import the modules we're testing
from insightspike.core.experiment_framework import (
    BaseExperiment, ExperimentConfig, ExperimentResult, 
    PerformanceMetrics, ExperimentSuite,
    create_simple_experiment_config, create_performance_metrics
)
from insightspike.utils.visualization import (
    InsightSpikeVisualizer, quick_performance_chart, 
    quick_comparison, quick_progress_chart
)


class MockExperiment(BaseExperiment):
    """Mock experiment for testing"""
    
    def setup_experiment(self) -> bool:
        return True
    
    def run_single_test(self, test_id: int):
        # Simulate test execution
        import time
        time.sleep(0.01)  # Small delay to simulate work
        
        metrics = PerformanceMetrics(
            success_rate=0.8 + (test_id % 3) * 0.1,
            processing_time=0.1 + test_id * 0.01,
            memory_usage=100 + test_id * 10,
            accuracy=0.9 - (test_id % 2) * 0.1,
            efficiency_score=0.85,
            insight_detection_count=test_id % 3
        )
        
        raw_data = {
            'test_id': test_id,
            'timestamp': '2025-06-12T10:00:00',
            'details': f'Test {test_id} completed'
        }
        
        return metrics, raw_data
    
    def cleanup_experiment(self) -> None:
        pass


class TestExperimentFramework:
    """Test cases for experiment framework"""
    
    def test_experiment_config_creation(self):
        """Test experiment configuration creation"""
        config = ExperimentConfig(
            name="test_experiment",
            description="Test experiment description",
            test_cases=5,
            timeout_seconds=120.0
        )
        
        assert config.name == "test_experiment"
        assert config.test_cases == 5
        assert config.timeout_seconds == 120.0
        assert config.save_results is True  # Default value
    
    def test_performance_metrics_creation(self):
        """Test performance metrics data structure"""
        metrics = PerformanceMetrics(
            success_rate=0.95,
            processing_time=1.2,
            memory_usage=256.0,
            accuracy=0.88,
            efficiency_score=0.92,
            insight_detection_count=3
        )
        
        assert metrics.success_rate == 0.95
        assert metrics.insight_detection_count == 3
        assert metrics.custom_metrics == {}  # Default empty dict
    
    def test_simple_experiment_execution(self):
        """Test basic experiment execution"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                name="test_run",
                description="Test run",
                test_cases=3,
                output_dir=temp_dir,
                save_results=False,  # Don't save to avoid file I/O
                generate_plots=False
            )
            
            experiment = MockExperiment(config)
            result = experiment.run_experiment()
            
            assert result.status == 'success'
            assert result.config.name == "test_run"
            assert result.metrics.success_rate > 0
            assert result.execution_time > 0
            assert len(result.raw_data) == 3  # 3 test cases
    
    def test_experiment_suite(self):
        """Test experiment suite execution"""
        with tempfile.TemporaryDirectory() as temp_dir:
            suite = ExperimentSuite("test_suite", temp_dir)
            
            # Add multiple experiments
            for i in range(2):
                config = ExperimentConfig(
                    name=f"experiment_{i}",
                    description=f"Test experiment {i}",
                    test_cases=2,
                    output_dir=temp_dir,
                    save_results=False,
                    generate_plots=False
                )
                experiment = MockExperiment(config)
                suite.add_experiment(experiment)
            
            # Run suite
            results = suite.run_all()
            
            assert len(results) == 2
            assert all(r.status == 'success' for r in results)
    
    def test_convenience_functions(self):
        """Test convenience functions"""
        config = create_simple_experiment_config("test", "description", 5)
        assert config.name == "test"
        assert config.test_cases == 5
        
        metrics = create_performance_metrics(0.9, 1.5, 0.85, 3)
        assert metrics.success_rate == 0.9
        assert metrics.processing_time == 1.5
        assert metrics.insight_detection_count == 3


class TestVisualizationUtils:
    """Test cases for visualization utilities"""
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization"""
        viz = InsightSpikeVisualizer()
        assert viz.style == 'default'
        assert viz.figsize == (12, 8)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_performance_dashboard_creation(self, mock_close, mock_savefig):
        """Test performance dashboard creation"""
        viz = InsightSpikeVisualizer()
        
        metrics = {
            'success_rate': 0.95,
            'processing_time': 1.2,
            'accuracy': 0.88,
            'memory_usage': 256.0,
            'efficiency_score': 0.92,
            'insight_detection_count': 5
        }
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            result = viz.create_performance_dashboard(
                metrics, 
                title="Test Dashboard",
                save_path=tmp.name
            )
            
            assert result == tmp.name
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_comparison_chart_creation(self, mock_close, mock_savefig):
        """Test comparison chart creation"""
        viz = InsightSpikeVisualizer()
        
        data1 = [0.9, 0.8, 0.95, 0.87]
        data2 = [0.7, 0.6, 0.85, 0.75]
        labels = ['Test1', 'Test2', 'Test3', 'Test4']
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            result = viz.create_comparison_chart(
                data1, data2, labels,
                title="Test Comparison",
                save_path=tmp.name
            )
            
            assert result == tmp.name
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_progress_visualization(self, mock_close, mock_savefig):
        """Test progress visualization"""
        viz = InsightSpikeVisualizer()
        
        episodes = list(range(10))
        rewards = [i * 0.1 + np.random.random() * 0.05 for i in episodes]
        success_indicators = [r > 0.5 for r in rewards]
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            result = viz.create_progress_visualization(
                episodes, rewards, success_indicators,
                title="Test Progress",
                save_path=tmp.name
            )
            
            assert result == tmp.name
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_radar_chart_creation(self, mock_close, mock_savefig):
        """Test radar chart creation"""
        viz = InsightSpikeVisualizer()
        
        categories = ['Speed', 'Accuracy', 'Efficiency', 'Reliability']
        values = [0.8, 0.9, 0.7, 0.85]
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            result = viz.create_radar_chart(
                categories, values,
                title="Test Radar",
                save_path=tmp.name
            )
            
            assert result == tmp.name
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
    
    def test_convenience_visualization_functions(self):
        """Test convenience visualization functions"""
        with patch('src.insightspike.utils.visualization.InsightSpikeVisualizer') as mock_viz_class:
            mock_viz = Mock()
            mock_viz_class.return_value = mock_viz
            mock_viz.create_performance_dashboard.return_value = "test_path"
            
            metrics = {'success_rate': 0.9}
            result = quick_performance_chart(metrics)
            
            assert result == "test_path"
            mock_viz.create_performance_dashboard.assert_called_once()
    
    def test_config_save_load(self):
        """Test visualization configuration save/load"""
        viz = InsightSpikeVisualizer()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            viz.save_visualization_config(tmp.name)
            
            # Create new visualizer and load config
            viz2 = InsightSpikeVisualizer()
            viz2.load_visualization_config(tmp.name)
            
            assert viz2.style == viz.style
            assert viz2.figsize == viz.figsize


class TestIntegration:
    """Integration tests combining both modules"""
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_experiment_with_visualization(self, mock_close, mock_savefig):
        """Test experiment execution with visualization output"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create experiment
            config = ExperimentConfig(
                name="integration_test",
                description="Integration test with visualization",
                test_cases=3,
                output_dir=temp_dir,
                save_results=True,
                generate_plots=True
            )
            
            experiment = MockExperiment(config)
            result = experiment.run_experiment()
            
            # Verify experiment succeeded
            assert result.status == 'success'
            
            # Create visualization from results
            viz = InsightSpikeVisualizer()
            metrics_dict = {
                'success_rate': result.metrics.success_rate,
                'processing_time': result.metrics.processing_time,
                'accuracy': result.metrics.accuracy,
                'efficiency_score': result.metrics.efficiency_score,
                'insight_detection_count': result.metrics.insight_detection_count
            }
            
            viz_path = str(Path(temp_dir) / "integration_test_viz.png")
            viz_result = viz.create_performance_dashboard(
                metrics_dict,
                title="Integration Test Results",
                save_path=viz_path
            )
            
            assert viz_result == viz_path
    
    def test_experiment_suite_with_comparison(self):
        """Test experiment suite with comparison visualization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            suite = ExperimentSuite("comparison_test", temp_dir)
            
            # Add experiments with different configurations
            configs = [
                ("fast_experiment", 2),
                ("thorough_experiment", 5)
            ]
            
            for name, test_cases in configs:
                config = ExperimentConfig(
                    name=name,
                    description=f"Test {name}",
                    test_cases=test_cases,
                    output_dir=temp_dir,
                    save_results=False,
                    generate_plots=False
                )
                experiment = MockExperiment(config)
                suite.add_experiment(experiment)
            
            # Run suite
            results = suite.run_all()
            
            # Extract metrics for comparison
            success_rates = [r.metrics.success_rate for r in results]
            processing_times = [r.metrics.processing_time for r in results]
            labels = [r.config.name for r in results]
            
            # Create comparison visualization
            with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
                viz = InsightSpikeVisualizer()
                viz_result = viz.create_comparison_chart(
                    success_rates, processing_times, labels,
                    title="Experiment Suite Comparison"
                )
                
                assert viz_result == "displayed"  # No save path provided


# Test fixtures and utilities

@pytest.fixture
def sample_metrics():
    """Sample performance metrics for testing"""
    return PerformanceMetrics(
        success_rate=0.85,
        processing_time=2.3,
        memory_usage=512.0,
        accuracy=0.92,
        efficiency_score=0.78,
        insight_detection_count=7,
        custom_metrics={'custom_score': 0.88}
    )


@pytest.fixture
def sample_experiment_config():
    """Sample experiment configuration for testing"""
    return ExperimentConfig(
        name="sample_experiment",
        description="Sample experiment for testing",
        test_cases=10,
        timeout_seconds=300.0,
        save_results=True,
        generate_plots=True,
        output_dir="test_output"
    )


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
