#!/usr/bin/env python3
"""
Generic Experiment Framework for InsightSpike-AI
================================================

Reusable experiment execution, visualization, and reporting framework
extracted from experiments directory for core system use.

This module provides standardized interfaces for:
- Experiment configuration and execution
- Performance metrics collection
- Result visualization
- Report generation
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Standardized performance metrics data structure"""
    success_rate: float
    processing_time: float
    memory_usage: float
    accuracy: float
    efficiency_score: float
    insight_detection_count: int = 0
    custom_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}


@dataclass
class ExperimentConfig:
    """Generic experiment configuration"""
    name: str
    description: str
    test_cases: int
    timeout_seconds: float = 300.0
    save_results: bool = True
    generate_plots: bool = True
    output_dir: str = "experiments/results"
    random_seed: Optional[int] = None


@dataclass
class ExperimentResult:
    """Standardized experiment result structure"""
    config: ExperimentConfig
    metrics: PerformanceMetrics
    execution_time: float
    timestamp: str
    status: str  # 'success', 'failed', 'timeout'
    error_message: Optional[str] = None
    raw_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.raw_data is None:
            self.raw_data = {}


class BaseExperiment(ABC):
    """Abstract base class for all experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: List[ExperimentResult] = []
        self.start_time: Optional[float] = None
        
        # Setup directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed if specified
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
    
    @abstractmethod
    def setup_experiment(self) -> bool:
        """Setup experiment environment. Return True if successful."""
        pass
    
    @abstractmethod
    def run_single_test(self, test_id: int) -> Tuple[PerformanceMetrics, Dict[str, Any]]:
        """Run a single test case. Return metrics and raw data."""
        pass
    
    @abstractmethod
    def cleanup_experiment(self) -> None:
        """Clean up experiment resources."""
        pass
    
    def run_experiment(self) -> ExperimentResult:
        """Execute complete experiment with error handling"""
        self.start_time = time.time()
        
        try:
            # Setup
            if not self.setup_experiment():
                return self._create_failed_result("Setup failed")
            
            # Run tests
            all_metrics = []
            all_raw_data = {}
            
            for test_id in range(self.config.test_cases):
                try:
                    metrics, raw_data = self.run_single_test(test_id)
                    all_metrics.append(metrics)
                    all_raw_data[f"test_{test_id}"] = raw_data
                    
                except Exception as e:
                    print(f"Warning: Test {test_id} failed: {e}")
                    continue
            
            if not all_metrics:
                return self._create_failed_result("No tests completed successfully")
            
            # Aggregate metrics
            aggregated_metrics = self._aggregate_metrics(all_metrics)
            
            # Create result
            execution_time = time.time() - self.start_time
            result = ExperimentResult(
                config=self.config,
                metrics=aggregated_metrics,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                status='success',
                raw_data=all_raw_data
            )
            
            # Save and visualize if requested
            if self.config.save_results:
                self._save_results(result)
            
            if self.config.generate_plots:
                self._generate_visualizations(result)
            
            return result
            
        except Exception as e:
            return self._create_failed_result(f"Execution error: {e}")
        
        finally:
            self.cleanup_experiment()
    
    def _aggregate_metrics(self, metrics_list: List[PerformanceMetrics]) -> PerformanceMetrics:
        """Aggregate metrics from multiple test runs"""
        if not metrics_list:
            return PerformanceMetrics(0, 0, 0, 0, 0)
        
        return PerformanceMetrics(
            success_rate=np.mean([m.success_rate for m in metrics_list]),
            processing_time=np.mean([m.processing_time for m in metrics_list]),
            memory_usage=np.mean([m.memory_usage for m in metrics_list]),
            accuracy=np.mean([m.accuracy for m in metrics_list]),
            efficiency_score=np.mean([m.efficiency_score for m in metrics_list]),
            insight_detection_count=sum([m.insight_detection_count for m in metrics_list]),
            custom_metrics=self._merge_custom_metrics([m.custom_metrics for m in metrics_list])
        )
    
    def _merge_custom_metrics(self, custom_metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge custom metrics from multiple runs"""
        merged = {}
        for metrics_dict in custom_metrics_list:
            for key, value in metrics_dict.items():
                if key not in merged:
                    merged[key] = []
                merged[key].append(value)
        
        # Average numeric values
        for key, values in merged.items():
            if all(isinstance(v, (int, float)) for v in values):
                merged[key] = np.mean(values)
        
        return merged
    
    def _create_failed_result(self, error_message: str) -> ExperimentResult:
        """Create failed experiment result"""
        execution_time = time.time() - self.start_time if self.start_time else 0
        
        return ExperimentResult(
            config=self.config,
            metrics=PerformanceMetrics(0, 0, 0, 0, 0),
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            status='failed',
            error_message=error_message
        )
    
    def _save_results(self, result: ExperimentResult) -> None:
        """Save experiment results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Convert to serializable format
        result_dict = asdict(result)
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"📊 Results saved: {filepath}")
    
    def _generate_visualizations(self, result: ExperimentResult) -> None:
        """Generate standard performance visualizations"""
        try:
            self._create_metrics_dashboard(result)
            self._create_performance_comparison(result)
        except Exception as e:
            print(f"Warning: Visualization generation failed: {e}")
    
    def _create_metrics_dashboard(self, result: ExperimentResult) -> None:
        """Create metrics dashboard visualization"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Experiment: {self.config.name}', fontsize=16, fontweight='bold')
        
        metrics = result.metrics
        
        # Success rate
        ax1 = axes[0, 0]
        ax1.bar(['Success Rate'], [metrics.success_rate], color='green', alpha=0.7)
        ax1.set_title('Success Rate')
        ax1.set_ylabel('Rate')
        ax1.set_ylim(0, 1)
        
        # Processing time
        ax2 = axes[0, 1]
        ax2.bar(['Processing Time'], [metrics.processing_time], color='blue', alpha=0.7)
        ax2.set_title('Processing Time (s)')
        ax2.set_ylabel('Time (seconds)')
        
        # Accuracy
        ax3 = axes[1, 0]
        ax3.bar(['Accuracy'], [metrics.accuracy], color='orange', alpha=0.7)
        ax3.set_title('Accuracy')
        ax3.set_ylabel('Score')
        ax3.set_ylim(0, 1)
        
        # Efficiency
        ax4 = axes[1, 1]
        ax4.bar(['Efficiency'], [metrics.efficiency_score], color='purple', alpha=0.7)
        ax4.set_title('Efficiency Score')
        ax4.set_ylabel('Score')
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.name}_dashboard_{timestamp}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Dashboard saved: {filepath}")
    
    def _create_performance_comparison(self, result: ExperimentResult) -> None:
        """Create performance comparison visualization"""
        # Implementation for performance comparison charts
        # This would compare against baselines or previous runs
        pass


class ExperimentSuite:
    """Manager for running multiple related experiments"""
    
    def __init__(self, name: str, output_dir: str = "experiments/results"):
        self.name = name
        self.output_dir = Path(output_dir)
        self.experiments: List[BaseExperiment] = []
        self.results: List[ExperimentResult] = []
    
    def add_experiment(self, experiment: BaseExperiment) -> None:
        """Add experiment to the suite"""
        self.experiments.append(experiment)
    
    def run_all(self) -> List[ExperimentResult]:
        """Run all experiments in the suite"""
        print(f"🚀 Running Experiment Suite: {self.name}")
        print("=" * 60)
        
        self.results = []
        
        for i, experiment in enumerate(self.experiments, 1):
            print(f"\n📋 Experiment {i}/{len(self.experiments)}: {experiment.config.name}")
            
            result = experiment.run_experiment()
            self.results.append(result)
            
            if result.status == 'success':
                print(f"✅ Completed in {result.execution_time:.2f}s")
            else:
                print(f"❌ Failed: {result.error_message}")
        
        # Generate suite summary
        self._generate_suite_summary()
        
        return self.results
    
    def _generate_suite_summary(self) -> None:
        """Generate summary report for the entire suite"""
        successful_experiments = [r for r in self.results if r.status == 'success']
        
        summary = {
            'suite_name': self.name,
            'total_experiments': len(self.experiments),
            'successful_experiments': len(successful_experiments),
            'success_rate': len(successful_experiments) / len(self.experiments) if self.experiments else 0,
            'timestamp': datetime.now().isoformat(),
            'results': [asdict(result) for result in self.results]
        }
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_suite_summary_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📋 Suite Summary:")
        print(f"  Total: {summary['total_experiments']} experiments")
        print(f"  Successful: {summary['successful_experiments']}")
        print(f"  Success Rate: {summary['success_rate']:.1%}")
        print(f"  Summary saved: {filepath}")


# Utility functions for common experiment patterns

def create_simple_experiment_config(name: str, description: str, test_cases: int = 10) -> ExperimentConfig:
    """Create a simple experiment configuration"""
    return ExperimentConfig(
        name=name,
        description=description,
        test_cases=test_cases,
        timeout_seconds=300.0,
        save_results=True,
        generate_plots=True
    )


def create_performance_metrics(success_rate: float, processing_time: float, 
                             accuracy: float = 0.0, insights: int = 0) -> PerformanceMetrics:
    """Create performance metrics with sensible defaults"""
    efficiency_score = success_rate * accuracy / max(processing_time, 0.001)
    
    return PerformanceMetrics(
        success_rate=success_rate,
        processing_time=processing_time,
        memory_usage=0.0,  # To be measured by implementation
        accuracy=accuracy,
        efficiency_score=efficiency_score,
        insight_detection_count=insights
    )


def load_experiment_result(filepath: str) -> ExperimentResult:
    """Load experiment result from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Reconstruct objects from dict
    config = ExperimentConfig(**data['config'])
    metrics = PerformanceMetrics(**data['metrics'])
    
    return ExperimentResult(
        config=config,
        metrics=metrics,
        execution_time=data['execution_time'],
        timestamp=data['timestamp'],
        status=data['status'],
        error_message=data.get('error_message'),
        raw_data=data.get('raw_data', {})
    )
