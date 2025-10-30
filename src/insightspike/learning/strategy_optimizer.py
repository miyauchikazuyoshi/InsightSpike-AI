"""
Strategy Optimizer for Adaptive Learning
========================================

Optimizes retrieval and reasoning strategies based on pattern analysis.

IMPORT INSTRUMENTATION: lightweight prints for diagnosing pytest collection hang.
Remove after localization.
"""
print('[strategy_optimizer] module import start', flush=True)

import logging
from typing import Any, Dict, List, Optional, Tuple
print('[strategy_optimizer] stdlib typing imported', flush=True)

import numpy as np
print('[strategy_optimizer] numpy imported', flush=True)

from .pattern_logger import PatternLogger, ReasoningPattern
print('[strategy_optimizer] pattern_logger imported', flush=True)

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """
    Optimizes system strategies based on learning from past patterns.
    
    Features:
    - Dynamic parameter adjustment
    - Multi-armed bandit approach for exploration/exploitation
    - Performance tracking and analysis
    - Adaptive learning rates
    """
    
    def __init__(self, config=None, pattern_logger: Optional[PatternLogger] = None):
        """Initialize StrategyOptimizer (instrumented)."""
        print('[strategy_optimizer] __init__ start', flush=True)
        self.config = config
        self.pattern_logger = pattern_logger or PatternLogger(config)
        print('[strategy_optimizer] pattern_logger ready', flush=True)

        # Strategy parameters with bounds
        self.parameter_bounds = {
            "similarity_threshold": (0.1, 0.8),
            "hop_limit": (1, 3),
            "path_decay": (0.3, 0.9),
            "max_retrieved_docs": (5, 20),
            "spike_ged_threshold": (-1.0, 0.0),
            "spike_ig_threshold": (0.1, 0.5),
        }

        # Learning parameters
        self.exploration_rate = 0.1  # ε-greedy
        self.learning_rate = 0.1
        self.momentum = 0.9

        # Parameter value history for momentum
        self.param_velocities = {param: 0.0 for param in self.parameter_bounds}

        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.best_config = None
        self.best_performance = -float('inf')
        print('[strategy_optimizer] __init__ complete', flush=True)
    
    def optimize_strategy(
        self,
        current_config: Dict[str, Any],
        recent_performance: float,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters based on recent performance.
        
        Args:
            current_config: Current configuration parameters
            recent_performance: Recent average reward/performance
            context: Optional context (question type, domain, etc.)
            
        Returns:
            Optimized configuration
        """
        
        # Record performance
        self.performance_history.append({
            "config": current_config.copy(),
            "performance": recent_performance,
            "timestamp": time.time() if context else 0,
        })
        
        # Update best config
        if recent_performance > self.best_performance:
            self.best_performance = recent_performance
            self.best_config = current_config.copy()
            logger.info(f"New best performance: {recent_performance:.3f}")
        
        # Decide whether to explore or exploit
        if np.random.random() < self.exploration_rate:
            # Explore: try new parameters
            return self._explore_parameters(current_config)
        else:
            # Exploit: use gradient-based optimization
            return self._exploit_best_direction(current_config, recent_performance)
    
    def _explore_parameters(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Explore parameter space with controlled randomness"""
        
        new_config = current_config.copy()
        
        # Randomly adjust 1-2 parameters
        params_to_adjust = np.random.choice(
            list(self.parameter_bounds.keys()),
            size=min(2, len(self.parameter_bounds)),
            replace=False
        )
        
        for param in params_to_adjust:
            if param not in current_config:
                continue
            
            bounds = self.parameter_bounds[param]
            current_value = current_config[param]
            
            # Add Gaussian noise
            if param == "hop_limit":
                # Integer parameter
                noise = np.random.choice([-1, 0, 1])
                new_value = current_value + noise
            else:
                # Float parameter
                std_dev = (bounds[1] - bounds[0]) * 0.1  # 10% of range
                noise = np.random.normal(0, std_dev)
                new_value = current_value + noise
            
            # Clip to bounds
            new_value = np.clip(new_value, bounds[0], bounds[1])
            
            if param == "hop_limit":
                new_value = int(new_value)
            
            new_config[param] = new_value
            
            logger.debug(f"Exploring: {param} {current_value:.3f} → {new_value:.3f}")
        
        return new_config
    
    def _exploit_best_direction(
        self,
        current_config: Dict[str, Any],
        recent_performance: float
    ) -> Dict[str, Any]:
        """Use gradient estimation to move toward better performance"""
        
        new_config = current_config.copy()
        
        # Estimate gradient from recent history
        if len(self.performance_history) < 3:
            return new_config
        
        # Get recent config-performance pairs
        recent_history = self.performance_history[-10:]
        
        for param in self.parameter_bounds:
            if param not in current_config:
                continue
            
            # Estimate gradient for this parameter
            gradient = self._estimate_gradient(param, recent_history, current_config)
            
            # Update with momentum
            self.param_velocities[param] = (
                self.momentum * self.param_velocities[param] +
                self.learning_rate * gradient
            )
            
            # Apply update
            current_value = current_config[param]
            new_value = current_value + self.param_velocities[param]
            
            # Clip to bounds
            bounds = self.parameter_bounds[param]
            new_value = np.clip(new_value, bounds[0], bounds[1])
            
            if param == "hop_limit":
                new_value = int(round(new_value))
            
            new_config[param] = new_value
            
            if abs(self.param_velocities[param]) > 0.001:
                logger.debug(
                    f"Optimizing: {param} {current_value:.3f} → {new_value:.3f} "
                    f"(gradient={gradient:.3f})"
                )
        
        return new_config
    
    def _estimate_gradient(
        self,
        param: str,
        history: List[Dict[str, Any]],
        reference_config: Dict[str, Any]
    ) -> float:
        """Estimate gradient for a parameter using finite differences"""
        
        if param not in reference_config:
            return 0.0
        
        reference_value = reference_config[param]
        
        # Find configs that differ mainly in this parameter
        gradients = []
        
        for i in range(len(history) - 1):
            config1 = history[i]["config"]
            config2 = history[i + 1]["config"]
            
            if param not in config1 or param not in config2:
                continue
            
            # Check if other parameters are similar
            other_params_similar = True
            for other_param in self.parameter_bounds:
                if other_param == param or other_param not in config1:
                    continue
                
                if abs(config1.get(other_param, 0) - config2.get(other_param, 0)) > 0.1:
                    other_params_similar = False
                    break
            
            if other_params_similar:
                # Estimate gradient
                param_diff = config2[param] - config1[param]
                perf_diff = history[i + 1]["performance"] - history[i]["performance"]
                
                if abs(param_diff) > 0.001:
                    gradient = perf_diff / param_diff
                    gradients.append(gradient)
        
        if gradients:
            # Return weighted average, recent gradients weighted more
            weights = np.exp(np.linspace(-1, 0, len(gradients)))
            weights /= weights.sum()
            return np.average(gradients, weights=weights)
        
        return 0.0
    
    def get_adaptive_config(
        self,
        question: str,
        base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get adaptive configuration for a specific question"""
        
        # Get recommendation from pattern logger
        recommended = self.pattern_logger.recommend_strategy(question, base_config)
        
        # Apply current optimization
        performance = self._estimate_recent_performance()
        optimized = self.optimize_strategy(recommended, performance)
        
        return optimized
    
    def _estimate_recent_performance(self) -> float:
        """Estimate recent system performance"""
        
        if not self.performance_history:
            return 0.5
        
        # Weighted average of recent performances
        recent = self.performance_history[-5:]
        performances = [h["performance"] for h in recent]
        
        if performances:
            # Recent performances weighted more
            weights = np.exp(np.linspace(-1, 0, len(performances)))
            weights /= weights.sum()
            return np.average(performances, weights=weights)
        
        return 0.5
    
    def report_performance(self) -> Dict[str, Any]:
        """Generate performance report"""
        
        strategy_perf = self.pattern_logger.get_strategy_performance()
        
        report = {
            "current_performance": self._estimate_recent_performance(),
            "best_performance": self.best_performance,
            "best_config": self.best_config,
            "strategy_performance": strategy_perf,
            "total_patterns": len(self.pattern_logger.patterns),
            "exploration_rate": self.exploration_rate,
            "parameter_velocities": self.param_velocities.copy(),
        }
        
        # Add parameter statistics
        param_stats = {}
        for param in self.parameter_bounds:
            values = [
                h["config"].get(param, 0) 
                for h in self.performance_history 
                if param in h["config"]
            ]
            if values:
                param_stats[param] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "current": self.performance_history[-1]["config"].get(param, 0),
                }
        
        report["parameter_stats"] = param_stats
        
        return report
    
    def decay_exploration(self, decay_factor: float = 0.99) -> None:
        """Decay exploration rate over time"""
        self.exploration_rate *= decay_factor
        self.exploration_rate = max(0.01, self.exploration_rate)  # Min 1%


# Add missing import
import time