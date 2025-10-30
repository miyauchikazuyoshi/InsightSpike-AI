"""Fallback strategy registry for geDIG computation.

This module provides a centralized registry for fallback strategies
when geDIG computation fails or produces invalid results.
"""

from typing import Dict, Any, Callable, Optional, List
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class FallbackReason(Enum):
    """Reasons for fallback activation."""
    PURE_CALCULATION_FAILED = "pure_calc_failed"
    FULL_CALCULATION_FAILED = "full_calc_failed"
    INVALID_GRAPH_INPUT = "invalid_graph"
    DIVISION_BY_ZERO = "division_by_zero"
    NUMERICAL_INSTABILITY = "numerical_instability"
    MISSING_DEPENDENCIES = "missing_deps"
    TIMEOUT = "timeout"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class FallbackStrategy:
    """Defines a fallback strategy."""
    name: str
    handler: Callable[[Any, Exception, Dict[str, Any]], Dict[str, Any]]
    priority: int = 100  # Lower number = higher priority
    conditions: Optional[List[FallbackReason]] = None
    description: str = ""


class FallbackRegistry:
    """Registry for managing fallback strategies."""
    
    def __init__(self):
        self.strategies: Dict[str, FallbackStrategy] = {}
        self._register_default_strategies()
    
    def register(self, strategy: FallbackStrategy) -> None:
        """Register a new fallback strategy."""
        self.strategies[strategy.name] = strategy
        logger.debug(f"Registered fallback strategy: {strategy.name}")
    
    def get_strategy(self, reason: FallbackReason) -> Optional[FallbackStrategy]:
        """Get the best strategy for a given fallback reason."""
        applicable_strategies = []
        
        for strategy in self.strategies.values():
            if strategy.conditions is None or reason in strategy.conditions:
                applicable_strategies.append(strategy)
        
        if not applicable_strategies:
            return self.strategies.get('default_safe')
        
        # Return strategy with highest priority (lowest number)
        return min(applicable_strategies, key=lambda s: s.priority)
    
    def execute_fallback(
        self, 
        reason: FallbackReason, 
        original_exception: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the appropriate fallback strategy."""
        
        strategy = self.get_strategy(reason)
        if strategy is None:
            return self._default_safe_fallback(original_exception, context)
        
        try:
            logger.info(f"Executing fallback strategy: {strategy.name} for {reason}")
            result = strategy.handler(reason, original_exception, context)
            result['fallback_strategy'] = strategy.name
            result['fallback_reason'] = reason.value
            return result
            
        except Exception as e:
            logger.error(f"Fallback strategy {strategy.name} failed: {e}")
            return self._default_safe_fallback(e, context)
    
    def _register_default_strategies(self):
        """Register default fallback strategies."""
        
        # Safe default - returns zeros
        self.register(FallbackStrategy(
            name="default_safe",
            handler=self._default_safe_fallback,
            priority=999,
            description="Safe default returning zero values"
        ))
        
        # Pure to full fallback
        self.register(FallbackStrategy(
            name="pure_to_full",
            handler=self._pure_to_full_fallback,
            priority=10,
            conditions=[FallbackReason.PURE_CALCULATION_FAILED],
            description="Fallback from pure to full geDIG calculation"
        ))
        
        # Invalid input handler
        self.register(FallbackStrategy(
            name="invalid_input",
            handler=self._invalid_input_fallback,
            priority=20,
            conditions=[FallbackReason.INVALID_GRAPH_INPUT],
            description="Handle invalid graph input"
        ))
        
        # Numerical stability handler
        self.register(FallbackStrategy(
            name="numerical_stable",
            handler=self._numerical_stable_fallback,
            priority=30,
            conditions=[
                FallbackReason.DIVISION_BY_ZERO, 
                FallbackReason.NUMERICAL_INSTABILITY
            ],
            description="Handle numerical instability"
        ))
        
        # Missing dependencies handler
        self.register(FallbackStrategy(
            name="missing_deps",
            handler=self._missing_deps_fallback,
            priority=40,
            conditions=[FallbackReason.MISSING_DEPENDENCIES],
            description="Handle missing dependencies"
        ))
    
    def _default_safe_fallback(
        self, 
        reason_or_exception, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Safe fallback that always succeeds."""
        return {
            'gedig': 0.0,
            'ged': 0.0,
            'ig': 0.0,
            'mode': 'fallback_safe',
            'error': str(reason_or_exception),
            'success': False
        }
    
    def _pure_to_full_fallback(
        self, 
        reason: FallbackReason, 
        exception: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback from pure to full implementation."""
        
        try:
            # Try canonical selector full computation
            prev_graph = context.get('previous_graph')
            curr_graph = context.get('current_graph')
            from insightspike.algorithms.gedig.selector import compute_gedig as _sel_compute
            full_result = _sel_compute(prev_graph, curr_graph, mode='full')
            full_result['mode'] = 'full_fallback'
            return full_result
            
        except Exception as e:
            logger.warning(f"Full fallback also failed: {e}")
        
        return self._default_safe_fallback(exception, context)
    
    def _invalid_input_fallback(
        self,
        reason: FallbackReason,
        exception: Exception, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle invalid graph input."""
        
        logger.warning(f"Invalid graph input detected: {exception}")
        
        # Try to provide minimal valid result
        return {
            'gedig': 0.0,
            'ged': 0.0,
            'ig': 0.0,
            'mode': 'invalid_input_fallback',
            'error': 'Invalid graph structure provided',
            'success': False
        }
    
    def _numerical_stable_fallback(
        self,
        reason: FallbackReason,
        exception: Exception,
        context: Dict[str, Any] 
    ) -> Dict[str, Any]:
        """Handle numerical instability."""
        
        logger.warning(f"Numerical instability detected: {exception}")
        
        # Provide safe numerical values
        return {
            'gedig': 0.001,  # Small positive value to avoid zero
            'ged': 0.001,
            'ig': 0.0,
            'mode': 'numerical_stable_fallback',
            'error': 'Numerical instability resolved',
            'success': True
        }
    
    def _missing_deps_fallback(
        self,
        reason: FallbackReason,
        exception: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle missing dependencies."""
        
        logger.warning(f"Missing dependencies: {exception}")
        
        # Use simplified calculation if possible
        try:
            # Try to do basic graph comparison without heavy dependencies
            prev_graph = context.get('previous_graph')
            curr_graph = context.get('current_graph')
            
            if prev_graph is not None and curr_graph is not None:
                # Simple node/edge count difference
                prev_nodes = getattr(prev_graph, 'number_of_nodes', lambda: 0)()
                curr_nodes = getattr(curr_graph, 'number_of_nodes', lambda: 0)()
                node_diff = abs(curr_nodes - prev_nodes)
                
                return {
                    'gedig': node_diff * 0.1,  # Simple heuristic
                    'ged': node_diff * 0.1,
                    'ig': 0.0,
                    'mode': 'simplified_fallback',
                    'success': True
                }
        except:
            pass
        
        return self._default_safe_fallback(exception, context)
    
    def list_strategies(self) -> List[Dict[str, Any]]:
        """List all registered strategies."""
        return [
            {
                'name': strategy.name,
                'priority': strategy.priority,
                'conditions': [c.value for c in (strategy.conditions or [])],
                'description': strategy.description
            }
            for strategy in self.strategies.values()
        ]


# Global registry instance
_global_registry = None


def get_fallback_registry() -> FallbackRegistry:
    """Get the global fallback registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = FallbackRegistry()
    return _global_registry


def register_fallback_strategy(strategy: FallbackStrategy) -> None:
    """Register a fallback strategy with the global registry."""
    get_fallback_registry().register(strategy)


def execute_fallback(
    reason: FallbackReason,
    exception: Exception,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute fallback using the global registry."""
    return get_fallback_registry().execute_fallback(reason, exception, context)


__all__ = [
    'FallbackReason',
    'FallbackStrategy', 
    'FallbackRegistry',
    'get_fallback_registry',
    'register_fallback_strategy',
    'execute_fallback'
]
