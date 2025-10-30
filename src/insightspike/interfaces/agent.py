"""
Agent Interfaces
===============

Protocols for agent implementations.
"""

from typing import Protocol, Dict, Any, Optional, runtime_checkable

from ..core.structures import CycleResult


@runtime_checkable
class IAgent(Protocol):
    """
    Base agent interface for question-answering systems.
    """
    
    def process_question(
        self, 
        question: str, 
        max_cycles: int = 10,
        verbose: bool = False
    ) -> CycleResult:
        """
        Process a question and return the result.
        
        Args:
            question: The question to process
            max_cycles: Maximum reasoning cycles
            verbose: Whether to log detailed information
            
        Returns:
            CycleResult containing the answer and metadata
        """
        ...
    
    def initialize(self) -> bool:
        """
        Initialize the agent.
        
        Returns:
            True if initialization successful
        """
        ...
    
    def cleanup(self) -> None:
        """Clean up resources."""
        ...


@runtime_checkable
class IMemoryAgent(IAgent, Protocol):
    """
    Agent with memory management capabilities.
    """
    
    def add_knowledge(
        self, 
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add knowledge to the agent's memory.
        
        Args:
            text: Knowledge text to add
            metadata: Optional metadata
            
        Returns:
            ID of the added knowledge
        """
        ...
    
    def clear_memory(self) -> None:
        """Clear all memory."""
        ...
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dictionary with memory stats
        """
        ...