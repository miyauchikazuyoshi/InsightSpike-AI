"""
Layer 4 Interface Definition
===========================

Defines the interface for Layer 4 (Semantic Generation) components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class L4Interface(ABC):
    """Abstract interface for Layer 4 components.

    Layer 4 is responsible for:
    - Semantic response generation
    - Transforming graph analysis into human-readable text
    - Synthesizing insights from multiple sources
    - Direct generation without LLM dependency
    """

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through Layer 4.

        Args:
            input_data: Dictionary containing:
                - context: Graph analysis and retrieved documents
                - question: User query
                - mode: 'prompt' or 'direct' generation mode

        Returns:
            Dictionary containing:
                - output: Generated text (prompt or direct response)
                - mode: Generation mode used
                - confidence: Confidence score
                - metadata: Additional information
        """
        pass

    @abstractmethod
    def build_prompt(self, context: Dict[str, Any], question: str) -> str:
        """Build a structured prompt for LLM processing.

        Args:
            context: Analysis context from previous layers
            question: User query

        Returns:
            Structured prompt string
        """
        pass

    @abstractmethod
    def build_direct_response(self, context: Dict[str, Any], question: str) -> str:
        """Build a complete response without LLM.

        Args:
            context: Analysis context from previous layers
            question: User query

        Returns:
            Complete response string
        """
        pass

    @abstractmethod
    def get_generation_mode(self, context: Dict[str, Any]) -> str:
        """Determine the appropriate generation mode.

        Args:
            context: Analysis context

        Returns:
            'prompt' or 'direct'
        """
        pass


class L4_1Interface(ABC):
    """Abstract interface for Layer 4.1 (Optional LLM Polish) components.

    Layer 4.1 is responsible for:
    - Optional text polishing using LLMs
    - Style enhancement
    - Grammar and fluency improvements
    - Can be bypassed when not needed
    """

    @abstractmethod
    def polish(self, text: str, style: Optional[str] = None) -> str:
        """Polish the text using LLM.

        Args:
            text: Input text from Layer 4
            style: Optional style guide ('formal', 'casual', 'technical')

        Returns:
            Polished text
        """
        pass

    @abstractmethod
    def should_polish(self, text: str, confidence: float) -> bool:
        """Determine if polishing is needed.

        Args:
            text: Generated text
            confidence: Generation confidence

        Returns:
            True if polishing should be applied
        """
        pass
