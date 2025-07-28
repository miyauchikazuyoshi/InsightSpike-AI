"""
LLM Provider Interface
=====================

Protocol for language model providers.
"""

from typing import Protocol, Dict, Any, Optional, List, runtime_checkable


@runtime_checkable
class ILLMProvider(Protocol):
    """
    Interface for LLM providers.
    """
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text
        """
        ...
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Model information dictionary
        """
        ...
    
    def validate_config(self) -> bool:
        """
        Validate provider configuration.
        
        Returns:
            True if configuration is valid
        """
        ...
    
    @property
    def supports_streaming(self) -> bool:
        """
        Check if provider supports streaming.
        
        Returns:
            True if streaming is supported
        """
        ...
    
    @property
    def max_context_length(self) -> int:
        """
        Get maximum context length.
        
        Returns:
            Maximum context length in tokens
        """
        ...