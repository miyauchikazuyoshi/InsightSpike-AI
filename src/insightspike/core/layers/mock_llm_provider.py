"""
Lightweight LLM Provider for Testing
===================================

A minimal LLM provider that doesn't load heavy models to avoid segmentation faults.
Useful for testing and development when full model functionality isn't needed.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MockLLMProvider:
    """Lightweight mock LLM provider for testing and development"""
    
    def __init__(self, config=None):
        self.config = config
        self._initialized = False
        logger.info("Mock LLM provider initialized (no model loading)")
    
    def initialize(self) -> bool:
        """Initialize mock provider (always succeeds)"""
        self._initialized = True
        logger.info("Mock LLM provider ready")
        return True
    
    def generate_response(self, context: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Generate a mock response"""
        mock_response = f"[Mock Response] This is a test response to: {prompt[:50]}..."
        
        return {
            'success': True,
            'response': mock_response,
            'reasoning_quality': 0.8,
            'confidence': 0.7,
            'tokens_used': len(prompt.split()) + 10,
            'model_used': 'mock-model'
        }
    
    def generate_stream(self, context: Dict[str, Any], prompt: str):
        """Generate a streaming mock response"""
        words = ["This", "is", "a", "mock", "streaming", "response", "for", "testing"]
        for word in words:
            yield {
                'success': True,
                'response': word,
                'done': False
            }
        yield {
            'success': True,
            'response': "",
            'done': True
        }
    
    def is_available(self) -> bool:
        """Check if provider is available"""
        return True

def get_safe_llm_provider(config=None):
    """Get a safe LLM provider that won't cause segmentation faults"""
    return MockLLMProvider(config)
