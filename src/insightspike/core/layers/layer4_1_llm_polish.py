"""
Layer 4.1 LLM Polish - Optional Text Enhancement
===============================================

Optional layer that polishes text from Layer 4 using LLMs.
Can be completely bypassed for high-confidence direct generation.
"""

import logging
from typing import Any, Dict, Optional

from ..interfaces.layer4_interface import L4_1Interface
from .layer4_llm_provider import get_llm_provider

logger = logging.getLogger(__name__)

__all__ = ["L4_1LLMPolish"]


class L4_1LLMPolish(L4_1Interface):
    """Layer 4.1: Optional LLM Text Polish
    
    This layer:
    - Takes structured output from Layer 4
    - Applies LLM polish when needed
    - Can be bypassed for direct generation
    - Maintains semantic content while improving style
    """
    
    def __init__(self, config=None):
        from ...config import get_config
        
        self.config = config or get_config()
        self.llm_provider = None
        self._initialized = False
        
        # Polish thresholds
        self.polish_threshold = getattr(self.config.llm, "polish_threshold", 0.6)
        self.always_polish_below = getattr(self.config.llm, "always_polish_below", 0.4)
        
    def initialize(self):
        """Initialize the LLM provider."""
        if not self._initialized:
            self.llm_provider = get_llm_provider(self.config)
            self._initialized = True
            logger.info("Layer 4.1 LLM Polish initialized")
    
    def polish(self, text: str, style: Optional[str] = None) -> str:
        """Polish the text using LLM.
        
        Args:
            text: Input text from Layer 4
            style: Optional style guide
            
        Returns:
            Polished text
        """
        if not self._initialized:
            self.initialize()
        
        # Build polishing prompt
        prompt = self._build_polish_prompt(text, style)
        
        # Use LLM to polish
        context = {"polishing_mode": True}
        result = self.llm_provider.generate_response(context, prompt)
        
        if result.get("success"):
            return result.get("response", text)
        else:
            logger.warning("Polish failed, returning original text")
            return text
    
    def should_polish(self, text: str, confidence: float) -> bool:
        """Determine if polishing is needed.
        
        Args:
            text: Generated text
            confidence: Generation confidence
            
        Returns:
            True if polishing should be applied
        """
        # Always polish low confidence
        if confidence < self.always_polish_below:
            return True
        
        # Never polish very high confidence
        if confidence > self.polish_threshold:
            return False
        
        # Check text quality indicators
        quality_issues = self._check_text_quality(text)
        return quality_issues > 2
    
    def _build_polish_prompt(self, text: str, style: Optional[str] = None) -> str:
        """Build prompt for text polishing."""
        style_guide = ""
        if style == "formal":
            style_guide = "Use formal, professional language. "
        elif style == "casual":
            style_guide = "Use casual, conversational language. "
        elif style == "technical":
            style_guide = "Use precise technical language. "
        
        return f"""Please polish the following text while maintaining its semantic content.
{style_guide}Fix any grammatical errors and improve clarity.

Original text:
{text}

Polished text:"""
    
    def _check_text_quality(self, text: str) -> int:
        """Check text for quality issues.
        
        Returns count of quality issues found.
        """
        issues = 0
        
        # Check for very short responses
        if len(text.split()) < 10:
            issues += 2
        
        # Check for repetition
        sentences = text.split('. ')
        if len(sentences) > len(set(sentences)):
            issues += 1
        
        # Check for incomplete sentences
        if not text.strip().endswith(('.', '!', '?')):
            issues += 1
        
        # Check for multiple spaces or formatting issues
        if '  ' in text or '\n\n\n' in text:
            issues += 1
        
        return issues
    
    def process_with_layer4(self, layer4_output: Dict[str, Any]) -> Dict[str, Any]:
        """Process output from Layer 4, applying polish if needed.
        
        Args:
            layer4_output: Output from Layer 4 containing:
                - output: Generated text
                - mode: Generation mode
                - confidence: Confidence score
                
        Returns:
            Final output with optional polish applied
        """
        text = layer4_output.get("output", "")
        confidence = layer4_output.get("confidence", 0.0)
        mode = layer4_output.get("mode", "prompt")
        
        # Check if polishing is needed
        if mode == "direct" and self.should_polish(text, confidence):
            logger.info(f"Applying LLM polish (confidence={confidence:.3f})")
            polished_text = self.polish(text)
            
            return {
                **layer4_output,
                "output": polished_text,
                "polished": True,
                "original_output": text
            }
        else:
            return {
                **layer4_output,
                "polished": False
            }