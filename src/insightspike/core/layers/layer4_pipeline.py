"""
Layer 4 Pipeline - Integrated Semantic Generation
===============================================

Coordinates Layer 4 (PromptBuilder) and Layer 4.1 (LLM Polish)
to provide flexible text generation with optional LLM enhancement.
"""

import logging
from typing import Any, Dict, Optional

from .layer4_prompt_builder import L4PromptBuilder
from .layer4_1_llm_polish import L4_1LLMPolish

logger = logging.getLogger(__name__)

__all__ = ["Layer4Pipeline"]


class Layer4Pipeline:
    """Integrated Layer 4 Pipeline
    
    Manages the flow between:
    - Layer 4: Core semantic generation (L4PromptBuilder)
    - Layer 4.1: Optional LLM polish (L4_1LLMPolish)
    
    Provides a unified interface for the rest of the system.
    """
    
    def __init__(self, config=None):
        from ...config import get_config
        
        self.config = config or get_config()
        
        # Initialize layers
        self.layer4 = L4PromptBuilder(config)
        self.layer4_1 = L4_1LLMPolish(config)
        
        # Pipeline configuration
        self.enable_polish = getattr(self.config.llm, "enable_polish", True)
        self.force_direct = getattr(self.config.llm, "force_direct_generation", False)
        
        logger.info("Layer 4 Pipeline initialized")
    
    def generate_response(
        self, 
        context: Dict[str, Any], 
        question: str,
        mode: Optional[str] = None,
        style: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response through the Layer 4 pipeline.
        
        Args:
            context: Context from previous layers
            question: User query
            mode: Optional mode override ('prompt', 'direct', 'auto')
            style: Optional style for polishing
            
        Returns:
            Generated response with metadata
        """
        # Prepare input for Layer 4
        layer4_input = {
            "context": context,
            "question": question,
            "mode": mode
        }
        
        # Force direct mode if configured
        if self.force_direct:
            layer4_input["mode"] = "direct"
        
        # Process through Layer 4
        layer4_output = self.layer4.process(layer4_input)
        
        # Apply Layer 4.1 polish if enabled and appropriate
        if self.enable_polish and layer4_output.get("mode") == "direct":
            final_output = self.layer4_1.process_with_layer4(layer4_output)
        else:
            final_output = layer4_output
        
        # Add pipeline metadata
        final_output["pipeline"] = {
            "layer4_mode": layer4_output.get("mode"),
            "polish_applied": final_output.get("polished", False),
            "style": style
        }
        
        # For backward compatibility, map 'output' to 'response'
        final_output["response"] = final_output.get("output", "")
        final_output["success"] = True
        
        return final_output
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "layer4_config": {
                "direct_generation_enabled": getattr(self.config.llm, "use_direct_generation", False),
                "threshold": getattr(self.config.llm, "direct_generation_threshold", 0.7),
                "force_direct": self.force_direct
            },
            "layer4_1_config": {
                "polish_enabled": self.enable_polish,
                "polish_threshold": self.layer4_1.polish_threshold,
                "always_polish_below": self.layer4_1.always_polish_below
            }
        }
    
    def set_mode(self, mode: str):
        """Set pipeline mode.
        
        Args:
            mode: 'auto', 'direct', 'prompt', 'polish_all', 'no_polish'
        """
        if mode == "direct":
            self.force_direct = True
            self.enable_polish = True
        elif mode == "prompt":
            self.force_direct = False
            self.enable_polish = False
        elif mode == "polish_all":
            self.enable_polish = True
            self.layer4_1.polish_threshold = 1.0  # Polish everything
        elif mode == "no_polish":
            self.enable_polish = False
        elif mode == "auto":
            self.force_direct = False
            self.enable_polish = True
            self.layer4_1.polish_threshold = getattr(self.config.llm, "polish_threshold", 0.6)
        else:
            raise ValueError(f"Unknown mode: {mode}")