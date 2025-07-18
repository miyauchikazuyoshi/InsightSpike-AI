"""
Layer 4: Language Interface
==========================

Natural language synthesis and interaction layer (Broca's/Wernicke's areas analog).
Consolidates all LLM provider implementations into a single configurable interface.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Optional imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.info("OpenAI not available")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.info("Anthropic not available")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.info("Ollama not available")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.info("Transformers not available")


class LLMProviderType(Enum):
    """Available LLM provider types"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    LOCAL = "local"           # HuggingFace transformers
    CLEAN = "clean"           # Clean responses (no data leaks)
    MOCK = "mock"             # Mock for testing


@dataclass
class LLMConfig:
    """Unified configuration for LLM providers"""
    
    # Provider selection
    provider: LLMProviderType = LLMProviderType.CLEAN
    
    # Model settings
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 500
    top_p: float = 0.9
    
    # API settings
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout: int = 30
    
    # Feature toggles
    add_special_tokens: bool = True
    use_system_prompt: bool = True
    enable_caching: bool = False
    enable_retry: bool = True
    
    # Prompt settings
    system_prompt: Optional[str] = None
    prompt_template: Optional[str] = None
    
    # Local model settings
    device: str = "cpu"
    load_in_8bit: bool = False
    
    @classmethod
    def from_provider(cls, provider: str, **kwargs) -> "LLMConfig":
        """Create config for specific provider"""
        provider_type = LLMProviderType(provider.lower())
        config = cls(provider=provider_type, **kwargs)
        
        # Provider-specific defaults
        if provider_type == LLMProviderType.OPENAI:
            config.model_name = kwargs.get("model_name", "gpt-3.5-turbo")
            
        elif provider_type == LLMProviderType.ANTHROPIC:
            config.model_name = kwargs.get("model_name", "claude-2")
            config.max_tokens = kwargs.get("max_tokens", 1000)
            
        elif provider_type == LLMProviderType.OLLAMA:
            config.model_name = kwargs.get("model_name", "llama2")
            
        elif provider_type == LLMProviderType.LOCAL:
            config.model_name = kwargs.get("model_name", "distilgpt2")
            if torch.cuda.is_available():
                config.device = "cuda"
                
        return config


class L4LLMInterface:
    """
    Layer 4 Language Interface - Natural language synthesis.
    
    Brain analog: Broca's area (language production) + Wernicke's area (comprehension)
    
    Features:
    - Multiple provider support: OpenAI, Anthropic, Ollama, Local, Clean, Mock
    - Unified prompt building and response generation
    - Context-aware language synthesis based on retrieved episodes and graph analysis
    - Caching and performance optimization
    """
    
    def __init__(self, config: Optional[Union[LLMConfig, 'InsightSpikeConfig']] = None, legacy_config=None):
        """Initialize with unified configuration"""
        from ...config.models import InsightSpikeConfig

        # Handle different config types
        if isinstance(config, InsightSpikeConfig):
            # Create LLMConfig from Pydantic config
            self.config = LLMConfig.from_provider(
                config.llm.provider,
                model_name=config.llm.model,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                api_key=config.llm.api_key,
                system_prompt=config.llm.system_prompt
            )
        elif isinstance(config, LLMConfig):
            self.config = config
        elif legacy_config:
            # Legacy path - create from legacy config
            self.config = LLMConfig()
            self._apply_legacy_config(legacy_config)
        else:
            # Default config
            self.config = LLMConfig()
            
        # Provider-specific components
        self.client = None
        self.model = None
        self.tokenizer = None
        
        # State
        self.initialized = False
        self.response_cache = {} if self.config.enable_caching else None
        
        logger.info(f"Initialized {self.config.provider.value} LLM provider")
        
    def _apply_legacy_config(self, legacy_config):
        """Apply settings from legacy config objects"""
        if hasattr(legacy_config, 'llm'):
            if hasattr(legacy_config.llm, 'model'):
                self.config.model_name = legacy_config.llm.model
            if hasattr(legacy_config.llm, 'temperature'):
                self.config.temperature = legacy_config.llm.temperature
            if hasattr(legacy_config.llm, 'provider'):
                try:
                    self.config.provider = LLMProviderType(legacy_config.llm.provider.lower())
                except ValueError:
                    logger.warning(f"Unknown provider: {legacy_config.llm.provider}")
                    
    def initialize(self) -> bool:
        """Initialize the LLM provider"""
        try:
            if self.config.provider == LLMProviderType.OPENAI:
                return self._initialize_openai()
                
            elif self.config.provider == LLMProviderType.ANTHROPIC:
                return self._initialize_anthropic()
                
            elif self.config.provider == LLMProviderType.OLLAMA:
                return self._initialize_ollama()
                
            elif self.config.provider == LLMProviderType.LOCAL:
                return self._initialize_local()
                
            elif self.config.provider == LLMProviderType.CLEAN:
                # Clean provider needs no initialization
                self.initialized = True
                logger.info("Clean LLM provider ready - no data leaks")
                return True
                
            elif self.config.provider == LLMProviderType.MOCK:
                # Mock provider needs no initialization
                self.initialized = True
                logger.info("Mock LLM provider ready")
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize {self.config.provider.value}: {e}")
            return False
            
    def generate_response(self, context: Dict[str, Any], question: str) -> str:
        """Generate response (simple interface)"""
        result = self.generate_response_detailed(context, question)
        return result.get("response", "")
        
    def generate_response_detailed(self, context: Dict[str, Any], question: str) -> Dict[str, Any]:
        """
        Generate response with detailed results.
        
        Unified interface that works across all providers.
        """
        if not self.initialized:
            return {
                "response": "LLM provider not initialized",
                "success": False,
                "provider": self.config.provider.value
            }
            
        try:
            # Check cache
            if self.response_cache is not None:
                cache_key = self._get_cache_key(context, question)
                if cache_key in self.response_cache:
                    logger.debug("Returning cached response")
                    return self.response_cache[cache_key]
                    
            # Build prompt
            prompt = self._build_prompt(context, question)
            
            # Generate based on provider
            if self.config.provider == LLMProviderType.CLEAN:
                result = self._generate_clean(prompt, context)
                
            elif self.config.provider == LLMProviderType.MOCK:
                result = self._generate_mock(prompt, context)
                
            elif self.config.provider == LLMProviderType.OPENAI:
                result = self._generate_openai(prompt)
                
            elif self.config.provider == LLMProviderType.ANTHROPIC:
                result = self._generate_anthropic(prompt)
                
            elif self.config.provider == LLMProviderType.OLLAMA:
                result = self._generate_ollama(prompt)
                
            elif self.config.provider == LLMProviderType.LOCAL:
                result = self._generate_local(prompt)
                
            else:
                result = {
                    "response": f"Provider {self.config.provider} not implemented",
                    "success": False
                }
                
            # Add metadata
            result["provider"] = self.config.provider.value
            result["model"] = self.config.model_name
            
            # Cache if enabled
            if self.response_cache is not None and result.get("success", False):
                cache_key = self._get_cache_key(context, question)
                self.response_cache[cache_key] = result
                
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "response": f"Error: {str(e)}",
                "success": False,
                "provider": self.config.provider.value,
                "error": str(e)
            }
            
    def _build_prompt(self, context: Dict[str, Any], question: str) -> str:
        """Build prompt from context and question"""
        # Extract relevant information
        retrieved_docs = context.get("retrieved_documents", [])
        graph_analysis = context.get("graph_analysis", {})
        reasoning_quality = context.get("reasoning_quality", 0.0)
        
        # Build context section
        context_parts = []
        
        if retrieved_docs:
            context_parts.append("Retrieved Information:")
            for i, doc in enumerate(retrieved_docs[:3], 1):
                text = doc.get("text", "")
                relevance = doc.get("relevance", 0.0)
                context_parts.append(f"{i}. {text} (relevance: {relevance:.2f})")
                
        if graph_analysis and graph_analysis.get("spike_detected", False):
            context_parts.append("\nInsight Detection: Significant pattern identified")
            
        # Use template if provided
        if self.config.prompt_template:
            prompt = self.config.prompt_template.format(
                context="\n".join(context_parts),
                question=question
            )
        else:
            # Default template
            if context_parts:
                prompt = f"Context:\n{chr(10).join(context_parts)}\n\nQuestion: {question}\n\nAnswer:"
            else:
                prompt = f"Question: {question}\n\nAnswer:"
                
        # Add special tokens if configured
        if self.config.add_special_tokens and self.config.provider == LLMProviderType.LOCAL:
            prompt = self._add_special_tokens(prompt)
            
        return prompt
        
    def _add_special_tokens(self, prompt: str) -> str:
        """Add special tokens for local models"""
        if self.config.use_system_prompt and self.config.system_prompt:
            return f"<|system|>\n{self.config.system_prompt}\n\n<|user|>\n{prompt}\n\n<|assistant|>\n"
        else:
            return f"<|user|>\n{prompt}\n\n<|assistant|>\n"
            
    # Provider-specific implementations
    
    def _initialize_openai(self) -> bool:
        """Initialize OpenAI provider"""
        if not OPENAI_AVAILABLE:
            logger.error("OpenAI package not installed")
            return False
            
        if not self.config.api_key:
            logger.error("OpenAI API key not provided")
            return False
            
        openai.api_key = self.config.api_key
        if self.config.api_base:
            openai.api_base = self.config.api_base
            
        self.initialized = True
        logger.info(f"OpenAI provider initialized with model {self.config.model_name}")
        return True
        
    def _generate_openai(self, prompt: str) -> Dict[str, Any]:
        """Generate using OpenAI"""
        messages = []
        
        if self.config.use_system_prompt and self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
            
        messages.append({"role": "user", "content": prompt})
        
        response = openai.ChatCompletion.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            timeout=self.config.timeout
        )
        
        return {
            "response": response.choices[0].message.content,
            "success": True,
            "usage": response.usage.to_dict() if hasattr(response.usage, 'to_dict') else {}
        }
        
    def _initialize_anthropic(self) -> bool:
        """Initialize Anthropic provider"""
        if not ANTHROPIC_AVAILABLE:
            logger.error("Anthropic package not installed")
            return False
            
        if not self.config.api_key:
            logger.error("Anthropic API key not provided")
            return False
            
        self.client = anthropic.Anthropic(api_key=self.config.api_key)
        self.initialized = True
        logger.info(f"Anthropic provider initialized with model {self.config.model_name}")
        return True
        
    def _generate_anthropic(self, prompt: str) -> Dict[str, Any]:
        """Generate using Anthropic"""
        system_prompt = self.config.system_prompt if self.config.use_system_prompt else None
        
        response = self.client.messages.create(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            "response": response.content[0].text,
            "success": True,
            "usage": {"input_tokens": response.usage.input_tokens, 
                     "output_tokens": response.usage.output_tokens}
        }
        
    def _initialize_ollama(self) -> bool:
        """Initialize Ollama provider"""
        if not OLLAMA_AVAILABLE:
            logger.error("Ollama package not installed")
            return False
            
        # Check if model is available
        try:
            models = ollama.list()
            model_names = [m['name'] for m in models['models']]
            if self.config.model_name not in model_names:
                logger.warning(f"Model {self.config.model_name} not found in Ollama")
                
        except Exception as e:
            logger.error(f"Failed to check Ollama models: {e}")
            
        self.initialized = True
        logger.info(f"Ollama provider initialized with model {self.config.model_name}")
        return True
        
    def _generate_ollama(self, prompt: str) -> Dict[str, Any]:
        """Generate using Ollama"""
        response = ollama.chat(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens
            }
        )
        
        return {
            "response": response['message']['content'],
            "success": True,
            "usage": {"total_duration": response.get('total_duration', 0)}
        }
        
    def _initialize_local(self) -> bool:
        """Initialize local transformers model"""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers package not installed")
            return False
            
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load model with appropriate settings
            model_kwargs = {"device_map": "auto" if self.config.device == "cuda" else None}
            
            if self.config.load_in_8bit and self.config.device == "cuda":
                model_kwargs["load_in_8bit"] = True
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            # Move to device if needed
            if self.config.device == "cpu":
                self.model = self.model.to("cpu")
                
            self.initialized = True
            logger.info(f"Local model {self.config.model_name} loaded on {self.config.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            return False
            
    def _generate_local(self, prompt: str) -> Dict[str, Any]:
        """Generate using local transformers model"""
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Move to device
        if self.config.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
            
        return {
            "response": response,
            "success": True,
            "tokens_generated": outputs.shape[1] - inputs['input_ids'].shape[1]
        }
        
    def _generate_clean(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate clean response without data leaks"""
        # Extract key information
        num_docs = len(context.get("retrieved_documents", []))
        has_spike = context.get("graph_analysis", {}).get("spike_detected", False)
        quality = context.get("reasoning_quality", 0.5)
        
        # Generate appropriate response based on context
        if num_docs == 0:
            response = "I don't have enough information to answer this question accurately."
        elif has_spike:
            response = "Based on the identified patterns, this represents a significant insight that connects multiple concepts in a novel way."
        elif quality > 0.7:
            response = "The analysis reveals strong connections between the concepts, suggesting a coherent understanding of the relationships."
        else:
            response = "The available information provides a partial understanding, though additional context would strengthen the analysis."
            
        return {
            "response": response,
            "success": True,
            "confidence": quality
        }
        
    def _generate_mock(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock response for testing"""
        # Simple mock responses
        responses = [
            "This is a mock response for testing purposes.",
            "The mock provider returns predetermined responses.",
            "Testing response: All systems operational."
        ]
        
        # Use question length to select response
        idx = len(prompt) % len(responses)
        
        return {
            "response": responses[idx],
            "success": True,
            "mock": True
        }
        
    def _get_cache_key(self, context: Dict[str, Any], question: str) -> str:
        """Generate cache key for response"""
        # Include relevant context in key
        key_parts = [
            question,
            str(len(context.get("retrieved_documents", []))),
            str(context.get("reasoning_quality", 0)),
            str(context.get("graph_analysis", {}).get("spike_detected", False))
        ]
        
        return "|".join(key_parts)
        
    def cleanup(self):
        """Cleanup resources"""
        if self.model is not None and hasattr(self.model, 'cpu'):
            # Move model to CPU to free GPU memory
            self.model.cpu()
            
        self.initialized = False
        logger.info(f"Cleaned up {self.config.provider.value} provider")


# Factory function for backward compatibility
def get_llm_provider(config=None, safe_mode: bool = False) -> L4LLMInterface:
    """Get LLM provider instance"""
    from ...config.models import InsightSpikeConfig
    
    if safe_mode or config is None:
        # Use clean provider in safe mode
        llm_config = LLMConfig(provider=LLMProviderType.CLEAN)
        return L4LLMInterface(llm_config)
    elif isinstance(config, InsightSpikeConfig):
        # Direct Pydantic config support
        return L4LLMInterface(config)
    else:
        # Legacy config support
        llm_config = LLMConfig()
        if hasattr(config, 'llm'):
            llm_config = LLMConfig.from_provider(
                getattr(config.llm, 'provider', 'clean'),
                model_name=getattr(config.llm, 'model_name', 'gpt-3.5-turbo'),
                temperature=getattr(config.llm, 'temperature', 0.7),
                max_tokens=getattr(config.llm, 'max_tokens', 500)
            )
            
        return L4LLMInterface(llm_config, legacy_config=config)


# Aliases for backward compatibility
LLMProvider = L4LLMInterface
CleanLLMProvider = L4LLMInterface
MockLLMProvider = L4LLMInterface
UnifiedLLMProvider = L4LLMInterface  # For migration