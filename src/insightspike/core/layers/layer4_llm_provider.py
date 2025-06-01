"""
L4 LLM Interface - Enhanced Language Model Integration
===================================================

Provides flexible LLM integration with multiple providers and enhanced prompt building.
"""

from typing import Dict, Any, Optional, List, Iterator
import logging
from pathlib import Path

from ..interfaces import L4LLMInterface
from ...config import get_config
from ...utils.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

__all__ = ["L4LLMProvider", "OpenAIProvider", "LocalProvider", "get_llm_provider"]


class L4LLMProvider(L4LLMInterface):
    """Base class for LLM providers."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.prompt_builder = PromptBuilder(config)
        self._initialized = False
    
    def generate_response(self, context: Dict[str, Any], question: str, 
                         streaming: bool = False) -> Dict[str, Any]:
        """Generate response using the LLM."""
        if not self._initialized:
            self.initialize()
        
        try:
            # Build enhanced prompt
            prompt = self.prompt_builder.build_prompt(context, question)
            
            # Generate response
            if streaming:
                response_iter = self._generate_streaming(prompt)
                return {
                    'response': response_iter,
                    'prompt': prompt,
                    'streaming': True,
                    'success': True
                }
            else:
                response = self._generate_sync(prompt)
                return {
                    'response': response,
                    'prompt': prompt,
                    'streaming': False,
                    'success': True,
                    'confidence': self._estimate_confidence(response, context)
                }
                
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                'response': "I apologize, but I encountered an error while processing your question.",
                'error': str(e),
                'success': False
            }
    
    def initialize(self) -> bool:
        """Initialize the LLM provider."""
        raise NotImplementedError
    
    def _generate_sync(self, prompt: str) -> str:
        """Generate response synchronously."""
        raise NotImplementedError
    
    def _generate_streaming(self, prompt: str) -> Iterator[str]:
        """Generate response with streaming."""
        # Fallback to sync generation
        response = self._generate_sync(prompt)
        for char in response:
            yield char
    
    def _estimate_confidence(self, response: str, context: Dict[str, Any]) -> float:
        """Estimate confidence in the response."""
        # Simple heuristics for confidence estimation
        
        # Length-based confidence (very short or very long responses are less confident)
        length_score = min(1.0, len(response) / 200) * min(1.0, 500 / max(len(response), 1))
        
        # Context relevance (if we have relevant documents)
        relevance_score = 1.0
        if 'retrieved_documents' in context and context['retrieved_documents']:
            num_docs = len(context['retrieved_documents'])
            relevance_score = min(1.0, num_docs / 3)  # More docs = higher confidence
        
        # Uncertainty indicators in response
        uncertainty_keywords = ['maybe', 'perhaps', 'might', 'could', 'uncertain', 'not sure']
        uncertainty_count = sum(1 for word in uncertainty_keywords if word in response.lower())
        uncertainty_penalty = min(0.5, uncertainty_count * 0.1)
        
        confidence = (length_score + relevance_score) / 2 - uncertainty_penalty
        return max(0.0, min(1.0, confidence))


class OpenAIProvider(L4LLMProvider):
    """OpenAI GPT provider."""
    
    def initialize(self) -> bool:
        """Initialize OpenAI client."""
        try:
            import openai
            
            api_key = self.config.llm.openai_api_key
            if not api_key:
                logger.error("OpenAI API key not found in config")
                return False
            
            self.client = openai.OpenAI(api_key=api_key)
            self.model = self.config.llm.openai_model
            self._initialized = True
            
            logger.info(f"Initialized OpenAI provider with model: {self.model}")
            return True
            
        except ImportError:
            logger.error("OpenAI package not installed. Run: pip install openai")
            return False
        except Exception as e:
            logger.error(f"OpenAI initialization failed: {e}")
            return False
    
    def _generate_sync(self, prompt: str) -> str:
        """Generate response using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant specialized in answering questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    def _generate_streaming(self, prompt: str) -> Iterator[str]:
        """Generate streaming response using OpenAI API."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant specialized in answering questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            yield f"Error: {e}"


class LocalProvider(L4LLMProvider):
    """Local model provider using transformers."""
    
    def initialize(self) -> bool:
        """Initialize local model."""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            
            model_name = self.config.llm.model_name
            
            logger.info(f"Loading local model: {model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="auto" if hasattr(self.config.llm, 'use_gpu') and self.config.llm.use_gpu else "cpu"
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                do_sample=True if self.config.llm.temperature > 0 else False
            )
            
            self._initialized = True
            logger.info(f"Initialized local model: {model_name}")
            return True
            
        except ImportError:
            logger.error("Transformers package not installed. Run: pip install transformers torch")
            return False
        except Exception as e:
            logger.error(f"Local model initialization failed: {e}")
            return False
    
    def _generate_sync(self, prompt: str) -> str:
        """Generate response using local model."""
        try:
            # Add instruction formatting for better results
            formatted_prompt = self._format_prompt(prompt)
            
            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                do_sample=True if self.config.llm.temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = outputs[0]["generated_text"]
            
            # Extract only the new part (after the prompt)
            response = generated_text[len(formatted_prompt):].strip()
            
            # Clean up common artifacts
            response = self._clean_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Local generation failed: {e}")
            raise
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for better local model performance."""
        return f"""<|system|>
You are a helpful AI assistant. Answer the question based on the provided context.

<|user|>
{prompt}

<|assistant|>
"""
    
    def format_context(self, episodes: List[Dict[str, Any]]) -> str:
        """Format episodes into context string."""
        if not episodes:
            return ""
        
        context_parts = []
        for i, episode in enumerate(episodes[:10]):  # Limit to 10 most relevant episodes
            text = episode.get('text', str(episode))
            c_value = episode.get('c', 0.5)
            context_parts.append(f"Context {i+1} (relevance: {c_value:.2f}):\n{text}")
        
        return "\n\n".join(context_parts)

    def process(self, input_data) -> Dict[str, Any]:
        """Process input through LLM layer."""
        from ..interfaces import LayerInput, LayerOutput
        
        if isinstance(input_data, LayerInput):
            context = input_data.context or {}
            question = input_data.data
        else:
            # Handle direct dict input for backward compatibility
            context = input_data.get('context', {})
            question = input_data.get('question', str(input_data))
        
        result = self.generate_response(context, question)
        
        if isinstance(input_data, LayerInput):
            return LayerOutput(
                result=result['response'],
                confidence=0.8 if result['success'] else 0.0,
                metadata=result
            )
        else:
            return result
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            if hasattr(self, 'pipeline'):
                del self.pipeline
            
            # Clear GPU memory if using CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
                
            logger.info("LocalProvider cleanup completed")
        except Exception as e:
            logger.warning(f"Error during LocalProvider cleanup: {e}")

    def _clean_response(self, response: str) -> str:
        """Clean up response artifacts."""
        # Remove common stop sequences
        stop_sequences = ["<|", "</s>", "<s>", "[END]", "[STOP]"]
        
        for stop in stop_sequences:
            if stop in response:
                response = response.split(stop)[0]
        
        # Remove excessive whitespace
        response = response.strip()
        
        # Remove incomplete sentences at the end
        sentences = response.split('. ')
        if len(sentences) > 1 and not sentences[-1].endswith('.'):
            response = '. '.join(sentences[:-1]) + '.'
        
        return response


def get_llm_provider(config=None, safe_mode=False) -> L4LLMProvider:
    """Get appropriate LLM provider based on configuration."""
    config = config or get_config()
    
    # Check for safe mode from config or parameter
    use_safe_mode = safe_mode or getattr(config.llm, 'safe_mode', False) or getattr(config, 'environment', 'local') == 'testing'
    
    if use_safe_mode:
        from .mock_llm_provider import MockLLMProvider
        logger.info("Using mock LLM provider for safe operation")
        return MockLLMProvider(config)
    
    provider_type = config.llm.provider.lower()
    
    if provider_type == "openai":
        return OpenAIProvider(config)
    elif provider_type == "local":
        return LocalProvider(config)
    else:
        logger.warning(f"Unknown provider type: {provider_type}, falling back to local")
        return LocalProvider(config)


# Legacy compatibility function
def generate(prompt: str) -> str:
    """Legacy generate function for backward compatibility."""
    try:
        provider = get_llm_provider()
        result = provider.generate_response({}, prompt)
        
        if result['success']:
            return result['response']
        else:
            return "Error generating response."
            
    except Exception as e:
        logger.error(f"Legacy generate failed: {e}")
        return "Error generating response."
