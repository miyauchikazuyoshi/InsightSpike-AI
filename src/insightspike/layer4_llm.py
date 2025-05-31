"""TinyLlama wrapper"""
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Import from the legacy config.py file using the same pattern as __init__.py
import os
import importlib.util

try:
    # Import from the legacy config.py file explicitly 
    _config_file = os.path.join(os.path.dirname(__file__), 'config.py')
    _spec = importlib.util.spec_from_file_location("legacy_config", _config_file)
    _config = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_config)
    LLM_NAME = _config.LLM_NAME
except (ImportError, AttributeError):
    # Fallback for testing or if config is not available
    LLM_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

__all__ = ["generate"]

_pipe = None

def _init():
    global _pipe
    if _pipe is None:
        tok  = AutoTokenizer.from_pretrained(LLM_NAME, trust_remote_code=True)
        mdl  = AutoModelForCausalLM.from_pretrained(LLM_NAME, trust_remote_code=True)
        _pipe = pipeline("text-generation", model=mdl, tokenizer=tok, max_new_tokens=256)
    return _pipe

def generate(prompt: str) -> str:
    pipe = _init()
    out  = pipe(prompt, do_sample=False)[0]["generated_text"]
    return out.split("[Answer]")[-1].strip()
