"""TinyLlama wrapper with optional cleanup to avoid memory leaks."""

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from .config import LLM_NAME
import gc

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # torch may not be installed in minimal test env
    torch = None

__all__ = ["generate", "clear_model"]

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

def clear_model():
    """Release cached pipeline to free memory."""
    global _pipe
    if _pipe is not None:
        # remove reference to underlying model to help GC
        try:
            if hasattr(_pipe, "model"):
                del _pipe.model
        except Exception:
            pass
        _pipe = None
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
