"""TinyLlama wrapper"""
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from .config import LLM_NAME

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