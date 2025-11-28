#!/usr/bin/env python3
import os
import time
import importlib.util
import pytest

def _has_torch():
    try:
        return importlib.util.find_spec("torch") is not None
    except Exception:
        return False

if not _has_torch():
    pytest.skip("torch/transformers stack not available in this environment", allow_module_level=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Testing DistilGPT2 loading...", flush=True)

start = time.time()
print("1. Importing transformers...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"   ✓ Import done ({time.time()-start:.2f}s)", flush=True)

print("2. Loading tokenizer...", flush=True)
start = time.time()
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
print(f"   ✓ Tokenizer loaded ({time.time()-start:.2f}s)", flush=True)

print("3. Loading model...", flush=True)
start = time.time()
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
print(f"   ✓ Model loaded ({time.time()-start:.2f}s)", flush=True)

print("4. Testing generation...", flush=True)
start = time.time()
inputs = tokenizer("Energy is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"   ✓ Generated: '{text}' ({time.time()-start:.2f}s)", flush=True)

print("\n✅ DistilGPT2 works fine!", flush=True)
