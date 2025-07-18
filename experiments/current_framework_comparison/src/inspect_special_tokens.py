#!/usr/bin/env python3
"""
Inspect what special tokens are being generated
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from transformers import AutoTokenizer

print("🔍 Inspecting Special Tokens in DistilGPT2")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

print("\n1️⃣ Tokenizer special tokens:")
print(f"  - pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
print(f"  - eos_token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
print(f"  - bos_token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
print(f"  - unk_token: {tokenizer.unk_token} (id: {tokenizer.unk_token_id})")
print(f"  - All special tokens: {tokenizer.all_special_tokens}")

print("\n2️⃣ Testing InsightSpike's format prompt:")
# This is what Layer4 _format_prompt does
prompt = """<|system|>
You are a helpful AI assistant. Answer the question based on the provided context.

<|user|>
Question: What is energy?

<|assistant|>
"""

tokens = tokenizer.encode(prompt)
print(f"\nOriginal prompt:\n{prompt}")
print(f"\nTokenized (first 20 tokens): {tokens[:20]}")
print(f"Token count: {len(tokens)}")

# Decode each special part
print("\n3️⃣ How each part is tokenized:")
parts = [
    "<|system|>",
    "<|user|>", 
    "<|assistant|>",
    "You are a helpful AI assistant."
]

for part in parts:
    tokens = tokenizer.encode(part, add_special_tokens=False)
    decoded_tokens = [tokenizer.decode([t]) for t in tokens]
    print(f"\n'{part}':")
    print(f"  Token IDs: {tokens}")
    print(f"  Decoded: {decoded_tokens}")

print("\n4️⃣ Testing if DistilGPT2 recognizes these tokens:")
# Check if these are in vocabulary
test_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "[INST]", "[/INST]", "###"]
for token in test_tokens:
    try:
        # Try to encode as a single token
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            print(f"✅ '{token}' is a single token (id: {ids[0]})")
        else:
            print(f"❌ '{token}' is split into {len(ids)} tokens: {ids}")
    except:
        print(f"❌ '{token}' causes encoding error")

print("\n5️⃣ What Layer4 Prompt Builder adds:")
# Simulate what the prompt builder creates
from pathlib import Path
import sys
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Create a sample prompt like Layer4 does
layer4_prompt = """You are an advanced AI assistant specialized in analytical reasoning and insight generation. 

Your role is to:
1. Analyze provided documents and context carefully
2. Identify key patterns, connections, and insights
3. Provide well-reasoned answers based on evidence
4. Acknowledge uncertainty when information is insufficient
5. Highlight novel insights or "spikes of understanding" when they emerge

Always base your responses on the provided context and clearly distinguish between what the evidence supports versus speculative reasoning.

## Retrieved Context Documents
The following documents are relevant to your query:

### Document 1 🟢 (High Confidence)
**Relevance:** 0.900 | **Confidence:** 0.800
Energy is the capacity to do work.

## User Question
"What is energy?"

## Instructions
High reasoning quality detected. Provide a comprehensive, well-structured answer.

Please:
1. Synthesize information from the provided context
2. Highlight key insights and connections
3. Indicate confidence levels in your reasoning
4. Note any novel patterns or 'insight spikes' you detect
5. Provide a clear, actionable answer"""

print(f"\nLayer4 prompt length: {len(layer4_prompt)} characters")
layer4_tokens = tokenizer.encode(layer4_prompt)
print(f"Layer4 token count: {len(layer4_tokens)}")

print("\n💡 Summary:")
print("  - DistilGPT2 doesn't recognize <|system|>, <|user|>, <|assistant|> as special tokens")
print("  - These get tokenized as regular text, confusing the model")
print("  - The Layer4 prompt is very long and complex")
print("  - This explains why responses often include parts of the prompt!")