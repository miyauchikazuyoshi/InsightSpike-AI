#!/usr/bin/env python3
"""
Demonstrate the special token problem
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import pipeline

print("🔍 Demonstrating Special Token Problem")

generator = pipeline("text-generation", model="distilgpt2", device=-1)

# Test 1: With special tokens (problematic)
print("\n❌ With special tokens:")
bad_prompt = """<|system|>
You are a helpful AI assistant.

<|user|>
What is energy?

<|assistant|>
"""

result = generator(bad_prompt, max_new_tokens=30, temperature=0.7)
response = result[0]["generated_text"]
print(f"Prompt: {bad_prompt[:50]}...")
print(f"Full response: {response[:150]}...")
print(f"Extracted response: {response[len(bad_prompt):].strip()[:100]}...")

# Test 2: Without special tokens (better)
print("\n✅ Without special tokens:")
good_prompt = "Question: What is energy?\nAnswer:"

result = generator(good_prompt, max_new_tokens=30, temperature=0.7)
response = result[0]["generated_text"]
print(f"Prompt: {good_prompt}")
print(f"Full response: {response}")
print(f"Extracted response: {response[len(good_prompt):].strip()}")

# Test 3: Show how model sees the special tokens
print("\n🤖 How DistilGPT2 interprets special tokens:")
interpret_prompt = "<|system|> means"
result = generator(interpret_prompt, max_new_tokens=20, temperature=0.7)
print(f"Input: '{interpret_prompt}'")
print(f"Model continues: '{result[0]['generated_text'][len(interpret_prompt):].strip()}'")

print("\n💡 Conclusion:")
print("  - Special tokens like <|system|> are not recognized by DistilGPT2")
print("  - They become part of the text, confusing the model")
print("  - Simple prompts without special tokens work much better!")