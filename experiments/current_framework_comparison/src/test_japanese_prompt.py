#!/usr/bin/env python3
"""
Test if using Japanese prompts would work better
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from transformers import pipeline

print("🧪 Testing Language in Prompts")

# Create pipeline
generator = pipeline("text-generation", model="distilgpt2", device=-1)

# Test 1: English prompt
print("\n1️⃣ English prompt:")
en_prompt = "Question: What is energy?\nAnswer:"
result = generator(en_prompt, max_new_tokens=30, temperature=0.7, do_sample=True)
print(f"Prompt: {en_prompt}")
print(f"Response: {result[0]['generated_text'][len(en_prompt):]}")

# Test 2: Japanese prompt (DistilGPT2 won't understand this well)
print("\n2️⃣ Japanese prompt:")
jp_prompt = "質問: エネルギーとは何ですか？\n答え:"
result = generator(jp_prompt, max_new_tokens=30, temperature=0.7, do_sample=True)
print(f"Prompt: {jp_prompt}")
print(f"Response: {result[0]['generated_text'][len(jp_prompt):]}")

# Test 3: Mixed (Japanese content but English structure)
print("\n3️⃣ Mixed content:")
mixed_prompt = "Context: エネルギーは仕事をする能力です。\n\nQuestion: What is energy?\nAnswer:"
result = generator(mixed_prompt, max_new_tokens=30, temperature=0.7, do_sample=True)
print(f"Prompt: {mixed_prompt}")
print(f"Response: {result[0]['generated_text'][len(mixed_prompt):]}")

print("\n💡 Observation: DistilGPT2 is trained on English, so Japanese prompts won't work well")
print("   The framework is passing English prompts, which is correct for this model")
print("   But the model quality is still the limiting factor")

# Test with better English prompt
print("\n4️⃣ Better structured English prompt:")
better_prompt = "Energy is the capacity to do work. Based on this, energy can be defined as"
result = generator(better_prompt, max_new_tokens=30, temperature=0.7, do_sample=True)
print(f"Prompt: {better_prompt}")
print(f"Response: {result[0]['generated_text'][len(better_prompt):]}")