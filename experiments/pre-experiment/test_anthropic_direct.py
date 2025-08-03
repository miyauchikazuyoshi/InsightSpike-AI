"""
Test Anthropic API directly
"""

import os
from anthropic import Anthropic

# Set API key
os.environ["ANTHROPIC_API_KEY"] = "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"

# Create client
client = Anthropic()

# Test simple message
message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=100,
    messages=[
        {"role": "user", "content": "What is 2 + 2? Please answer with just the number."}
    ]
)

print("Response:", message.content[0].text)

# Test with context
message2 = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=200,
    messages=[
        {"role": "user", "content": """Context: Apples are fruits that grow on trees. They come in red, green, and yellow colors.

Question: What colors can apples be?

Answer:"""}
    ]
)

print("\nResponse with context:", message2.content[0].text)