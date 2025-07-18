#!/usr/bin/env python3
"""
Direct DistilGPT2 Test
Tests DistilGPT2 directly on RAT problems
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_distilgpt2_rat():
    """Test DistilGPT2 on RAT problems"""
    logger.info("🚀 Testing DistilGPT2 on RAT problems...")
    
    # Load model
    logger.info("Loading DistilGPT2...")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load knowledge base
    kb_dir = Path(__file__).parent.parent / "data" / "knowledge_base"
    with open(kb_dir / "rat_episodes.json", 'r') as f:
        episodes_data = json.load(f)
        episodes = episodes_data["episodes"]
    
    # Test problems
    problems = [
        {
            "question": "What word associates with COTTAGE, SWISS, and CAKE?",
            "answer": "CHEESE"
        },
        {
            "question": "What word connects CREAM, SKATE, and WATER?", 
            "answer": "ICE"
        },
        {
            "question": "What concept links DUCK, FOLD, and DOLLAR?",
            "answer": "BILL"
        },
        {
            "question": "Find the word that relates to NIGHT, WRIST, and STOP",
            "answer": "WATCH"
        },
        {
            "question": "What connects RIVER, NOTE, and ACCOUNT?",
            "answer": "BANK"
        }
    ]
    
    logger.info(f"\nLoaded {len(episodes)} episodes")
    
    for i, problem in enumerate(problems):
        logger.info(f"\n{'='*60}")
        logger.info(f"Problem {i+1}: {problem['question']}")
        logger.info(f"{'='*60}")
        
        # Find relevant episodes
        relevant = []
        for ep in episodes[:5]:  # Just use first 5
            relevant.append(ep['text'])
        
        # Create prompt
        context = "Knowledge:\n" + "\n".join(f"- {text}" for text in relevant)
        prompt = f"{context}\n\nQuestion: {problem['question']}\nAnswer:"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=20,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        logger.info(f"Response: {response}")
        logger.info(f"Expected: {problem['answer']}")
        
        # Check if answer is in response
        if problem['answer'] in response.upper():
            logger.info("✅ Correct!")
        else:
            logger.info("❌ Wrong")
    
    logger.info("\n✅ Test completed!")


if __name__ == "__main__":
    test_distilgpt2_rat()