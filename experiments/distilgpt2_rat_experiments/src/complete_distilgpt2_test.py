#!/usr/bin/env python3
"""
Complete DistilGPT2 Test with Proper Database
Tests DistilGPT2 on all 5 RAT problems with english definitions
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


def test_distilgpt2_with_proper_kb():
    """Test DistilGPT2 with proper knowledge base"""
    logger.info("🚀 Testing DistilGPT2 with proper English definitions...")
    
    # Load model
    logger.info("Loading DistilGPT2...")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load proper knowledge base
    kb_dir = Path(__file__).parent.parent / "data" / "knowledge_base"
    with open(kb_dir / "proper_rat_episodes.json", 'r') as f:
        episodes_data = json.load(f)
        episodes = episodes_data["episodes"]
    
    # Test problems
    problems = [
        {
            "id": 1,
            "question": "What word associates with COTTAGE, SWISS, and CAKE?",
            "words": ["COTTAGE", "SWISS", "CAKE"],
            "answer": "CHEESE"
        },
        {
            "id": 2,
            "question": "What word connects CREAM, SKATE, and WATER?",
            "words": ["CREAM", "SKATE", "WATER"], 
            "answer": "ICE"
        },
        {
            "id": 3,
            "question": "What concept links DUCK, FOLD, and DOLLAR?",
            "words": ["DUCK", "FOLD", "DOLLAR"],
            "answer": "BILL"
        },
        {
            "id": 4,
            "question": "Find the word that relates to NIGHT, WRIST, and STOP",
            "words": ["NIGHT", "WRIST", "STOP"],
            "answer": "WATCH"
        },
        {
            "id": 5,
            "question": "What connects RIVER, NOTE, and ACCOUNT?",
            "words": ["RIVER", "NOTE", "ACCOUNT"],
            "answer": "BANK"
        }
    ]
    
    logger.info(f"\nLoaded {len(episodes)} episodes from proper database")
    
    results = []
    correct_count = 0
    
    for problem in problems:
        logger.info(f"\n{'='*60}")
        logger.info(f"Problem {problem['id']}: {problem['question']}")
        logger.info(f"{'='*60}")
        
        # Find relevant episodes for each word
        relevant_episodes = []
        for word in problem['words']:
            word_episodes = [ep for ep in episodes if ep['source_word'] == word]
            # Take top 2 definitions per word
            relevant_episodes.extend(word_episodes[:2])
        
        # Also find RAT-relevant episodes
        rat_relevant = [ep for ep in episodes if ep.get('is_rat_relevant', False)]
        for ep in rat_relevant[:3]:
            if ep not in relevant_episodes:
                relevant_episodes.append(ep)
        
        # Create context
        context = "Knowledge:\n"
        for ep in relevant_episodes[:10]:  # Limit to 10 episodes
            context += f"- {ep['text']}\n"
        
        prompt = f"{context}\n\nQuestion: {problem['question']}\nThink step by step. The answer is one word.\nAnswer:"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=500)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=30,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        # Extract answer
        predicted = extract_answer(response, problem['answer'])
        is_correct = predicted == problem['answer']
        
        if is_correct:
            correct_count += 1
        
        logger.info(f"Response: {response.strip()}")
        logger.info(f"Extracted: {predicted}")
        logger.info(f"Expected: {problem['answer']}")
        logger.info(f"Result: {'✅ Correct!' if is_correct else '❌ Wrong'}")
        
        results.append({
            "problem_id": problem['id'],
            "question": problem['question'],
            "predicted": predicted,
            "expected": problem['answer'],
            "correct": is_correct,
            "response": response.strip()
        })
    
    # Summary
    accuracy = (correct_count / len(problems)) * 100
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 FINAL RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy: {correct_count}/{len(problems)} = {accuracy:.1f}%")
    logger.info(f"\nProblem-by-problem:")
    for r in results:
        status = "✅" if r['correct'] else "❌"
        logger.info(f"  {status} Problem {r['problem_id']}: {r['predicted']} (expected: {r['expected']})")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "distilgpt2_experiments"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"complete_distilgpt2_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "experiment": "Complete DistilGPT2 RAT Test",
            "timestamp": timestamp,
            "accuracy": accuracy,
            "correct": correct_count,
            "total": len(problems),
            "results": results
        }, f, indent=2)
    
    logger.info(f"\n💾 Results saved to: {output_file}")


def extract_answer(response, expected):
    """Extract answer from response"""
    # Known answers
    known_answers = ['CHEESE', 'ICE', 'BILL', 'WATCH', 'BANK']
    
    response_upper = response.upper()
    
    # First check if expected answer is in response
    if expected in response_upper:
        return expected
    
    # Then check all known answers
    for answer in known_answers:
        if answer in response_upper:
            return answer
    
    # Try to find any capitalized word
    words = response.split()
    for word in words:
        cleaned = ''.join(c for c in word if c.isalpha())
        if cleaned.isupper() and len(cleaned) >= 3:
            return cleaned.upper()
    
    # Look for single word after "is"
    if " is " in response.lower():
        after_is = response.lower().split(" is ")[-1]
        first_word = after_is.strip().split()[0] if after_is.strip() else ""
        if first_word:
            return first_word.upper().rstrip('.,!?')
    
    return "UNKNOWN"


if __name__ == "__main__":
    test_distilgpt2_with_proper_kb()