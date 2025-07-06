#!/usr/bin/env python3
"""
Extract Real SQuAD Questions
============================

Extract actual questions from the SQuAD dataset we processed.
"""

import json
from pathlib import Path

def extract_questions():
    """Extract questions from our processed SQuAD data."""
    
    # Check what's in the episodes
    data_dir = Path(__file__).parent / "data"
    episodes_file = data_dir / "episodes.json"
    
    print("=== Extracting SQuAD Questions ===\n")
    
    if not episodes_file.exists():
        print("Episodes file not found!")
        return []
    
    # Load episodes
    with open(episodes_file, 'r') as f:
        episodes = json.load(f)
    
    print(f"Total episodes: {len(episodes)}")
    
    # Extract questions and contexts
    questions = []
    contexts = []
    
    for i, episode in enumerate(episodes[:100]):  # Check first 100
        text = episode.get('text', '')
        
        # Identify if it's a question (usually shorter and ends with ?)
        if '?' in text and len(text) < 200:
            questions.append({
                'episode_idx': i,
                'question': text,
                'length': len(text)
            })
        elif len(text) > 200:  # Likely a context
            contexts.append({
                'episode_idx': i,
                'preview': text[:100] + '...',
                'length': len(text)
            })
    
    print(f"\nFound {len(questions)} questions")
    print(f"Found {len(contexts)} contexts")
    
    # Show sample questions
    print("\n=== Sample Questions ===")
    for q in questions[:10]:
        print(f"[{q['episode_idx']}] {q['question']}")
    
    # Show sample contexts
    print("\n=== Sample Contexts ===")
    for c in contexts[:5]:
        print(f"[{c['episode_idx']}] {c['preview']}")
    
    # Try to pair questions with nearby contexts
    print("\n=== Question-Context Pairs ===")
    pairs = []
    
    for q in questions[:20]:
        q_idx = q['episode_idx']
        # Look for context before or after
        for c in contexts:
            c_idx = c['episode_idx']
            if abs(c_idx - q_idx) <= 2:  # Within 2 episodes
                pairs.append({
                    'question': q['question'],
                    'question_idx': q_idx,
                    'context_idx': c_idx,
                    'distance': abs(c_idx - q_idx)
                })
                break
    
    print(f"\nFound {len(pairs)} potential Q-A pairs")
    
    # Save extracted questions
    output = {
        'total_episodes': len(episodes),
        'questions_found': len(questions),
        'contexts_found': len(contexts),
        'sample_questions': questions[:50],
        'sample_pairs': pairs[:20]
    }
    
    output_file = Path(__file__).parent / "extracted_squad_questions.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Saved to {output_file}")
    
    return questions


if __name__ == "__main__":
    questions = extract_questions()