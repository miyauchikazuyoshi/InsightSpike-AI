#!/usr/bin/env python3
"""
Analyze Q&A Alignment in Episodes
=================================

Find which episodes contain answers to specific questions.
"""

import json
from pathlib import Path

def analyze_qa_alignment():
    """Analyze how questions and answers are aligned in episodes."""
    
    data_dir = Path(__file__).parent / "data"
    episodes_file = data_dir / "episodes.json"
    
    # Load episodes
    with open(episodes_file, 'r') as f:
        episodes = json.load(f)
    
    print(f"Total episodes: {len(episodes)}")
    
    # Test questions with expected answers
    test_qa = [
        {
            "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
            "answer": "Saint Bernadette Soubirous",
            "keywords": ["Bernadette", "Soubirous", "1858", "Lourdes"]
        },
        {
            "question": "What is in front of the Notre Dame Main Building?",
            "answer": "a copper statue of Christ",
            "keywords": ["copper statue", "Christ", "front", "Main Building"]
        },
        {
            "question": "What sits on top of the Main Building at Notre Dame?",
            "answer": "a golden statue of the Virgin Mary",
            "keywords": ["golden statue", "Virgin Mary", "top", "Main Building", "gold dome"]
        },
        {
            "question": "What is the Grotto at Notre Dame?",
            "answer": "a Marian place of prayer and reflection",
            "keywords": ["Grotto", "Marian", "prayer", "reflection", "replica"]
        },
        {
            "question": "When did the Scholastic Magazine of Notre dame begin publishing?",
            "answer": "Unknown from current episodes",
            "keywords": ["Scholastic", "Magazine", "publishing", "begin"]
        }
    ]
    
    print("\n=== Searching for Answers in Episodes ===\n")
    
    for qa in test_qa:
        print(f"Q: {qa['question']}")
        print(f"Expected: {qa['answer']}")
        
        found_episodes = []
        
        # Search through episodes
        for i, episode in enumerate(episodes):
            text = episode.get('text', '').lower()
            
            # Check if keywords appear in episode
            keyword_matches = sum(1 for kw in qa['keywords'] if kw.lower() in text)
            
            if keyword_matches >= len(qa['keywords']) * 0.6:  # 60% match threshold
                found_episodes.append({
                    'idx': i,
                    'matches': keyword_matches,
                    'preview': text[:200] + '...' if len(text) > 200 else text
                })
        
        if found_episodes:
            print(f"✓ Found in {len(found_episodes)} episodes:")
            best_match = max(found_episodes, key=lambda x: x['matches'])
            print(f"  Best match: Episode {best_match['idx']} ({best_match['matches']} keywords)")
            
            # Extract answer from episode
            episode_text = episodes[best_match['idx']]['text']
            for answer_keyword in qa['answer'].split():
                if answer_keyword.lower() in episode_text.lower():
                    # Find sentence containing answer
                    sentences = episode_text.split('.')
                    for sent in sentences:
                        if answer_keyword.lower() in sent.lower():
                            print(f"  Answer context: ...{sent.strip()}...")
                            break
                    break
        else:
            print("✗ Not found in episodes")
        
        print()
    
    # Analyze episode structure
    print("\n=== Episode Structure Analysis ===")
    
    # Check for repeated content
    unique_texts = set()
    repeated_count = 0
    
    for i, episode in enumerate(episodes[:50]):  # Check first 50
        text = episode.get('text', '')
        if text in unique_texts:
            repeated_count += 1
        else:
            unique_texts.add(text)
    
    print(f"Unique texts in first 50 episodes: {len(unique_texts)}")
    print(f"Repeated texts: {repeated_count}")
    
    # Check episode 0 structure
    print("\n=== Episode 0 Analysis ===")
    ep0_text = episodes[0]['text']
    print(f"Length: {len(ep0_text)} characters")
    
    # Count repetitions
    base_text = "Architecturally, the school has a Catholic character"
    repetitions = ep0_text.count(base_text)
    print(f"Text repeated {repetitions} times")
    
    # Extract unique content
    parts = ep0_text.split(' | ')
    print(f"Parts separated by |: {len(parts)}")
    
    if parts[0] == parts[1]:
        print("✓ Parts are identical - this is integrated content")
    
    # Show Q&A recommendation
    print("\n=== Recommendation ===")
    print("Episode 0 contains answers to:")
    print("- Virgin Mary appeared to Saint Bernadette Soubirous")
    print("- Copper statue of Christ in front of Main Building")
    print("- Golden statue of Virgin Mary on top of Main Building")
    print("- Grotto is a Marian place of prayer and reflection")
    
    return episodes


if __name__ == "__main__":
    episodes = analyze_qa_alignment()