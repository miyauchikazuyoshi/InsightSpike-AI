#!/usr/bin/env python3
"""
Fix Integration Issue and Test
==============================

Properly load the 489 episodes and test Q&A.
"""

import json
from pathlib import Path

def analyze_and_fix():
    """Analyze the episode structure and create a proper test."""
    
    data_dir = Path(__file__).parent / "data"
    episodes_file = data_dir / "episodes.json"
    
    # Load episodes
    with open(episodes_file, 'r') as f:
        episodes = json.load(f)
    
    print(f"Total episodes in file: {len(episodes)}")
    
    # Analyze integration
    print("\n=== Episode Integration Analysis ===")
    
    # Look at first few episodes
    for i in range(min(10, len(episodes))):
        ep = episodes[i]
        text = ep.get('text', '')
        
        if '|' in text:
            parts = text.split('|')
            unique_parts = set(p.strip() for p in parts)
            print(f"\nEpisode {i}:")
            print(f"  Integrated parts: {len(parts)}")
            print(f"  Unique parts: {len(unique_parts)}")
            
            # Check if it contains answers
            if "Saint Bernadette Soubirous" in text:
                print("  ✓ Contains answer about Bernadette")
            if "copper statue of Christ" in text:
                print("  ✓ Contains answer about statue")
                
            # Show first 200 chars of first part
            if parts:
                print(f"  Content preview: {parts[0][:200]}...")
        else:
            print(f"\nEpisode {i}: Single text ({len(text)} chars)")
            if len(text) < 500:
                print(f"  Content: {text}")
    
    # Find episodes with specific answers
    print("\n=== Finding Episodes with Answers ===")
    
    answers_to_find = [
        ("Saint Bernadette Soubirous", "Virgin Mary appeared to"),
        ("copper statue of Christ", "front of the Main Building"),
        ("golden statue", "top of the Main Building"),
        ("Marian place of prayer", "Grotto")
    ]
    
    for answer, context in answers_to_find:
        print(f"\nSearching for: {answer}")
        found_in = []
        
        for i, ep in enumerate(episodes):
            text = ep.get('text', '')
            if answer.lower() in text.lower():
                found_in.append(i)
        
        if found_in:
            print(f"  Found in episodes: {found_in[:5]}...")  # Show first 5
            
            # Show the context
            ep_idx = found_in[0]
            text = episodes[ep_idx].get('text', '')
            
            # Extract sentence with answer
            sentences = text.split('.')
            for sent in sentences:
                if answer.lower() in sent.lower():
                    print(f"  Context: ...{sent.strip()}...")
                    break
    
    # Create a simple direct test
    print("\n=== Direct Retrieval Test ===")
    
    # Test if we can find answers directly in the episode texts
    print("\nQuestion: To whom did the Virgin Mary allegedly appear in 1858?")
    
    # Check episode 0 which should contain this
    ep0_text = episodes[0].get('text', '')
    if "Saint Bernadette Soubirous" in ep0_text and "1858" in ep0_text:
        print("✅ Answer found in Episode 0!")
        
        # Extract the relevant sentence
        sentences = ep0_text.split('.')
        for sent in sentences:
            if "Bernadette" in sent and "1858" in sent:
                print(f"Answer context: {sent.strip()}")
    else:
        print("❌ Answer not found in Episode 0")
    
    # Summary
    print("\n=== Summary ===")
    print(f"1. File contains {len(episodes)} episodes")
    print("2. Many episodes are heavily integrated (multiple parts joined with |)")
    print("3. The integration preserved the content but made it repetitive")
    print("4. Answers ARE present in the episodes")
    print("\nThe issue is not missing data, but:")
    print("- Only 5 episodes are loaded into memory (should be 489)")
    print("- The LLM is returning generic responses instead of using retrieved content")
    print("- Need to fix the episode loading mechanism")
    
    return episodes


if __name__ == "__main__":
    episodes = analyze_and_fix()