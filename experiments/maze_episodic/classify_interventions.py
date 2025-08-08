#!/usr/bin/env python3
"""
Classify experiments by intervention timing
==========================================

Criteria:
1. Memory/visual info for episode creation: OK
2. Intervention AFTER insight vector generation: CHEAT
"""

import os
import re
import glob

def analyze_intervention_timing(filepath):
    """Analyze where interventions occur in the decision flow"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Key patterns to identify decision flow stages
    insight_patterns = [
        r'insight.*=',
        r'form_insight',
        r'message_pass',
        r'deep_message',
        r'gedig.*evaluate',
        r'evaluate.*action'
    ]
    
    decision_patterns = [
        r'decide_action',
        r'select_action',
        r'choose_action',
        r'action.*=.*max',
        r'np\.random\.choice.*action'
    ]
    
    # Find intervention patterns
    visited_usage = re.findall(r'(visited|self\.visited)[^=]*(?:bonus|penalty|score|if)', content, re.IGNORECASE)
    visual_usage = re.findall(r'(visual|wall).*(?:bonus|penalty|score|if)', content, re.IGNORECASE)
    exploration_bonus = re.findall(r'exploration.*bonus|bonus.*exploration', content, re.IGNORECASE)
    goal_distance_bonus = re.findall(r'goal_dist.*[<>].*(?:score|bonus)', content, re.IGNORECASE)
    
    # Check if interventions occur after insight generation
    post_insight_intervention = False
    
    # Simple heuristic: check if bonus/penalty appears after score calculation
    if re.search(r'score.*\+=.*(?:bonus|penalty|visited|visual)', content, re.IGNORECASE):
        post_insight_intervention = True
    
    # Check for action filtering based on walls
    if re.search(r'action.*=.*wall|wall.*action.*=', content, re.IGNORECASE):
        # This might be OK if it's just filtering invalid actions
        wall_filtering = True
    else:
        wall_filtering = False
    
    return {
        'has_visited': bool(re.search(r'self\.visited', content)),
        'has_visual': bool(re.search(r'visual_memory|visual_info', content)),
        'visited_intervention': len(visited_usage) > 0,
        'visual_intervention': len(visual_usage) > 0,
        'exploration_bonus': len(exploration_bonus) > 0,
        'goal_distance_bonus': len(goal_distance_bonus) > 0,
        'post_insight_intervention': post_insight_intervention,
        'wall_filtering': wall_filtering
    }

def classify_experiments():
    """Classify all experiments"""
    
    # Categories
    pure_memory = []  # No interventions after insight
    memory_with_filtering = []  # Only filters invalid actions
    cheating = []  # Interventions after insight generation
    
    # Analyze maze_episodic
    print("="*80)
    print("EXPERIMENT CLASSIFICATION BY INTERVENTION TIMING")
    print("="*80)
    
    for directory in ['maze_episodic', '../maze-sota-comparison']:
        print(f"\n\nDirectory: {directory}")
        print("-"*60)
        
        pattern = f"{directory}/*.py"
        files = glob.glob(pattern)
        
        for filepath in files:
            filename = os.path.basename(filepath)
            if filename.startswith('test_') or filename.startswith('debug_'):
                continue
                
            analysis = analyze_intervention_timing(filepath)
            
            # Classify based on intervention timing
            if not (analysis['has_visited'] or analysis['has_visual']):
                # No memory tracking at all
                category = "No memory"
            elif analysis['post_insight_intervention'] or analysis['exploration_bonus'] or analysis['goal_distance_bonus']:
                category = "CHEAT (post-insight intervention)"
                cheating.append(filename)
            elif analysis['wall_filtering'] and not analysis['visited_intervention']:
                category = "Memory + wall filtering only"
                memory_with_filtering.append(filename)
            elif analysis['has_visited'] or analysis['has_visual']:
                if not (analysis['visited_intervention'] or analysis['visual_intervention']):
                    category = "Pure memory-based"
                    pure_memory.append(filename)
                else:
                    category = "CHEAT (memory used for bonuses)"
                    cheating.append(filename)
            else:
                category = "Unknown"
            
            print(f"\n{filename}:")
            print(f"  Category: {category}")
            if analysis['has_visited'] or analysis['has_visual']:
                print(f"  - Has visited tracking: {analysis['has_visited']}")
                print(f"  - Has visual memory: {analysis['has_visual']}")
                print(f"  - Visited intervention: {analysis['visited_intervention']}")
                print(f"  - Visual intervention: {analysis['visual_intervention']}")
                print(f"  - Exploration bonus: {analysis['exploration_bonus']}")
                print(f"  - Goal distance bonus: {analysis['goal_distance_bonus']}")
    
    # Summary
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nPure memory-based (no post-insight intervention): {len(pure_memory)}")
    for f in pure_memory[:10]:
        print(f"  - {f}")
        
    print(f"\nMemory with wall filtering only: {len(memory_with_filtering)}")
    for f in memory_with_filtering[:10]:
        print(f"  - {f}")
        
    print(f"\nCHEATING (interventions after insight): {len(cheating)}")
    for f in cheating[:10]:
        print(f"  - {f}")
    if len(cheating) > 10:
        print(f"  ... and {len(cheating)-10} more")

if __name__ == "__main__":
    classify_experiments()