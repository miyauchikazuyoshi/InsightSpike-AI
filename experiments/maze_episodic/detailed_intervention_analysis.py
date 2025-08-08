#!/usr/bin/env python3
"""
Detailed analysis of intervention timing
=======================================

Focus on specific implementations to understand intervention flow
"""

import re

def analyze_decision_flow(filepath):
    """Analyze the decision flow in detail"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    analysis = {
        'filename': filepath.split('/')[-1],
        'memory_usage': {'visited': False, 'visual': False},
        'insight_generation': {'method': None, 'line': None},
        'post_insight_interventions': [],
        'action_selection': {'method': None, 'line': None}
    }
    
    for i, line in enumerate(lines):
        # Check memory types
        if 'self.visited' in line:
            analysis['memory_usage']['visited'] = True
        if 'visual_memory' in line or 'visual_info' in line:
            analysis['memory_usage']['visual'] = True
            
        # Find insight generation
        if any(pattern in line for pattern in ['message_pass', 'form_insight', 'evaluate_action', 'gedig']):
            if not analysis['insight_generation']['method']:
                analysis['insight_generation']['method'] = line.strip()
                analysis['insight_generation']['line'] = i + 1
        
        # Find action selection
        if any(pattern in line for pattern in ['decide_action', 'select_action', 'max(', 'argmax']):
            if 'def' not in line:  # Not a function definition
                analysis['action_selection']['method'] = line.strip()
                analysis['action_selection']['line'] = i + 1
        
        # Find post-insight interventions
        if analysis['insight_generation']['line'] and i > analysis['insight_generation']['line']:
            if any(pattern in line for pattern in ['bonus', 'penalty', '+=', '-=', 'if.*visited', 'if.*goal_dist']):
                if 'score' in line or 'action' in line:
                    analysis['post_insight_interventions'].append({
                        'line': i + 1,
                        'code': line.strip()
                    })
    
    return analysis

# Analyze key implementations
implementations = [
    'donut_gedig_navigator_simple.py',
    'pure_episodic_navigator.py',
    'true_gedig_flow_navigator.py',
    'pure_episodic_donut.py',
    '../maze-sota-comparison/sparse_deep_message_50x50.py',
    '../maze-sota-comparison/episodic_2hop_navigator.py'
]

print("="*100)
print("DETAILED INTERVENTION ANALYSIS")
print("="*100)

for impl in implementations:
    try:
        analysis = analyze_decision_flow(impl)
        
        print(f"\n\n{'='*80}")
        print(f"File: {analysis['filename']}")
        print(f"{'='*80}")
        
        print(f"\nMemory Usage:")
        print(f"  - Visited tracking: {analysis['memory_usage']['visited']}")
        print(f"  - Visual memory: {analysis['memory_usage']['visual']}")
        
        print(f"\nInsight Generation:")
        if analysis['insight_generation']['method']:
            print(f"  Line {analysis['insight_generation']['line']}: {analysis['insight_generation']['method']}")
        else:
            print("  Not found")
        
        print(f"\nAction Selection:")
        if analysis['action_selection']['method']:
            print(f"  Line {analysis['action_selection']['line']}: {analysis['action_selection']['method']}")
        else:
            print("  Not found")
        
        print(f"\nPost-Insight Interventions: {len(analysis['post_insight_interventions'])}")
        for intervention in analysis['post_insight_interventions'][:5]:
            print(f"  Line {intervention['line']}: {intervention['code']}")
        
        # Classification
        if not analysis['memory_usage']['visited'] and not analysis['memory_usage']['visual']:
            classification = "NO MEMORY (pure computation)"
        elif len(analysis['post_insight_interventions']) == 0:
            classification = "MEMORY-BASED (no post-insight intervention)"
        else:
            classification = "CHEAT (post-insight intervention)"
        
        print(f"\nCLASSIFICATION: {classification}")
        
    except FileNotFoundError:
        print(f"\n\nFile not found: {impl}")