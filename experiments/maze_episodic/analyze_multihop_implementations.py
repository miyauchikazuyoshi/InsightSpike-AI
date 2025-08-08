#!/usr/bin/env python3
"""
Analyze multi-hop implementations for cheats
===========================================

Focus on how multi-hop exploration is implemented
"""

import glob
import re

def analyze_multihop_implementation(filepath):
    """Analyze how multi-hop is implemented"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    analysis = {
        'filename': filepath.split('/')[-1],
        'has_multihop': False,
        'multihop_method': None,
        'uses_visited': False,
        'uses_bonus': False,
        'intervention_details': []
    }
    
    # Find multi-hop patterns
    multihop_patterns = [
        r'[1-3][\s\-_]*hop',
        r'n_hop',
        r'multi.*hop',
        r'message.*pass.*round',
        r'for.*hop.*in.*range'
    ]
    
    for pattern in multihop_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            analysis['has_multihop'] = True
            analysis['multihop_method'] = matches[0]
            break
    
    # Check for visited usage in scoring
    visited_in_scoring = re.findall(r'if.*visited.*:.*\n.*(?:score|bonus|penalty)', content, re.IGNORECASE)
    if visited_in_scoring:
        analysis['uses_visited'] = True
        analysis['intervention_details'].append("Visited check in scoring")
    
    # Check for exploration bonus
    bonus_patterns = re.findall(r'(?:exploration|visit).*(?:bonus|penalty)|(?:bonus|penalty).*(?:exploration|visit)', content, re.IGNORECASE)
    if bonus_patterns:
        analysis['uses_bonus'] = True
        analysis['intervention_details'].extend(bonus_patterns)
    
    # Check for hop-specific bonuses
    hop_bonus = re.findall(r'hop.*\*.*[0-9]|n_hop.*\*.*[0-9]', content, re.IGNORECASE)
    if hop_bonus:
        analysis['uses_bonus'] = True
        analysis['intervention_details'].append(f"Hop-based bonus: {hop_bonus[0]}")
    
    return analysis

# Find all multi-hop implementations
print("="*80)
print("MULTI-HOP IMPLEMENTATION ANALYSIS")
print("="*80)

multihop_files = []
for pattern in ['*.py', '../maze-sota-comparison/*.py']:
    files = glob.glob(pattern)
    for f in files:
        try:
            with open(f, 'r') as file:
                if 'hop' in file.read().lower():
                    multihop_files.append(f)
        except:
            pass

# Categorize
pure_multihop = []
cheating_multihop = []

for filepath in multihop_files:
    if any(skip in filepath for skip in ['test_', 'debug_', 'analyze_', 'classify_']):
        continue
        
    analysis = analyze_multihop_implementation(filepath)
    
    if analysis['has_multihop']:
        print(f"\n{analysis['filename']}:")
        print(f"  Multi-hop method: {analysis['multihop_method']}")
        print(f"  Uses visited in scoring: {analysis['uses_visited']}")
        print(f"  Uses bonus/penalty: {analysis['uses_bonus']}")
        
        if analysis['intervention_details']:
            print(f"  Interventions:")
            for detail in analysis['intervention_details'][:3]:
                print(f"    - {detail}")
        
        # Classify
        if analysis['uses_visited'] or analysis['uses_bonus']:
            cheating_multihop.append(analysis['filename'])
            print(f"  CLASSIFICATION: CHEAT")
        else:
            pure_multihop.append(analysis['filename'])
            print(f"  CLASSIFICATION: PURE")

# Summary
print("\n\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nPure multi-hop implementations: {len(pure_multihop)}")
for f in pure_multihop:
    print(f"  - {f}")

print(f"\nMulti-hop with cheats: {len(cheating_multihop)}")
for f in cheating_multihop[:10]:
    print(f"  - {f}")
if len(cheating_multihop) > 10:
    print(f"  ... and {len(cheating_multihop)-10} more")

# Check if any pure multi-hop succeeded
print("\n\nChecking success of pure multi-hop implementations...")
for filename in pure_multihop:
    try:
        with open(filename, 'r') as f:
            content = f.read()
            if any(word in content for word in ['SUCCESS', 'Goal reached', '✓']):
                print(f"\n{filename} claims success!")
                # Check what size maze
                size_match = re.search(r'(\d+)x\1', content)
                if size_match:
                    print(f"  Maze size: {size_match.group(1)}x{size_match.group(1)}")
    except:
        pass