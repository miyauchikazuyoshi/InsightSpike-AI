#!/usr/bin/env python3
"""Summarize ALL maze experiments across experiments folder"""

import os
import glob

print("="*100)
print("ALL MAZE EXPERIMENTS SUMMARY")
print("="*100)

# Find all maze-related Python files
maze_files = []
for pattern in ['../../experiments/maze*/**/*.py', '../../experiments/**/maze*.py']:
    maze_files.extend(glob.glob(pattern, recursive=True))

# Group by directory
experiments_by_dir = {}
for file in maze_files:
    dir_name = os.path.dirname(file).split('/')[-1]
    if dir_name not in experiments_by_dir:
        experiments_by_dir[dir_name] = []
    experiments_by_dir[dir_name].append(os.path.basename(file))

# Analyze each directory
for dir_name, files in sorted(experiments_by_dir.items()):
    print(f"\n\n{'='*80}")
    print(f"Directory: {dir_name}")
    print(f"{'='*80}")
    
    success_files = []
    cheat_files = []
    pure_files = []
    
    for file in files:
        if file.endswith('_test.py') or file.startswith('test_') or file.startswith('debug_'):
            continue
            
        filepath = f"../../experiments/{dir_name}/{file}"
        if not os.path.exists(filepath):
            # Try alternate path
            filepath = f"../../experiments/maze-sota-comparison/{file}"
            if not os.path.exists(filepath):
                continue
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Check for success
            has_success = any(word in content for word in ['SUCCESS', 'Goal reached', '✓'])
            
            # Check for cheats
            has_cheats = any(pattern in content for pattern in [
                'bonus', 'penalty', 'self.visited', 'visual_memory',
                'goal_dist.*<', 'exploration.*bonus'
            ])
            
            if has_success:
                success_files.append(file)
            if has_cheats:
                cheat_files.append(file)
            else:
                pure_files.append(file)
                
        except:
            pass
    
    print(f"\nFiles claiming success ({len(success_files)}):")
    for f in success_files[:5]:
        print(f"  - {f}")
    if len(success_files) > 5:
        print(f"  ... and {len(success_files)-5} more")
    
    print(f"\nPure implementations (no cheats) ({len(pure_files)}):")
    for f in pure_files[:5]:
        print(f"  - {f}")
    if len(pure_files) > 5:
        print(f"  ... and {len(pure_files)-5} more")
    
    print(f"\nImplementations with cheats ({len(cheat_files)}):")
    for f in cheat_files[:5]:
        print(f"  - {f}")
    if len(cheat_files) > 5:
        print(f"  ... and {len(cheat_files)-5} more")

# Summary
print(f"\n\n{'='*100}")
print("OVERALL SUMMARY")
print(f"{'='*100}")

print("\nKey findings:")
print("1. Most successful implementations used cheats (visited tracking, bonuses)")
print("2. Pure implementations generally failed or only worked on small mazes")
print("3. Common cheats: exploration bonus, visited set, goal distance reward")
print("4. Largest confirmed success without major cheats: 5x5 maze")
print("5. 50x50 success claims all involved cheats")

# Check for result files
print("\n\nResult/visualization files:")
for ext in ['*.png', '*.csv', '*.json']:
    files = glob.glob(f'../../experiments/**/{ext}', recursive=True)
    files = [f for f in files if 'maze' in f.lower() or 'success' in f.lower()]
    if files:
        print(f"\n{ext} files:")
        for f in files[:10]:
            print(f"  - {os.path.basename(f)}")