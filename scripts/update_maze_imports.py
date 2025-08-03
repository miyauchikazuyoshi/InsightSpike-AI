#!/usr/bin/env python3
"""
Update import paths for maze experimental modules.
"""

import os
import re
from pathlib import Path

def update_imports_in_file(filepath):
    """Update imports in a single file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Update navigator imports
    patterns = [
        (r'from insightspike\.navigators', 'from insightspike.maze_experimental.navigators'),
        (r'from src\.insightspike\.navigators', 'from src.insightspike.maze_experimental.navigators'),
        (r'import insightspike\.navigators', 'import insightspike.maze_experimental.navigators'),
    ]
    
    for old_pattern, new_pattern in patterns:
        content = re.sub(old_pattern, new_pattern, content)
    
    # Update query imports (if any)
    query_patterns = [
        (r'from insightspike\.query\.sphere_search', 'from insightspike.maze_experimental.query.sphere_search'),
        (r'from insightspike\.query import', 'from insightspike.maze_experimental.query import'),
    ]
    
    for old_pattern, new_pattern in query_patterns:
        content = re.sub(old_pattern, new_pattern, content)
    
    # Update maze_config imports
    content = re.sub(
        r'from insightspike\.config\.maze_config',
        'from insightspike.maze_experimental.maze_config',
        content
    )
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated: {filepath}")
        return True
    return False

def main():
    """Main function to update all Python files."""
    project_root = Path(__file__).parent.parent
    experiments_dir = project_root / "experiments"
    
    updated_files = []
    
    # Find all Python files in experiments directory
    for py_file in experiments_dir.rglob("*.py"):
        if update_imports_in_file(py_file):
            updated_files.append(py_file)
    
    print(f"\nTotal files updated: {len(updated_files)}")
    
    if updated_files:
        print("\nUpdated files:")
        for f in updated_files:
            print(f"  - {f.relative_to(project_root)}")

if __name__ == "__main__":
    main()